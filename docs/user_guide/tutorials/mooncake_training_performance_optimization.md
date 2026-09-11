# Mooncake 训练性能优化实施顺序

本文针对 Ascend Direct + Mooncake proxy 训练链路，给出优化项的实施顺序、依赖关系和验收标准。目标是提升有效训练吞吐（tokens/s），同时避免把数据一致性问题误判为性能问题。

## 结论

推荐按以下顺序实施：

1. 建立可重复的性能基线。
2. 消除 hidden states 的 NPU -> CPU -> NPU 往返。
3. 重新设计 proxy 多卡 batch 分发，使不同 rank 处理不同 batch。
4. 去掉 proxy rank 的无效 placeholder batch 和重复拷贝。
5. 增加训练进程内的异步预取。
6. 降低生产端和消费端的全量 CRC 校验频率。

“每步日志导致的同步”不在本文实施范围内，按当前要求暂不处理。实现其他优化时，基线测试应保持固定的 `log_freq`，避免测量口径变化。

## 为什么按这个顺序

### 阶段 0：建立基线

先使用 `scripts/benchmark.py` 或真实训练记录以下指标：

- `tokens_per_s`
- `step_ms`
- `fetch_ms`
- `fwd_ms`
- `bwd_ms`
- `opt_ms`
- `fetch_frac`
- NPU 显存峰值

至少记录三种场景：单卡、proxy 多卡、无 Mooncake 的本地数据基线。每种场景预热后再统计，固定模型、序列长度、数据集、batch token budget 和随机种子。

没有这一步，无法区分训练计算瓶颈、Mooncake 传输瓶颈和多卡同步瓶颈。

### 阶段 1：消除 NPU -> CPU -> NPU 往返

当前 consumer 大致经过：

```text
Mooncake DMA -> NPU ADXL slot -> CPU tensor -> CPU collate/noise -> NPU batch
```

优先增加面向目标设备的读取接口，例如 `get_sample_into(device=...)`，或让 `get_sample()` 返回可直接用于训练的 NPU tensor。hidden states 应尽量保持在 NPU 上完成：

```text
Mooncake DMA -> NPU buffer -> NPU collate/noise -> model
```

token ids 等小型整型张量可以继续走 `get_tensor()`。不要为了消除一次拷贝而改变当前浮点 hidden states 的 ADXL 写入路径。

验收标准：

- 单卡 `fetch_ms` 明显下降；
- NPU -> CPU 和 CPU -> NPU 大拷贝次数下降；
- checksum、shape、dtype 和 token id 校验保持不变；
- 与本地数据基线相比，训练 loss 和有效 batch 数一致。

这是单卡和多卡都有效、且对训练吞吐收益最高的优化。

### 阶段 2：proxy 多卡使用不同 batch

当前 proxy 模式由 rank 0 生成一个 batch，再广播给所有 rank。这样多卡重复计算同一批样本，增加了通信开销，却没有增加有效数据吞吐。

建议改为：

1. rank 0 一次读取与 DP world size 相匹配的多个 batch；
2. 为每个 rank 准备不同样本；
3. 使用 `scatter` 或等价的分发方式发送到各 rank；
4. 保持每个 rank 的 token budget 和 sampler 语义一致。

该阶段必须和 sampler、梯度同步、epoch 长度以及断点恢复一起设计，不能只把 `broadcast` 改成 `scatter`。如果每个 rank 的 batch shape 不一致，应使用长度元数据或固定 batch buffer，避免每步协商大量 shape 信息。

验收标准：

- 多卡有效 tokens/s 随卡数增加；
- 每个 rank 的样本 key 不重复；
- global step、epoch 长度和 checkpoint resume 语义不变；
- 梯度同步后模型结果与单卡/非 proxy 参考结果在允许误差内一致。

这是多卡吞吐的核心结构性优化，预期收益高，但改动风险也最高。

### 阶段 3：去掉 proxy rank 的 placeholder batch

当前非零 rank 的 dataset 返回 `None`，`CollateFn` 仍会创建完整长度的零 batch；之后 `_sync_batch()` 先把它搬到 NPU，再用广播结果覆盖。这是无效的显存分配和数据搬运。

应让 proxy rank：

- 不创建完整 CPU hidden states placeholder；
- 根据已协商的 batch schema 直接在目标 NPU 上分配接收 buffer；
- 直接接收 rank 0 的 payload；
- 只保留必要的 metadata 和小型控制张量。

该阶段最好与阶段 2 一起实现，因为 batch 分发协议一旦改为不同 batch，placeholder 的生命周期和 buffer 复用方式都需要重新确定。

验收标准：

- 非零 rank 不再产生完整 CPU 零 hidden states；
- proxy rank 的 CPU -> NPU 无效拷贝消失；
- NPU 显存峰值下降或至少不随 proxy rank 数量线性增加；
- batch schema 不一致时仍能给出明确错误。

### 阶段 4：增加异步预取

proxy 模式当前将 DataLoader worker 数设为 0。Mooncake 读取、完整性校验、noise、collate 和训练 step 可能串行执行。

建议在训练进程内增加受控预取队列，而不是直接恢复多个 DataLoader 进程：

```text
prefetch thread: Mooncake read -> device staging -> collate
training thread: current batch forward/backward
```

建议从队列深度 1 或 2 开始，避免一次性占用过多 NPU 显存。预取线程必须继承并恢复正确的 NPU/ACL context；否则会重新引入 ADXL context 错误。

验收标准：

- `fetch_frac` 下降；
- 训练线程等待下一 batch 的时间下降；
- 队列不会因 producer 速度变化无限增长；
- 发生 Mooncake 超时或错误时，异常能传回训练主线程。

只有当 `fetch_frac` 足够高时，这一阶段才会带来明显收益。若 fetch 已经低于约 10%，收益通常有限。

### 阶段 5：降低 CRC 校验频率

producer 和 consumer 当前都会遍历完整 hidden states 计算 CRC。producer 端还可能触发一次完整 D2H，这会抵消 ADXL 传输的一部分收益。

推荐增加校验策略：

- 调试和数据生成：全量 CRC；
- 正常训练：按样本抽样校验；
- 对已由传输层保证完整性的路径：关闭重复的全量 CRC。

不要在前面几个阶段完成前直接关闭校验，否则传输损坏、dtype 路径错误和数据竞争可能被误认为性能提升。

验收标准：

- producer/consumer CPU 时间下降；
- hidden states 校验失败仍能被测试策略捕获；
- 训练 loss 与全量 CRC 模式一致。

## 推荐实验矩阵

每次只改变一个变量，并保留 provenance：

| 实验 | 变化 | 主要观察 |
| --- | --- | --- |
| A0 | 本地数据，无 Mooncake | 计算上限 |
| A1 | 当前 Mooncake 单卡 | baseline fetch |
| A2 | 当前 proxy 多卡 | 重复 batch 和广播代价 |
| B1 | 消除 NPU/CPU 往返 | 单卡 fetch、tokens/s |
| B2 | 不同 batch 的 scatter | 多卡 scaling |
| B3 | 去掉 proxy placeholder | proxy 显存和 fetch |
| B4 | 预取深度 1/2 | fetch_frac 和等待时间 |
| B5 | 抽样 CRC | CPU 开销和数据一致性 |

每组至少运行固定数量的 warmup steps 和 measured steps，记录 `train_command.txt`、git SHA、环境和配置。不要只比较单个 step 的最小值，应比较 measured 区间的中位数和 p95。

## 风险和回滚

- 阶段 1 失败时，保留 CPU fallback；只有确认 device-side checksum、shape 和 dtype 校验后再默认启用。
- 阶段 2 失败时，保留当前 rank 0 broadcast 作为兼容模式，并通过配置开关切换。
- 阶段 4 必须限制队列深度并支持取消，否则异常或 epoch 结束时可能泄漏线程和 NPU buffer。
- 阶段 5 默认应保持全量校验，先通过 benchmark 证明 CRC 是瓶颈再降低频率。

## 最终优先级

如果只能做一项，先做阶段 1；如果目标是多卡扩展，阶段 2 是必须项；阶段 3 应随阶段 2 一起完成。阶段 4 和阶段 5 属于在前面链路稳定后继续压缩 fetch 开销的优化。

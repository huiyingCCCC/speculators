# Hidden State 的 Mooncake ADXL 实现与测试方案

本文是评审后的可运行实现方案，目标是：vLLM-Ascend 推理端从 NPU KV cache
生成 hidden states，通过 Mooncake Ascend Direct（`protocol=ascend`）发送到
Speculators 训练端。

## 实现边界

第一版采用 **NPU gather buffer + ADXL**：从 KV cache 按 block 提取 token 后，
在 NPU 上生成 contiguous tensor，再将该 tensor 注册并通过
`batch_put_from()` 发送。不会直接把原始 KV cache 的非连续 block 地址当作一
个连续 buffer 传输。

```text
KV cache (NPU)
  -> extract_from_kv_cache()
  -> contiguous NPU gather buffer
  -> register_buffer(ptr, size)
  -> batch_put_from()
  -> meta 最后发布
  -> 训练端 get_sample()/get_sample_into()
```

## 代码修改

### Store

`hs_connectors/src/hs_connectors/mooncake_store.py` 的 `put_sample()`：

- `protocol=ascend` 且 tensor 在 NPU 时保持设备 tensor；
- 检查 `tensor.is_contiguous()`；
- 调用 `register_buffer()`；
- 调用 `batch_put_from()`；
- CPU tensor（如 `token_ids`）继续使用 `put_tensor()`；
- 所有 tensor 写入成功后才写 manifest；
- manifest 仍然是完成标志。

### Connector

`hs_connectors/src/hs_connectors/mooncake_hidden_states_connector.py` 的
`_write_sample()`：

- 删除 NPU 到 pinned CPU 的 D2H staging；
- 在 copy stream 上完成 slot mapping、gather 和 finite 检查；
- `copy_stream.synchronize()` 后把 NPU tensor 交给 store；
- 当前 `Future` 会持有 NPU tensor 的局部引用直到 `put_sample()` 返回。

`request_finished()` 返回 `True`，继续利用 vLLM 的延迟 block 释放机制。

## 传输完成语义

实现前必须确认当前 Mooncake Python API 的 `batch_put_from()` 是否在返回时
已经完成设备 DMA。如果该 API 是异步提交，必须使用 Mooncake 提供的 task
query/wait 接口；只有传输完成后才能：

1. 写入 meta；
2. 删除 inflight tensor 引用；
3. 在 `get_finished()` 返回 `done_sending`。

如果当前版本确认 `batch_put_from()` 是同步完成，则应在代码注释和测试中固定
这一假设，防止未来升级时引入生命周期 bug。

## 启动环境

推理端和训练端使用同一 Mooncake master、metadata server 和兼容的 Mooncake
Ascend Direct wheel：

```bash
source /usr/local/Ascend/cann-9.0.1/bin/setenv.bash
export PYTHONPATH=/usr/local/python3.12.13/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/python3.12.13/lib/python3.12/site-packages/mooncake:$LD_LIBRARY_PATH
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_GLOBAL_EVENT_ENABLE=1
```

启动 master：

```bash
/usr/local/bin/mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080
```

推理侧使用：

```text
--mooncake-protocol ascend
--mooncake-master <MASTER_IP>:50051
--mooncake-metadata-server http://<MASTER_IP>:8080/metadata
```

## 测试建议

### 1. HIXL 基准回归

先确认底层 ADXL：

```bash
cd /tmp/hixl-ref/examples/third_parties/mooncake_store/python
bash run.sh batch_put_get_sample.py \
  --device_id=0 --schema=d2d --rank=0 --world_size=1
```

成功标准：所有 `hello_*` key 都返回正的字节数，且不出现
`Unsupported transport ascend`。

### 2. Store 单测

```bash
PYTHONPATH=hs_connectors/src pytest -q \
  tests/unit/hs_connectors/test_mooncake_store.py
```

应覆盖 CPU/TCP 兼容路径、manifest、checksum、失败清理和非有限值检查。

建议新增 fake store 测试：

- `protocol=ascend` 的 NPU tensor 调用 `register_buffer + batch_put_from`；
- 非 contiguous NPU tensor 被拒绝；
- `register_buffer` 或 `batch_put_from` 返回负值时不写 meta；
- CPU `token_ids` 仍调用 `put_tensor`。

### 3. Connector 单请求测试

使用 fake KV cache 和 fake Mooncake store，验证：

```text
_write_sample()
  -> 不创建 pinned CPU hidden-state tensor
  -> 传给 store 的 hidden_states.device.type == "npu"
  -> copy stream 同步后才调用 put_sample()
```

### 4. 单机 NPU producer/consumer

先用 `npu:0` producer、`npu:1` consumer 验证 handle 和数据一致性。TCP 路径
用于兼容性回归；ADXL 路径必须使用 `protocol=ascend` 和对应的直接传输 API。

### 5. 训练端读取

第一阶段使用 `get_sample()` 读取 CPU tensor，验证 shape、dtype、token ids 和
checksum。第二阶段增加 `get_sample_into()`：先读 meta，再分配 NPU tensor，
调用 `register_buffer + batch_get_into()` 接收数据。

### 6. 生命周期压力测试

连续发送多个不同长度请求，重点检查：

- hidden-state tensor 在传输完成前没有被复用；
- meta 不早于数据发布；
- producer/consumer 没有偶发 checksum mismatch；
- Mooncake segment 不发生溢出；
- 请求完成后 inflight 引用被释放。

## 验收标准

完整链路通过需要同时满足：

1. vLLM-Ascend 正常生成；
2. 响应包含 `kv_transfer_params.handle`；
3. 推理端日志显示 `protocol=ascend`，无 transport 初始化错误；
4. 训练端按 handle 读取 hidden states；
5. shape、dtype、token ids 对齐；
6. manifest/checksum 校验通过；
7. 多请求压力测试无数据覆盖。

## 不需要修改的代码

当前不需要修改：

```text
/data/c00444317/train/vllm
/data/c00444317/train/vllm-ascend
```

上游已经提供 connector 回调、KV cache 访问和延迟 block 释放。修改范围集中
在：

```text
hs_connectors/src/hs_connectors/mooncake_store.py
hs_connectors/src/hs_connectors/mooncake_hidden_states_connector.py
hs_connectors/src/hs_connectors/transfer.py
```

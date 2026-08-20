# 在 Ascend NPU 上验证 Mooncake Hidden State 传输

本文介绍如何验证 vLLM-Ascend 服务与另一台服务器之间的 hidden state
传输。这个流程不会启动 Speculators 训练，可以先将 vLLM hidden state
提取和 Mooncake 传输问题与训练问题分开排查。

本仓库声明支持使用 GLM 5.2 训练 DSpark，但现有 Mooncake 在线训练端到端
测试使用的是 `Qwen/Qwen3-0.6B`。因此，单独通过 NPU stream copy 测试并不
代表 GLM 5.2、vLLM-Ascend、Mooncake 和训练的完整链路已经验证。在正式
训练前，应先完成本文的跨节点 roundtrip 测试。

## 工作方式

vLLM-Ascend 作为 producer，从 NPU cache 中提取 hidden states，将其复制到
pinned host memory，然后通过 Mooncake 发布。远端 consumer 从 vLLM 响应中
取得 handle，并用这个 handle 从 Mooncake 读取和校验 tensors。

`mooncake_master` 负责协调 distributed store。它不是永久保存 hidden states
的进程，因此仅查看 master 日志不能证明数据传输成功。最终应以远端 consumer
成功读取 `hidden_states` 和 `token_ids` 为准。

## 环境准备

vLLM producer 和远端 consumer 都需要安装 Mooncake 和当前版本的
`hs_connectors`。运行 `mooncake_master` 的节点也需要安装 Mooncake。

```bash
python -m pip install mooncake-transfer-engine
python -m pip install -e ./hs_connectors
```

检查安装：

```bash
python -c "from mooncake.store import MooncakeDistributedStore; print('Mooncake Python API OK')"
command -v mooncake_master
mooncake_master --help
```

本仓库没有声明专用的 Ascend Mooncake wheel。NPU 节点必须使用 CPU/RDMA
通用构建或 Ascend 兼容构建。如果 `mooncake_master` 或 Mooncake 动态库依赖
`libcuda.so.1`，说明安装的是 CUDA 构建，不能直接用于纯 Ascend 节点。此时
应获取与 CANN/vLLM-Ascend 匹配的 Mooncake 包、从源码构建兼容版本，或者将
master 部署在能够运行该构建的独立节点。

vLLM-Ascend 还必须包含 connector 依赖的上游 vLLM API，包括：

- `extract_hidden_states`
- `CacheOnlyAttentionLayer`
- 自定义 KV connector module 加载
- hidden-state cache 访问

第一次测试建议使用 Mooncake `tcp` 协议。producer 和 consumer 必须能够访问
master，并且节点之间使用可路由的业务 IP。不要将 `127.0.0.1` 或无法跨节点
访问的容器内部 IP 配置为 Mooncake client 地址。

## 启动 Mooncake Master

在 producer 和 consumer 都能访问的节点上启动：

```bash
mooncake_master --port 50051
```

`MASTER_IP` 就是运行以上命令的节点业务 IP。例如 master 运行在
`7.24.10.11`：

```text
MASTER_IP=7.24.10.11
MASTER_PORT=50051
```

vLLM producer 和 consumer 必须连接同一个 master。不要在 consumer 上再启动
另一个 master，否则会形成两个相互独立的 store。

从 producer 和 consumer 分别检查 master 端口：

```bash
nc -vz <MASTER_IP> 50051
```

## 启动 vLLM-Ascend Producer

下面是一个 16 卡 GLM 5.2 示例。Mooncake 和 `launch_vllm.py` 参数必须放在
分隔符 `--` 之前；原生 vLLM-Ascend 参数必须放在 `--` 之后。

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO=/data/c00444317/train_code/speculators
MODEL=GLM-5.2-W4A8
MASTER_IP=<MASTER_IP>
MASTER_PORT=50051
LOCAL_IP=<VLLM节点业务IP>

source /path/to/vllm-ascend/venv/bin/activate
# 根据实际 CANN 安装位置加载环境。
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash

export PYTHONPATH="${REPO}/hs_connectors/src:${REPO}/src:${PYTHONPATH:-}"
export MOONCAKE_LOCAL_HOSTNAME="${LOCAL_IP}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

cd "${REPO}"

python scripts/launch_vllm.py "${MODEL}" \
  --hidden-states-backend mooncake \
  --mooncake-master "${MASTER_IP}:${MASTER_PORT}" \
  --mooncake-metadata-server P2PHANDSHAKE \
  --mooncake-protocol tcp \
  --mooncake-global-segment-gib 4 \
  --mooncake-local-buffer-gib 2 \
  --mooncake-writer-threads 2 \
  --target-layer-ids 8 23 39 55 70 \
  --trust-remote-code \
  -- \
  --served-model-name glm5.2 \
  --host 0.0.0.0 \
  --port 8077 \
  --tensor-parallel-size 16 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --seed 1024 \
  --max-num-seqs 512 \
  --max-model-len 8192 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --enforce-eager
```

启动前可以在第一个 `--` 之前临时加入 `--dry-run`，检查生成的最终命令。
输出中应包含：

```text
--kv_transfer_config
MooncakeHiddenStatesConnector
extract_hidden_states
eagle_aux_hidden_state_layer_ids
```

`--target-layer-ids` 不应作为独立参数传给 vLLM。它应被写入
`--speculative_config` JSON 中。启动器默认会追加模型最终层；如果给出的列表
已经包含最终层，可在第一个 `--` 之前添加 `--no-include-last-layer`。

检查服务：

```bash
curl http://<VLLM_IP>:8077/health
curl http://<VLLM_IP>:8077/v1/models
```

`/v1/models` 返回的模型 ID 应包含 `glm5.2`。

## 从远端读取 Hidden States

producer 和 consumer 都需要安装 Mooncake 和当前版本的 `hs_connectors`。
consumer 不需要运行另一个 `mooncake_master`，只需要连接 producer 使用的同一个
master。

### 1. 发送生成请求

在远端 consumer 节点执行：

```bash
curl -s http://<VLLM_IP>:8077/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm5.2",
    "prompt": "The capital of China is",
    "max_tokens": 1,
    "return_token_ids": true
  }' > /tmp/response.json
```

这里的 `model` 必须与 vLLM 的 `--served-model-name` 一致。

### 2. 检查 Mooncake Handle

```bash
python - <<'PY'
import json

with open("/tmp/response.json") as response_file:
    result = json.load(response_file)

print(result["kv_transfer_params"]["handle"])
PY
```

如果响应不存在 `kv_transfer_params.handle`，说明 connector 的 producer 路径
没有正常完成，应先检查 vLLM 日志，不要继续排查 consumer。

### 3. 从相同 Master 读取 Tensors

在 consumer 的 Speculators 仓库根目录执行。`CONSUMER_IP` 必须是其他节点可
路由到的 consumer 业务 IP；`MASTER_ADDR` 必须与 vLLM producer 的
`--mooncake-master` 完全一致。

```bash
export MASTER_ADDR=<MASTER_IP>:50051
export CONSUMER_IP=<CONSUMER节点业务IP>

PYTHONPATH=hs_connectors/src python - <<'PY'
import json
import os

import torch

from hs_connectors.mooncake_store import (
    MooncakeHiddenStatesStore,
    MooncakeStoreConfig,
)

with open("/tmp/response.json") as response_file:
    result = json.load(response_file)

handle = result["kv_transfer_params"]["handle"]
prompt_ids = (
    result["choices"][0].get("prompt_token_ids")
    or result.get("prompt_token_ids")
)

store = MooncakeHiddenStatesStore(
    MooncakeStoreConfig(
        local_hostname=os.environ["CONSUMER_IP"],
        master_server_address=os.environ["MASTER_ADDR"],
        metadata_server="P2PHANDSHAKE",
        protocol="tcp",
        global_segment_size=1024**3,
        local_buffer_size=512 * 1024**2,
    )
).setup()

sample = store.get_sample(handle, timeout=120)
hidden_states = sample["hidden_states"]
token_ids = sample["token_ids"]

print("handle:", handle)
print("hidden_states:", hidden_states.shape, hidden_states.dtype)
print("token_ids:", token_ids.shape)
print("finite:", bool(torch.isfinite(hidden_states).all()))

assert hidden_states.ndim == 3
assert hidden_states.shape[0] == len(token_ids)
assert torch.isfinite(hidden_states).all()
if prompt_ids:
    assert token_ids.tolist() == prompt_ids[: len(token_ids)]

print("Mooncake hidden-state transfer passed")
PY
```

Mooncake store 还会校验 manifest、tensor shape、dtype 和 checksum。成功输出类似：

```text
handle: ...
hidden_states: torch.Size([...]) torch.bfloat16
token_ids: torch.Size([...])
finite: True
Mooncake hidden-state transfer passed
```

## 通过标准

满足以下条件即表示跨节点传输测试通过：

1. vLLM-Ascend 可以正常完成普通生成请求。
2. completion 响应中存在 `kv_transfer_params.handle`。
3. 远端 consumer 能读取 `hidden_states` 和 `token_ids`。
4. `hidden_states` 是三维 tensor。
5. `hidden_states.shape[0]` 与 token 数量一致。
6. hidden states 中没有 NaN 或 Inf。
7. Mooncake manifest 和 checksum 校验通过。

这只能证明 remote hidden-state producer/consumer 路径正常，还不能证明完整训练
正常。在进行大规模 GLM 5.2 DSpark 训练前，应再执行一次单步训练冒烟测试：

```text
--speculator-type dspark
--max-steps 1
--draft-attn-impl eager
--loss-implementation eager
```

训练进程和 vLLM 必须使用相同的 `--target-layer-ids`。先验证 `tcp`，稳定后再
单独验证 RDMA。

## 常见问题

- 没有 `kv_transfer_params.handle`：connector 没有完成 producer 写入，或者当前
  vLLM-Ascend 缺少所需 connector API。
- 导入 `CacheOnlyAttentionLayer`、`SupportsHMA` 或其他 KV connector 类型失败：
  当前 vLLM-Ascend 没有包含对应上游 vLLM API。
- `get_sample()` 超时：检查 master 是否可达、producer 和 consumer 是否连接同
  一个 master，以及 `MOONCAKE_LOCAL_HOSTNAME`/`CONSUMER_IP` 是否为可路由 IP。
- `mooncake_master` 缺少 `libcuda.so.1`：安装的是 CUDA 构建，不能在纯 Ascend
  节点直接运行。
- FP8/W4A8 模型加载失败：这是模型和 Ascend 量化兼容问题，不是 Mooncake
  传输问题。可以先用兼容的 BF16 权重验证传输链路。

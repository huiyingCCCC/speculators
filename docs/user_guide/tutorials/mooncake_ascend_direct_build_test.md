# Mooncake Ascend Direct 编译与测试记录

本文记录在 Ascend910 单机 16 卡容器中，参考 HIXL Mooncake Store 示例编译
Mooncake 并验证 Ascend Direct d2d 传输的完整过程。目标是后续可以在同类
环境中复现，不代表跨机器 RDMA 链路已经验证。

## 环境

已验证的运行环境：

```text
Architecture: aarch64
Python:       3.12.13
CANN:         /usr/local/Ascend/cann-9.0.1
GCC:          12.3.1
CMake:        4.4.0
NPU:          16 x Ascend910
```

加载 CANN 环境并确认 NPU：

```bash
source /usr/local/Ascend/cann-9.0.1/bin/setenv.bash

python - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("npu_count:", torch.npu.device_count())
torch.npu.set_device("npu:0")
x = torch.arange(16, device="npu:0")
torch.npu.synchronize()
print("sum:", x.sum().item())
PY
```

## 源码

Mooncake 源码位于：

```text
/data/c00444317/train/Mooncake
```

HIXL 示例来自：

```text
https://github.com/kvcache-ai/hixl
examples/third_parties/mooncake_store/python
```

HIXL 使用的 transport 名称是 `ascend`，不是本仓库原先默认的 `tcp` 或
`rdma`。Ascend Direct 必须在 Mooncake 编译时启用。

## CMake 配置

不要覆盖已有的 `build-tcp`，使用独立目录：

```bash
rm -rf /data/c00444317/train/Mooncake/build-ascend

cmake -S /data/c00444317/train/Mooncake \
  -B /data/c00444317/train/Mooncake/build-ascend \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_ASCEND=OFF \
  -DUSE_ASCEND_DIRECT=ON \
  -DWITH_STORE=ON \
  -DWITH_STORE_RUST=OFF \
  -DWITH_STORE_GO=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_ETCD=OFF \
  -DBUILD_UNIT_TESTS=OFF
```

这里使用 `USE_ASCEND=OFF`、`USE_ASCEND_DIRECT=ON`。当前源码在
`USE_ASCEND=ON` 时会额外查找 MPI；本环境没有 `mpi.h/libmpi.so`，会导致
CMake 生成阶段失败。配置成功时应看到：

```text
-- Ascend support fabric mem.
-- Build files have been written to: .../build-ascend
```

## 编译

```bash
cmake --build /data/c00444317/train/Mooncake/build-ascend \
  --parallel 32
```

关键产物包括：

```text
build-ascend/mooncake-transfer-engine/src/transport/ascend_transport/ascend_transport.so
build-ascend/mooncake-transfer-engine/src/libtransfer_engine.so
build-ascend/mooncake-integration/engine.cpython-312-aarch64-linux-gnu.so
build-ascend/mooncake-integration/store.cpython-312-aarch64-linux-gnu.so
build-ascend/mooncake-store/src/mooncake_master
build-ascend/mooncake-store/src/mooncake_client
```

## 构建并安装 Python wheel

```bash
cd /data/c00444317/train/Mooncake

BUILD_DIR=build-ascend \
NON_CUDA_BUILD=1 \
OUTPUT_DIR=dist-ascend \
bash scripts/build_wheel.sh 3.12 dist-ascend
```

生成的 wheel 类似：

```text
mooncake_transfer_engine_non_cuda-0.3.11-cp312-cp312-manylinux_2_38_aarch64.whl
```

安装：

```bash
python -m pip install --force-reinstall \
  /data/c00444317/train/Mooncake/mooncake-wheel/dist-ascend/*.whl
```

注意：系统中可能同时存在两套 Mooncake。运行测试时必须优先使用新编译目录，
并把 Mooncake 包目录加入动态库搜索路径：

```bash
export PYTHONPATH=/usr/local/python3.12.13/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/python3.12.13/lib/python3.12/site-packages/mooncake:$LD_LIBRARY_PATH
```

确认加载位置：

```bash
python - <<'PY'
import mooncake
import mooncake.engine
import mooncake.store

print(mooncake.__file__)
print(mooncake.engine.__file__)
print(mooncake.store.__file__)
PY
```

## 启动 Mooncake master

HIXL 示例使用 HTTP metadata server：

```bash
/usr/local/bin/mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080
```

master 同时监听默认 gRPC 端口 `50051` 和 HTTP metadata 端口 `8080`。

## HIXL 单机 d2d 测试

准备 HIXL 示例：

```bash
git clone --depth 1 https://github.com/kvcache-ai/hixl.git /tmp/hixl-ref
cd /tmp/hixl-ref/examples/third_parties/mooncake_store/python
```

设置环境：

```bash
source /usr/local/Ascend/cann-9.0.1/bin/setenv.bash
export PYTHONPATH=/usr/local/python3.12.13/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/python3.12.13/lib/python3.12/site-packages/mooncake:$LD_LIBRARY_PATH
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_GLOBAL_EVENT_ENABLE=1
export MC_LOG_LEVEL=ERROR
```

执行单机单卡 d2d：

```bash
bash run.sh batch_put_get_sample.py \
  --device_id=0 \
  --schema=d2d \
  --rank=0 \
  --world_size=1
```

成功标志是大量类似以下输出：

```text
Retrieved hello_0_0_0 : 147456 bytes
...
Retrieved hello_0_31_60 : 147456 bytes
```

该测试覆盖：

```text
NPU buffer
  -> register_buffer()
  -> batch_put_from()
  -> batch_get_into()
  -> NPU target buffer
```

## 常见问题

### `Unsupported transport ascend`

说明加载的是没有 Ascend Direct 的旧 Mooncake，或 `LD_LIBRARY_PATH` 没有包含
新包目录。确认 `ascend_transport.so` 存在并优先设置新包路径。

### `Acl runtime header file is not exist`

检查 CANN 开发头文件和 `ASCEND_HOME_PATH`，并确认已执行
`setenv.bash`。该警告若只出现在配置阶段但最终生成了
`ascend_transport.so`，仍应以实际运行结果为准。

### `No RDMA devices found`

这是通用 RDMA HCA 探测日志。HIXL 单机 `ascend` d2d 测试可以走 HCCS；若要
验证跨机 RoCE，需要两台机器、可见 HCA 和正确的 HCCL/RoCE 配置。

### master 启动脚本找不到二进制

某些 Python CLI 会错误地在 Python 包目录查找 `mooncake_master`。此时直接
使用源码或系统中的原生二进制：

```bash
/usr/local/bin/mooncake_master ...
```

## 当前验证边界

已验证：

- Ascend910 NPU runtime；
- Mooncake master 和 HTTP metadata server；
- Ascend Direct transport 动态库加载；
- 单机 NPU d2d 零拷贝 `batch_put_from/batch_get_into`。

尚未验证：

- 两台机器之间的 RoCE/RDMA 链路；
- Speculators hidden-state connector 的 Ascend Direct 零拷贝改造；
- vLLM-Ascend 与训练机之间的完整在线训练链路。

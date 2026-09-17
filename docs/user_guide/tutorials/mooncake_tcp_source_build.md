# Docker 内源码构建 Mooncake TCP-only 版本

本文记录在 Ascend NPU Docker 容器中，从克隆源码开始构建并安装 Mooncake
`v0.3.11` TCP-only 版本的完整流程。该版本用于通过主机内存在线传输
hidden states，不加载 Ascend Direct、HCCL、UBSHMEM 或 CUDA transport。

本文命令已在以下环境验证：

- Linux `aarch64`，glibc 2.38
- Python 3.12
- Mooncake `v0.3.11`
- Ascend NPU，producer 使用物理卡 6，consumer 使用物理卡 7
- Mooncake master、producer 和 consumer 位于同一个 Docker 容器

最终验收输出为：

```text
Cross-node NPU hidden-state transfer PASSED
```

## 1. 为什么使用 TCP-only 构建

`mooncake-transfer-engine-npu==0.3.11.post1` 即使配置 `protocol=tcp`，仍可能加载
`AscendDirectTransport`，随后因为 ADXL/Ascend Direct 建链失败而中断。

本方案明确关闭所有设备侧 transport：

- NPU tensor 先由 Speculators connector 复制到 pinned CPU memory。
- Mooncake 只负责通过 TCP 传输 host memory。
- consumer 从 Mooncake 取得并校验 CPU tensors。

因此，这里关闭 `USE_ASCEND` 和 `USE_ASCEND_DIRECT` 是预期行为，不代表无法在
Ascend 训练环境中使用。NPU 与 host 之间的数据复制由 connector 和 `torch_npu`
负责，不由 Mooncake transport 负责。

## 2. 目录和权限约定

以下命令直接修改容器系统环境，不创建 venv。需要 root 权限，或确保当前用户能
写入 `/usr/local` 和 Python site-packages。

```bash
# 按实际目录修改。不要把该变量命名为 HOME。
export TRAIN_ROOT=/data/c00444317/train
export MOONCAKE_SRC="${TRAIN_ROOT}/Mooncake"
export SPECULATORS_SRC="${TRAIN_ROOT}/speculators"
```

安装前检查基础环境：

```bash
uname -m
python3.12 --version
cmake --version
ninja --version
g++ --version
npu-smi info
```

## 3. 克隆 Mooncake 源码

优先从 GitHub 克隆固定 tag，并初始化子模块：

```bash
cd "${TRAIN_ROOT}"

# 固定 v0.3.11，避免 main 分支后续变化影响复现。
git clone --branch v0.3.11 --depth 1 --recurse-submodules \
  https://github.com/kvcache-ai/Mooncake.git Mooncake

cd "${MOONCAKE_SRC}"
git describe --tags --always
git submodule status
```

预期版本：

```text
v0.3.11
```

### GitHub 子模块下载失败时

部分网络环境会在下载 GitHub 子模块时出现 `curl 18`、TLS EOF 或连接重置。
这种情况下保留 Mooncake 主仓库，使用 Gitee 镜像补齐本次构建需要的两个目录：

```bash
cd "${MOONCAKE_SRC}"

# 只有对应目录为空或子模块下载失败时才执行。
git clone --branch v3.0.1 --depth 1 \
  https://gitee.com/mirrors/pybind11.git extern/pybind11

git clone --depth 1 \
  https://gitee.com/alibaba/yalantinglibs.git extern/yalantinglibs
```

不要在已有内容的 `extern/pybind11` 或 `extern/yalantinglibs` 上重复 clone。检查：

```bash
test -f extern/pybind11/include/pybind11/pybind11.h
test -f extern/yalantinglibs/CMakeLists.txt
```

## 4. 安装系统编译依赖

openEuler/RHEL 系容器使用 `dnf`：

```bash
dnf install -y \
  gcc gcc-c++ cmake ninja-build git make pkgconf-pkg-config \
  yaml-cpp-devel jsoncpp-devel gflags-devel glog-devel \
  libibverbs-devel numactl-devel openssl-devel libcurl-devel \
  boost-devel libzstd-devel liburing-devel xxhash-devel \
  patchelf
```

这些包并不都属于 TCP 数据面。Mooncake 的默认 C++ 构建同时包含 Transfer
Engine、Store、RPC、master/client、日志、序列化、HTTP/metrics 和存储支持，
所以依赖数量会明显多于一个普通 TCP client。本文通过关闭 Rust、CUDA、Ascend
Direct 和 EP，避免引入 Cargo、CUDA toolkit、ADXL/HCCL 和 PyTorch 扩展编译依赖。

如果发行版没有 `yaml-cpp-devel`、`msgpack-cxx-devel` 等包，需要按下一节源码安装。

## 5. 安装缺失的 C++ 头文件库

### 5.1 yalantinglibs

本次环境使用 Gitee 上的 `yalantinglibs`，安装到 `/usr/local`：

```bash
cd /tmp
git clone --depth 1 \
  https://gitee.com/alibaba/yalantinglibs.git yalantinglibs

cmake -S yalantinglibs -B yalantinglibs/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARK=OFF \
  -DBUILD_UNIT_TESTS=OFF

cmake --build yalantinglibs/build -j "$(nproc)"
cmake --install yalantinglibs/build
```

### 5.2 msgpack-cxx

本次验证使用 `msgpack-cxx 6.1.1`。如果系统已经存在
`/usr/local/include/msgpack.hpp` 或发行版提供可用的开发包，可以跳过：

```bash
cd /tmp
git clone --branch cpp-6.1.1 --depth 1 \
  https://github.com/msgpack/msgpack-c.git msgpack-c

cmake -S msgpack-c -B msgpack-c/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMSGPACK_BUILD_TESTS=OFF \
  -DMSGPACK_BUILD_EXAMPLES=OFF

cmake --build msgpack-c/build -j "$(nproc)"
cmake --install msgpack-c/build
```

GitHub 不通时可将 URL 换为可访问的 msgpack-c 镜像，但必须检出
`cpp-6.1.1`，不要无版本约束地使用最新分支。

检查安装：

```bash
test -f /usr/local/include/msgpack.hpp
ldconfig
```

## 6. 配置 TCP-only 构建

```bash
cd "${MOONCAKE_SRC}"

cmake -S . -B build-tcp -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_TE=ON \
  -DWITH_STORE=ON \
  -DWITH_STORE_RUST=OFF \
  -DWITH_EP=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_EXAMPLES=ON \
  -DUSE_TCP=ON \
  -DUSE_CUDA=OFF \
  -DUSE_ASCEND=OFF \
  -DUSE_ASCEND_DIRECT=OFF \
  -DUSE_UBSHMEM=OFF \
  -DUSE_ASCEND_HETEROGENEOUS=OFF
```

参数说明：

| 参数 | 作用 |
| --- | --- |
| `WITH_TE=ON` | 构建 Transfer Engine 和 Python engine binding |
| `WITH_STORE=ON` | 构建 Store、master、client 和 Python store binding |
| `WITH_STORE_RUST=OFF` | 不构建 Rust binding，避免安装 Cargo/libclang |
| `WITH_EP=OFF` | 不构建 CUDA/NPU Expert Parallel 扩展 |
| `BUILD_SHARED_LIBS=ON` | 生成可系统安装的 Mooncake `.so` |
| `BUILD_EXAMPLES=ON` | 生成 wheel 脚本要求的 `transfer_engine_bench` |
| `USE_TCP=ON` | 启用 TCP transport |
| `USE_CUDA=OFF` | 不链接 CUDA |
| `USE_ASCEND*=OFF` | 不加载 Ascend Direct、HCCL 或异构 transport |
| `USE_UBSHMEM=OFF` | 不构建 UB memory transport |

检查最终配置，避免 CMake cache 中残留错误选项：

```bash
rg '^(USE_TCP|USE_CUDA|USE_ASCEND|USE_ASCEND_DIRECT|USE_UBSHMEM|WITH_STORE_RUST):' \
  build-tcp/CMakeCache.txt
```

## 7. 编译和系统安装

```bash
cd "${MOONCAKE_SRC}"

# 机器核心很多时不要盲目使用全部核心；32 路并发已足够。
cmake --build build-tcp -j 32

# 安装 Mooncake 自身共享库、头文件、master/client 和 Python binding。
cmake --install build-tcp
ldconfig
```

关键产物应包括：

```bash
test -f build-tcp/mooncake-integration/engine.cpython-312-aarch64-linux-gnu.so
test -f build-tcp/mooncake-integration/store.cpython-312-aarch64-linux-gnu.so
test -f build-tcp/mooncake-transfer-engine/src/libtransfer_engine.so
test -f build-tcp/mooncake-store/src/mooncake_master

test -f /usr/local/lib/libmooncake_common.so
test -f /usr/local/lib/libtransfer_engine.so
test -x /usr/local/bin/mooncake_master
```

注意：在包含 CANN Python 的容器中，CMake 可能把 Python binding 安装到
`/usr/local/Ascend/.../python/site-packages/mooncake`。只要该路径已经由 CANN
环境加入 Python 搜索路径，这是正常的。用下面的命令确认实际导入位置：

```bash
python3.12 -c 'import mooncake.engine as e; print(e.__file__)'
```

## 8. 可选：构建并登记 Python wheel

仅执行 `cmake --install` 已能使用源码构建版本。需要通过 `pip show` 管理版本时，
再执行本节。

### 8.1 构建 wheel

```bash
cd "${MOONCAKE_SRC}"

BUILD_DIR=build-tcp \
NON_CUDA_BUILD=1 \
OUTPUT_DIR=dist-tcp \
./scripts/build_wheel.sh 3.12 dist-tcp
```

脚本会执行 `pip install --upgrade pip build setuptools wheel auditwheel`。这会修改
系统 Python 环境，不只是安装 Mooncake。在带 vLLM 的环境中，升级后的
`setuptools` 可能违反 vLLM 的版本约束。本次环境需要恢复：

```bash
python3.12 -m pip install 'setuptools==80.10.2'
```

如果 `auditwheel repair` 失败，但此前已经出现 `Successfully built`，本机 wheel
通常已在以下目录生成：

```text
mooncake-wheel/dist-tcp/
```

未完成 `auditwheel repair` 的 wheel 只适合当前构建容器，不应直接分发到不同
Linux 发行版。

该脚本会临时修改 `mooncake-wheel/pyproject.toml` 中的包名。脚本异常退出时可能
来不及自动恢复；如果存在备份，应在打包后恢复源码文件：

```bash
cd "${MOONCAKE_SRC}"
if test -f mooncake-wheel/pyproject.toml.backup; then
  cp mooncake-wheel/pyproject.toml.backup mooncake-wheel/pyproject.toml
fi
```

### 8.2 替换旧 NPU wheel

Mooncake 的不同 wheel 都安装同一个 `mooncake` Python package，不能并存使用：

```bash
cd "${MOONCAKE_SRC}"

# 移除会加载 AscendDirectTransport 的旧 wheel。
python3.12 -m pip uninstall -y mooncake-transfer-engine-npu

# --no-deps 避免重新解析并升级训练环境中的 Python 依赖。
python3.12 -m pip install --no-deps --force-reinstall \
  mooncake-wheel/dist-tcp/mooncake_transfer_engine_non_cuda-0.3.11-*.whl
```

检查：

```bash
python3.12 -m pip show mooncake-transfer-engine-non-cuda
python3.12 -c 'import mooncake.engine as e; print(e.__file__)'
```

## 9. 验证没有设备侧动态依赖

```bash
ENGINE_SO=$(python3.12 -c 'import mooncake.engine as e; print(e.__file__)')

ldd "${ENGINE_SO}" | rg 'not found' || true
ldd "${ENGINE_SO}" | rg -i 'ascend|acl|hccl|cuda' || true
```

第二条命令预期没有输出。`libmooncake_common.so` 找不到时，执行：

```bash
cmake --install "${MOONCAKE_SRC}/build-tcp"
ldconfig
```

## 10. 启动 master

源码安装后应直接运行 C++ 二进制：

```bash
/usr/local/bin/mooncake_master --port 50051
```

预期日志包含：

```text
Master service started on port 50051
rpc protocol=tcp
```

不要优先使用 Python 生成的 `mooncake_master` console script。Mooncake `v0.3.11`
的该脚本可能错误地寻找 site-packages 内的 `mooncake_master` 文件并抛出
`FileNotFoundError`；`/usr/local/bin/mooncake_master` 是本次源码安装的真实二进制。

## 11. 使用 NPU 6 和 7 做 TCP roundtrip

打开另一个终端，在卡 6 上启动 producer：

```bash
cd "${SPECULATORS_SRC}"
export PYTHONPATH="${PWD}/hs_connectors/src:${PWD}/src:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=6

python3.12 scripts/test_mooncake_npu_transfer.py producer \
  --local-ip 127.0.0.1 \
  --master 127.0.0.1:50051 \
  --device npu:0 \
  --key tcp-source-build-test-67
```

producer 输出 `Published key` 后保持运行。在第三个终端使用卡 7 启动 consumer：

```bash
cd "${SPECULATORS_SRC}"
export PYTHONPATH="${PWD}/hs_connectors/src:${PWD}/src:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=7

python3.12 scripts/test_mooncake_npu_transfer.py consumer \
  --local-ip 127.0.0.1 \
  --master 127.0.0.1:50051 \
  --device npu:0 \
  --key tcp-source-build-test-67 \
  --timeout 60
```

成功输出：

```text
Received key: tcp-source-build-test-67
hidden_states: shape=(8, 3, 256), dtype=torch.bfloat16
token_ids: shape=(8,), dtype=torch.int64
Cross-node NPU hidden-state transfer PASSED
```

consumer 通过后，回到 producer 终端按 Enter 删除样本并退出，再用 `Ctrl-C` 停止
master。

`127.0.0.1` 只适用于同一容器测试。跨节点时，producer、consumer 的
`--local-ip` 必须分别设置为对端可路由的业务 IP，并连接同一个 master 地址。

## 12. 常见问题

### 12.1 为什么安装了很多包

原因不是 hidden-state tensor 本身，而是当前构建同时启用了 Mooncake Store。
Store 会带入 RPC、HTTP、metrics、序列化、压缩、日志、持久化和 master/client。
本文已经关闭 Rust、CUDA、EP、Ascend Direct 和 UBSHMEM，属于能够运行 Store
TCP 流程的收敛配置。

### 12.2 `transfer_engine_bench` 不存在

`scripts/build_wheel.sh` 无条件复制该文件。重新配置并补建：

```bash
cd "${MOONCAKE_SRC}"
cmake -S . -B build-tcp -DBUILD_EXAMPLES=ON
cmake --build build-tcp --target transfer_engine_bench -j 32
```

### 12.3 `auditwheel` 报找不到 `patchelf`

```bash
dnf install -y patchelf
```

该错误只发生在 wheel 修复阶段，不代表 C++ 源码编译失败。

### 12.4 导入时报 `libmooncake_common.so` 找不到

Mooncake `v0.3.11` 的 wheel 脚本可能没有把该库放进 wheel。执行源码系统安装：

```bash
cd "${MOONCAKE_SRC}"
cmake --install build-tcp
ldconfig
```

### 12.5 日志显示没有 RDMA 设备

TCP-only 环境出现以下日志是正常的：

```text
No RDMA devices found
TCP-only environment, memcpy enabled
```

验收依据是 consumer 的 tensor 内容校验通过，不是是否发现 RDMA HCA。

### 12.6 `pip check` 报训练镜像已有冲突

大型 CANN/vLLM 镜像可能本来就存在 profiler、OpenCV 或 telemetry 版本冲突。
应在安装前后分别保存 `pip check` 输出，只处理本次新增的差异。至少确认：

```bash
python3.12 -m pip show setuptools vllm
python3.12 -m pip show mooncake-transfer-engine-non-cuda
```

不要为了让整个基础镜像的 `pip check` 清零而批量升级训练依赖。

## 13. 最终检查清单

- `git describe` 输出 `v0.3.11`。
- CMake cache 中 `USE_TCP=ON`。
- `USE_CUDA`、`USE_ASCEND`、`USE_ASCEND_DIRECT` 和 `USE_UBSHMEM` 都是 `OFF`。
- `python3.12 -c 'import mooncake.engine'` 成功。
- `ldd engine.so` 没有 `not found`，也没有 Ascend/HCCL/CUDA 依赖。
- `/usr/local/bin/mooncake_master --port 50051` 能启动。
- NPU 6 producer 完成 NPU 到 pinned CPU copy。
- NPU 7 consumer 输出 `Cross-node NPU hidden-state transfer PASSED`。
- 测试完成后 producer 和 master 已退出。

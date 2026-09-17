#!/bin/bash

nic_name=""
local_ip=""
node_0_ip=$local_ip

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_VERSION="0.23.1"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export HCCL_NPU_SOCKET_PORT_RANGE="26000,27000"

TARGET_MODEL=""
HIDDEN_STATES_PATH=""

python scripts/launch_vllm.py $TARGET_MODEL \
    --target-layer-ids 8 23 39 55 70 \
    --hidden-states-path $HIDDEN_STATES_PATH \
    --served-model-name $TARGET_MODEL \
    --host 0.0.0.0 \
    --port 8007 \
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

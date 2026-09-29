#!/bin/bash

set -euo pipefail

NNODES=1
NPROC_PER_NODE=16
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29501"
nic_name=""
local_ip=""
OUTPUT_DIR=""

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

export HCCL_NPU_SOCKET_PORT_RANGE="26000,27000"

TARGET_MODEL=""
DATA_PATH=""
SAVE_PATH=""
VLLM_IP=""
VLLM_PORT=""
HIDDEN_STATES_PATH=""

echo "[$(DATE '+%f %t')] starting single-node train: master=${MASTER_ADDR}:${MASTER_PORT} nproc=${NPROC_PER_NODE}"

torchrun \
    --nnodes="$NNODES" \
    --nproc-per-node="$NPROC_PER_NODE" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    scripts/train.py \
        --verifier-name-or-path $TARGET_MODEL \
        --speculator-type dspark \
        --num-layers 5 \
        --block-size 8 \
        --data-path $DATA_PATH \
        --save-path $SAVE_PATH \
        --epochs 6 \
        --lr 3e-4 \
        --scheduler-type cosine \
        --total-seq-len 8192 \
        --draft-arch qwen3 \
        --draft-hidden-act silu \
        --draft-vocab-size 32000 \
        --target-layer-ids 8 23 39 55 70 \
        --max-anchors 512 \
        --markov-rank 256 \
        --enable-confidence-head \
        --confidence-head-with-markov \
        --loss-fn '{"ce": 0.3, "tv": 0.7}' \
        --confidence-head-alpha 1.0 \
        --loss-implementation eager \
        --checkpoint-freq 0.1 \
        --on-missing raise \
        --on-generate delete \
        --vllm-endpoint http://$VLLM_IP:$VLLM_PORT/v1 \
        --request-timeout 900 \
        --max-retries 2 \
        --seed 42 \
        -- max-steps 60 \
        --log-freq 1 \
        --prefetch-factor 4 \
        --num-workers 8 \
        --trust-remote-code \
        --hidden-states-path $HIDDEN_STATES_PATH \
        --draft-attn-impl sdpa \
        --fsdp-shard \
        --dflash-decay-gamma 6 \
        --train-data-ratio 0.90 \
        --logger tensorboard \
        --log-dir $OUTPUT_DIR/logs \
        --full-attention-indices 0 1 2 3 4


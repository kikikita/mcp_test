#!/bin/bash

# Model configuration  
MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507"
HOST="0.0.0.0"
PORT="8000"
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=137000
GPU_MEMORY_UTILIZATION=0.95

echo "=================================================="
echo "Starting Qwen3 vLLM Server"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "Host: $HOST:$PORT" 
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"

# Start vLLM server for Qwen3
vllm serve "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --disable-log-requests
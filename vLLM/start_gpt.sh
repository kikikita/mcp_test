#!/bin/bash

# Model configuration
MODEL_NAME="openai/gpt-oss-120b"
HOST="0.0.0.0"
PORT="8000"
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=131072
GPU_MEMORY_UTILIZATION=0.95

echo "=================================================="
echo "Starting gpt-oss-120b"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "Host: $HOST:$PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"

# Start vLLM server for GPT
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --tool-call-parser openai --reasoning-parser openai_gptoss --enable-auto-tool-choice
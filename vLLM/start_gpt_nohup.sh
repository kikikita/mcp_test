#!/usr/bin/env bash
set -eu

MODEL_NAME="openai/gpt-oss-120b"
HOST="0.0.0.0"
PORT="8000"
TP_SIZE=2
MAX_LEN=131072
GPU_UTIL=0.95

LOG_FILE="./vllm.log"      # single log file in this folder
PID_FILE="./vllm.pid"      # pid file in this folder

echo "Starting $MODEL_NAME on $HOST:$PORT (TP=$TP_SIZE)…"
nohup vllm serve "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --enable-auto-tool-choice \
  >>"$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Started. PID $(cat "$PID_FILE"). Logs: $LOG_FILE"
echo "Follow logs: tail -f \"$LOG_FILE\""
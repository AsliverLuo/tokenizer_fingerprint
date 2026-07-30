#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-key>" >&2
  echo "model-key: llama3-8b | llama3-8b-instruct | llama31-8b | llama31-8b-instruct" >&2
  exit 2
fi

MODEL_KEY="$1"
MODEL_ROOT="${MODEL_ROOT:-/mnt/vos-79jtuvax/models/LLM-Research}"
API_KEY="${LOCAL_DUMMY_KEY:-local-test-key}"
HOST="${VLLM_HOST:-127.0.0.1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-/tmp}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export LOCAL_DUMMY_KEY="$API_KEY"
export TORCH_COMPILE_DISABLE="$TORCH_COMPILE_DISABLE"
export TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE"
export VLLM_USE_V1="$VLLM_USE_V1"
export VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER"

mkdir -p "$LOG_DIR"

case "$MODEL_KEY" in
  llama3-8b)
    MODEL_DIR="$MODEL_ROOT/Meta-Llama-3-8B"
    SERVED_NAME="LLM-Research/Meta-Llama-3-8B"
    PORT="${PORT:-18101}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/llama3-8b.log"
    ;;
  llama3-8b-instruct)
    MODEL_DIR="$MODEL_ROOT/Meta-Llama-3-8B-Instruct"
    SERVED_NAME="LLM-Research/Meta-Llama-3-8B-Instruct"
    PORT="${PORT:-18102}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/llama3-8b-instruct.log"
    ;;
  llama31-8b)
    MODEL_DIR="$MODEL_ROOT/Meta-Llama-3.1-8B"
    SERVED_NAME="LLM-Research/Meta-Llama-3.1-8B"
    PORT="${PORT:-18103}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/llama31-8b.log"
    ;;
  llama31-8b-instruct)
    MODEL_DIR="$MODEL_ROOT/Meta-Llama-3.1-8B-Instruct"
    SERVED_NAME="LLM-Research/Meta-Llama-3.1-8B-Instruct"
    PORT="${PORT:-18104}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/llama31-8b-instruct.log"
    ;;
  *)
    echo "Unknown model-key: $MODEL_KEY" >&2
    echo "Expected one of: llama3-8b, llama3-8b-instruct, llama31-8b, llama31-8b-instruct" >&2
    exit 2
    ;;
esac

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Missing model directory: $MODEL_DIR" >&2
  exit 1
fi

echo "Starting $SERVED_NAME"
echo "  model_dir=$MODEL_DIR"
echo "  host=$HOST port=$PORT gpu=$GPU_ID"
echo "  max_model_len=$MAX_MODEL_LEN gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "  torch_compile_disable=$TORCH_COMPILE_DISABLE torchdynamo_disable=$TORCHDYNAMO_DISABLE enforce_eager=$ENFORCE_EAGER vllm_use_v1=$VLLM_USE_V1 flashinfer_sampler=$VLLM_USE_FLASHINFER_SAMPLER"
echo "  log_file=$LOG_FILE"

VLLM_EXTRA_ARGS=()
if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" || "$ENFORCE_EAGER" == "TRUE" ]]; then
  VLLM_EXTRA_ARGS+=(--enforce-eager)
fi

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  TORCH_COMPILE_DISABLE="$TORCH_COMPILE_DISABLE" \
  TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" \
  VLLM_USE_V1="$VLLM_USE_V1" \
  VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
  "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${VLLM_EXTRA_ARGS[@]}" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started pid=$PID"
echo "Health check:"
echo "  curl --noproxy \"*\" -H \"Authorization: Bearer $API_KEY\" http://127.0.0.1:$PORT/v1/models"
echo "Log tail:"
echo "  tail -n 80 $LOG_FILE"

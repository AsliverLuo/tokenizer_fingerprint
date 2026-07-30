#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-key>" >&2
  echo "model-key: 7b | 7b-instruct | coder-7b | coder-7b-instruct | qwen3-8b | qwen3-8b-base | qwen35-9b | qwen35-9b-base" >&2
  exit 2
fi

MODEL_KEY="$1"
MODEL_ROOT="${MODEL_ROOT:-/mnt/vos-79jtuvax/models}"
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
DISABLE_ASYNC_SCHEDULING="${DISABLE_ASYNC_SCHEDULING:-1}"
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

resolve_model_dir() {
  for candidate in "$@"; do
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "$1"
}

case "$MODEL_KEY" in
  7b)
    MODEL_DIR="$(resolve_model_dir "$MODEL_ROOT/Qwen2___5-7B" "$MODEL_ROOT/Qwen2.5-7B" "$MODEL_ROOT/Qwen/Qwen2___5-7B" "$MODEL_ROOT/Qwen/Qwen2.5-7B")"
    SERVED_NAME="Qwen/Qwen2.5-7B"
    PORT="${PORT:-18001}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/qwen25-7b.log"
    ;;
  7b-instruct)
    MODEL_DIR="$(resolve_model_dir "$MODEL_ROOT/Qwen2___5-7B-Instruct" "$MODEL_ROOT/Qwen2.5-7B-Instruct" "$MODEL_ROOT/Qwen/Qwen2___5-7B-Instruct" "$MODEL_ROOT/Qwen/Qwen2.5-7B-Instruct")"
    SERVED_NAME="Qwen/Qwen2.5-7B-Instruct"
    PORT="${PORT:-18002}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/qwen25-7b-instruct.log"
    ;;
  coder-7b)
    MODEL_DIR="$(resolve_model_dir "$MODEL_ROOT/Qwen2___5-Coder-7B" "$MODEL_ROOT/Qwen2.5-Coder-7B" "$MODEL_ROOT/Qwen/Qwen2___5-Coder-7B" "$MODEL_ROOT/Qwen/Qwen2.5-Coder-7B")"
    SERVED_NAME="Qwen/Qwen2.5-Coder-7B"
    PORT="${PORT:-18003}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/qwen25-coder-7b.log"
    ;;
  coder-7b-instruct)
    MODEL_DIR="$(resolve_model_dir "$MODEL_ROOT/Qwen2___5-Coder-7B-Instruct" "$MODEL_ROOT/Qwen2.5-Coder-7B-Instruct" "$MODEL_ROOT/Qwen/Qwen2___5-Coder-7B-Instruct" "$MODEL_ROOT/Qwen/Qwen2.5-Coder-7B-Instruct")"
    SERVED_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
    PORT="${PORT:-18004}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/qwen25-coder-7b-instruct.log"
    ;;
  qwen3-8b)
    MODEL_DIR="$(resolve_model_dir "$MODEL_ROOT/Qwen3-8B" "$MODEL_ROOT/Qwen/Qwen3-8B")"
    SERVED_NAME="Qwen/Qwen3-8B"
    PORT="${PORT:-18005}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/qwen3-8b.log"
    ;;
  qwen3-8b-base)
    MODEL_DIR="$(resolve_model_dir "$MODEL_ROOT/Qwen3-8B-Base" "$MODEL_ROOT/Qwen/Qwen3-8B-Base")"
    SERVED_NAME="Qwen/Qwen3-8B-Base"
    PORT="${PORT:-18006}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/qwen3-8b-base.log"
    ;;
  qwen35-9b)
    MODEL_DIR="$(resolve_model_dir "$MODEL_ROOT/Qwen3.5-9B" "$MODEL_ROOT/Qwen/Qwen3.5-9B")"
    SERVED_NAME="Qwen/Qwen3.5-9B"
    PORT="${PORT:-18007}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/qwen35-9b.log"
    ;;
  qwen35-9b-base)
    MODEL_DIR="$(resolve_model_dir "$MODEL_ROOT/Qwen3.5-9B-Base" "$MODEL_ROOT/Qwen/Qwen3.5-9B-Base")"
    SERVED_NAME="Qwen/Qwen3.5-9B-Base"
    PORT="${PORT:-18008}"
    GPU_ID="${GPU_ID:-0}"
    LOG_FILE="$LOG_DIR/qwen35-9b-base.log"
    ;;
  *)
    echo "Unknown model-key: $MODEL_KEY" >&2
    echo "Expected one of: 7b, 7b-instruct, coder-7b, coder-7b-instruct, qwen3-8b, qwen3-8b-base, qwen35-9b, qwen35-9b-base" >&2
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
echo "  torch_compile_disable=$TORCH_COMPILE_DISABLE torchdynamo_disable=$TORCHDYNAMO_DISABLE enforce_eager=$ENFORCE_EAGER vllm_use_v1=$VLLM_USE_V1 disable_async_scheduling=$DISABLE_ASYNC_SCHEDULING flashinfer_sampler=$VLLM_USE_FLASHINFER_SAMPLER"
echo "  log_file=$LOG_FILE"

VLLM_EXTRA_ARGS=()
if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" || "$ENFORCE_EAGER" == "TRUE" ]]; then
  VLLM_EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "$DISABLE_ASYNC_SCHEDULING" == "1" || "$DISABLE_ASYNC_SCHEDULING" == "true" || "$DISABLE_ASYNC_SCHEDULING" == "TRUE" ]]; then
  VLLM_EXTRA_ARGS+=(--no-async-scheduling)
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

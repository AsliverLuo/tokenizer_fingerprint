#!/usr/bin/env bash

set -euo pipefail

# Run local open-source models under /mnt/vos-79jtuvax/models against the
# multilingual next-token probe set, streaming one output token per query.
#
# Default behavior is sequential: start one vLLM service, run smoke probes,
# run the full multilingual probe set, then stop that service before moving
# to the next model. This avoids requiring one GPU per model.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_ROOT="${MODEL_ROOT:-/mnt/vos-79jtuvax/models}"
MODELS="${MODELS:-auto}"
CONFIG="${CONFIG:-}"
SMOKE_PROBES="${SMOKE_PROBES:-probes/en_de_fr_ar_zh_integrated_smoke50_probes.json}"
FULL_PROBES="${FULL_PROBES:-probes/en_de_fr_ar_zh_integrated_500000_probes.json}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-reference_bank_local_multilingual_smoke50_${RUN_TAG}}"
FULL_OUTPUT="${FULL_OUTPUT:-reference_bank_local_multilingual_500k_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/vllm_logs/local_multilingual_${RUN_TAG}}"
LOCAL_DUMMY_KEY="${LOCAL_DUMMY_KEY:-local-test-key}"
HOST="${VLLM_HOST:-127.0.0.1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS="${GPU_IDS:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
QUERY_CONCURRENCY_SMOKE="${QUERY_CONCURRENCY_SMOKE:-8}"
QUERY_CONCURRENCY_FULL="${QUERY_CONCURRENCY_FULL:-8}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"
REQUIRE_SMOKE_PASS="${REQUIRE_SMOKE_PASS:-1}"
MAX_SMOKE_EMPTY_RATE="${MAX_SMOKE_EMPTY_RATE:-0.30}"
START_SERVERS="${START_SERVERS:-1}"
STOP_AFTER_MODEL="${STOP_AFTER_MODEL:-1}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
DISABLE_ASYNC_SCHEDULING="${DISABLE_ASYNC_SCHEDULING:-1}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
BASE_MODEL_ENDPOINT="${BASE_MODEL_ENDPOINT:-completions}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_local_multilingual_nexttoken.sh

Default:
  - model root: /mnt/vos-79jtuvax/models
  - probes: probes/en_de_fr_ar_zh_integrated_500000_probes.json
  - smoke probes: probes/en_de_fr_ar_zh_integrated_smoke50_probes.json
  - model selection: auto-detect supported local LLM directories
  - execution: one model at a time on GPU 0

Useful environment variables:
  MODELS="qwen25-7b qwen3-8b llama31-8b"
  MODEL_ROOT=/mnt/vos-79jtuvax/models
  GPU_IDS="0 1"                    # round-robin across sequential models
  RUN_FULL=0                       # smoke only
  RUN_SMOKE=0                      # skip smoke
  REQUIRE_SMOKE_PASS=0             # continue full run even if smoke is noisy
  START_SERVERS=0                  # reuse already-running vLLM services
  STOP_AFTER_MODEL=0               # keep services running after each model
  QUERY_CONCURRENCY_FULL=8
  MAX_MODEL_LEN=2048
  GPU_MEMORY_UTILIZATION=0.5
  BASE_MODEL_ENDPOINT=chat_completions # reproduce the old base-model method

Supported aliases:
  qwen25-7b
  qwen25-7b-instruct
  qwen3-1.7b
  qwen3-4b-instruct-2507
  qwen3-8b
  qwen3-8b-base
  qwen35-9b
  qwen35-9b-base
  llama3-8b
  llama3-8b-instruct
  llama31-8b
  llama31-8b-instruct

Outputs:
  FULL_OUTPUT/raw_tokens/<model>.jsonl contains one row per multilingual query.
  FULL_OUTPUT/<family>/<model>.json contains the extracted fingerprint.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export LOCAL_DUMMY_KEY
export TORCH_COMPILE_DISABLE
export TORCHDYNAMO_DISABLE
export VLLM_USE_V1
export VLLM_USE_FLASHINFER_SAMPLER
if [[ "$BASE_MODEL_ENDPOINT" == "chat_completions" ]]; then
  export TOKENIZER_FP_FORCE_COMPLETIONS=0
else
  export TOKENIZER_FP_FORCE_COMPLETIONS=1
fi

mkdir -p "$LOG_DIR" "$SMOKE_OUTPUT" "$FULL_OUTPUT"

CURRENT_ALIAS=""

ensure_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
}

resolve_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

model_info() {
  local alias="$1"
  case "$alias" in
    qwen25-7b)
      echo "Qwen2.5-7B-Local|qwen|Qwen/Qwen2.5-7B|18001|$MODEL_ROOT/Qwen2.5-7B:$MODEL_ROOT/Qwen/Qwen2.5-7B:$MODEL_ROOT/Qwen2___5-7B:$MODEL_ROOT/Qwen/Qwen2___5-7B|$BASE_MODEL_ENDPOINT"
      ;;
    qwen25-7b-instruct)
      echo "Qwen2.5-7B-Instruct-Local|qwen|Qwen/Qwen2.5-7B-Instruct|18002|$MODEL_ROOT/Qwen2.5-7B-Instruct:$MODEL_ROOT/Qwen/Qwen2.5-7B-Instruct:$MODEL_ROOT/Qwen2___5-7B-Instruct:$MODEL_ROOT/Qwen/Qwen2___5-7B-Instruct|chat_completions"
      ;;
    qwen3-1.7b)
      echo "Qwen3-1.7B-Local|qwen|Qwen/Qwen3-1.7B|18009|$MODEL_ROOT/Qwen3-1.7B:$MODEL_ROOT/Qwen/Qwen3-1.7B|chat_completions"
      ;;
    qwen3-4b-instruct-2507)
      echo "Qwen3-4B-Instruct-2507-Local|qwen|Qwen/Qwen3-4B-Instruct-2507|18010|$MODEL_ROOT/Qwen3-4B-Instruct-2507:$MODEL_ROOT/Qwen/Qwen3-4B-Instruct-2507|chat_completions"
      ;;
    qwen3-8b)
      echo "Qwen3-8B-Local|qwen|Qwen/Qwen3-8B|18005|$MODEL_ROOT/Qwen3-8B:$MODEL_ROOT/Qwen/Qwen3-8B|chat_completions"
      ;;
    qwen3-8b-base)
      echo "Qwen3-8B-Base-Local|qwen|Qwen/Qwen3-8B-Base|18006|$MODEL_ROOT/Qwen3-8B-Base:$MODEL_ROOT/Qwen/Qwen3-8B-Base|$BASE_MODEL_ENDPOINT"
      ;;
    qwen35-9b)
      echo "Qwen3.5-9B-Local|qwen|Qwen/Qwen3.5-9B|18007|$MODEL_ROOT/Qwen3.5-9B:$MODEL_ROOT/Qwen/Qwen3.5-9B|chat_completions"
      ;;
    qwen35-9b-base)
      echo "Qwen3.5-9B-Base-Local|qwen|Qwen/Qwen3.5-9B-Base|18008|$MODEL_ROOT/Qwen3.5-9B-Base:$MODEL_ROOT/Qwen/Qwen3.5-9B-Base|chat_completions"
      ;;
    llama3-8b)
      echo "Meta-Llama-3-8B-Local|llama|LLM-Research/Meta-Llama-3-8B|18101|$MODEL_ROOT/LLM-Research/Meta-Llama-3-8B:$MODEL_ROOT/Meta-Llama-3-8B|$BASE_MODEL_ENDPOINT"
      ;;
    llama3-8b-instruct)
      echo "Meta-Llama-3-8B-Instruct-Local|llama|LLM-Research/Meta-Llama-3-8B-Instruct|18102|$MODEL_ROOT/LLM-Research/Meta-Llama-3-8B-Instruct:$MODEL_ROOT/Meta-Llama-3-8B-Instruct|chat_completions"
      ;;
    llama31-8b)
      echo "Meta-Llama-3.1-8B-Local|llama|LLM-Research/Meta-Llama-3.1-8B|18103|$MODEL_ROOT/LLM-Research/Meta-Llama-3.1-8B:$MODEL_ROOT/Meta-Llama-3.1-8B|$BASE_MODEL_ENDPOINT"
      ;;
    llama31-8b-instruct)
      echo "Meta-Llama-3.1-8B-Instruct-Local|llama|LLM-Research/Meta-Llama-3.1-8B-Instruct|18104|$MODEL_ROOT/LLM-Research/Meta-Llama-3.1-8B-Instruct:$MODEL_ROOT/Meta-Llama-3.1-8B-Instruct|chat_completions"
      ;;
    *)
      echo "Unknown model alias: $alias" >&2
      return 1
      ;;
  esac
}

raw_path_for_target() {
  local output_dir="$1"
  local target="$2"
  local safe_name
  safe_name="${target//\//_}"
  safe_name="${safe_name//:/_}"
  echo "$output_dir/raw_tokens/${safe_name}.jsonl"
}

log_path_for_alias() {
  local alias="$1"
  echo "$LOG_DIR/${alias}.log"
}

is_healthy() {
  local port="$1"
  curl --noproxy "*" --silent --fail \
    -H "Authorization: Bearer $LOCAL_DUMMY_KEY" \
    "http://$HOST:${port}/v1/models" \
    > /dev/null
}

server_start_failed() {
  local log_file="$1"
  [[ -f "$log_file" ]] || return 1
  grep -Eq \
    "Engine core initialization failed|Ninja build failed|CUDA versions below 12 are not supported|Traceback \\(most recent call last\\):|ModuleNotFoundError|ImportError|No such file or directory|address already in use|OutOfMemoryError|CUDA out of memory" \
    "$log_file"
}

wait_for_health() {
  local alias="$1"
  local port="$2"
  local log_file="$3"
  local deadline
  deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))

  echo "Waiting for $alias health on port $port ..."
  while (( SECONDS < deadline )); do
    if is_healthy "$port"; then
      echo "  healthy: $alias http://$HOST:$port/v1/models"
      return 0
    fi
    if server_start_failed "$log_file"; then
      echo "Server failed while starting $alias on port $port" >&2
      echo "Log tail: $log_file" >&2
      tail -n 100 "$log_file" >&2 || true
      return 1
    fi
    sleep 5
  done

  echo "Timed out waiting for $alias on port $port" >&2
  echo "Check log: $log_file" >&2
  return 1
}

start_model_if_needed() {
  local alias="$1"
  local served_name="$2"
  local port="$3"
  local model_dir="$4"
  local gpu_id="$5"
  local log_file="$6"

  if is_healthy "$port"; then
    echo "Service already healthy for $alias on port $port; skipping start."
    echo ""
    return 0
  fi

  if [[ "$START_SERVERS" != "1" ]]; then
    echo "START_SERVERS=$START_SERVERS and $alias is not healthy on port $port" >&2
    return 1
  fi

  echo "Starting $alias"
  echo "  model_dir=$model_dir"
  echo "  served_name=$served_name"
  echo "  host=$HOST port=$port gpu=$gpu_id"
  echo "  log=$log_file"

  local extra_args=()
  if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" || "$ENFORCE_EAGER" == "TRUE" ]]; then
    extra_args+=(--enforce-eager)
  fi
  if [[ "$DISABLE_ASYNC_SCHEDULING" == "1" || "$DISABLE_ASYNC_SCHEDULING" == "true" || "$DISABLE_ASYNC_SCHEDULING" == "TRUE" ]]; then
    extra_args+=(--no-async-scheduling)
  fi

  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    TORCH_COMPILE_DISABLE="$TORCH_COMPILE_DISABLE" \
    TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" \
    VLLM_USE_V1="$VLLM_USE_V1" \
    VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" \
    --served-model-name "$served_name" \
    --host "$HOST" \
    --port "$port" \
    --api-key "$LOCAL_DUMMY_KEY" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    "${extra_args[@]}" \
    > "$log_file" 2>&1 &

  echo "$!" > "$LOG_DIR/${alias}.pid"
  echo "  pid=$(cat "$LOG_DIR/${alias}.pid")"
}

stop_model_if_started() {
  local alias="$1"
  local pid_file="$LOG_DIR/${alias}.pid"

  if [[ "$STOP_AFTER_MODEL" != "1" || ! -f "$pid_file" ]]; then
    return 0
  fi

  local pid
  pid="$(cat "$pid_file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $alias pid=$pid"
    kill -- "-$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
    sleep 5
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 -- "-$pid" 2>/dev/null || true
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pid_file"
}

cleanup_on_exit() {
  if [[ -n "$CURRENT_ALIAS" ]]; then
    stop_model_if_started "$CURRENT_ALIAS"
  fi
}

trap cleanup_on_exit EXIT

write_config() {
  local config_path="$1"
  shift

  {
    cat <<'YAML'
query_protocol:
  max_tokens: 1
  temperature: 0
  top_p: 1
  presence_penalty: 0
  frequency_penalty: 0
  system_prompt: |-
    Continue the provided prefix as raw text.
    Output only the immediate continuation.
    Do not explain, quote, or add a newline.
  stability_repeat_ratio: 0.1
  stability_repeat_count: 3

reference_models:
YAML
    local alias info target family served port candidates endpoint
    for alias in "$@"; do
      IFS='|' read -r target family served port candidates endpoint < <(model_info "$alias")
      cat <<YAML
- name: $target
  family: $family
  provider: openai
  api_config:
    api_key: \${LOCAL_DUMMY_KEY}
    model: $served
    base_url: http://$HOST:$port/v1
    endpoint: $endpoint
    request_timeout: 300
    max_retries: 5
    retry_delay: 2
    request_interval: 0
    empty_output_retries: 3
    empty_retry_delay: 0.2
    output_normalization:
      strip_leading_newlines: true
YAML
    done
  } > "$config_path"
}

run_build_reference() {
  local target="$1"
  local probes="$2"
  local output="$3"
  local concurrency="$4"

  echo
  echo "Running build-reference:"
  echo "  target=$target"
  echo "  probes=$probes"
  echo "  output=$output"
  echo "  concurrency=$concurrency"

  "$PYTHON_BIN" -m tokenizer_fingerprint.cli build-reference \
    --config "$CONFIG" \
    --probes "$probes" \
    --output "$output" \
    --target "$target" \
    --concurrency "$concurrency"
}

summarize_raw() {
  local raw_path="$1"
  "$PYTHON_BIN" - "$raw_path" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"missing raw file: {path}")
    sys.exit(2)

n = 0
empty = 0
errors = 0
by_lang = Counter()
empty_by_lang = Counter()
outputs = Counter()
for line in path.open(encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    n += 1
    probe_id = row.get("probe_id", "")
    lang = probe_id.split("_partial_")[0] if "_partial_" in probe_id else "unknown"
    text = row.get("output_text", "") or ""
    by_lang[lang] += 1
    outputs[text] += 1
    empty += int(text == "")
    empty_by_lang[lang] += int(text == "")
    errors += int("error" in (row.get("raw_response", {}) or {}))

print({"raw": str(path), "n": n, "empty": empty, "errors": errors})
print("by_lang:", dict(sorted(by_lang.items())))
print("empty_by_lang:", dict(sorted(empty_by_lang.items())))
print("top outputs:")
for text, count in outputs.most_common(10):
    print(f"  {text!r}: {count}")
PY
}

check_smoke_quality() {
  local raw_path="$1"
  "$PYTHON_BIN" - "$raw_path" "$MAX_SMOKE_EMPTY_RATE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_empty_rate = float(sys.argv[2])
n = 0
empty = 0
errors = 0
for line in path.open(encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    n += 1
    text = row.get("output_text", "") or ""
    empty += int(text == "")
    errors += int("error" in (row.get("raw_response", {}) or {}))

if n == 0:
    print("Smoke failed: no rows", file=sys.stderr)
    sys.exit(1)
empty_rate = empty / n
if errors:
    print(f"Smoke failed: errors={errors}", file=sys.stderr)
    sys.exit(1)
if empty_rate > max_empty_rate:
    print(
        f"Smoke failed: empty_rate={empty_rate:.4f} > {max_empty_rate:.4f}",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"Smoke passed: n={n}, empty={empty}, empty_rate={empty_rate:.4f}, errors={errors}")
PY
}

auto_models() {
  local aliases=(
    qwen25-7b
    qwen25-7b-instruct
    qwen3-1.7b
    qwen3-4b-instruct-2507
    qwen3-8b
    qwen3-8b-base
    qwen35-9b
    qwen35-9b-base
    llama3-8b
    llama3-8b-instruct
    llama31-8b
    llama31-8b-instruct
  )
  local alias target family served port candidates endpoint
  for alias in "${aliases[@]}"; do
    IFS='|' read -r target family served port candidates endpoint < <(model_info "$alias")
    IFS=':' read -r -a dirs <<< "$candidates"
    if resolve_existing_dir "${dirs[@]}" > /dev/null; then
      echo "$alias"
    fi
  done
}

ensure_file "$SMOKE_PROBES"
ensure_file "$FULL_PROBES"

if [[ "$MODELS" == "auto" ]]; then
  MODELS="$(auto_models | xargs)"
fi

if [[ -z "$MODELS" ]]; then
  echo "No supported local LLM directories found under $MODEL_ROOT" >&2
  exit 1
fi

read -r -a SELECTED_MODELS <<< "$MODELS"
read -r -a GPU_ID_LIST <<< "$GPU_IDS"
if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  echo "GPU_IDS is empty" >&2
  exit 1
fi

if [[ -z "$CONFIG" ]]; then
  CONFIG="$LOG_DIR/local_multilingual_config.yaml"
  write_config "$CONFIG" "${SELECTED_MODELS[@]}"
fi
ensure_file "$CONFIG"

cp "$FULL_PROBES" "$FULL_OUTPUT/probes_used.json"
cp "$SMOKE_PROBES" "$SMOKE_OUTPUT/probes_used.json"

echo "Run configuration:"
echo "  MODEL_ROOT=$MODEL_ROOT"
echo "  MODELS=$MODELS"
echo "  CONFIG=$CONFIG"
echo "  SMOKE_PROBES=$SMOKE_PROBES"
echo "  FULL_PROBES=$FULL_PROBES"
echo "  SMOKE_OUTPUT=$SMOKE_OUTPUT"
echo "  FULL_OUTPUT=$FULL_OUTPUT"
echo "  LOG_DIR=$LOG_DIR"
echo "  GPU_IDS=$GPU_IDS"
echo "  START_SERVERS=$START_SERVERS"
echo "  STOP_AFTER_MODEL=$STOP_AFTER_MODEL"
echo "  RUN_SMOKE=$RUN_SMOKE"
echo "  RUN_FULL=$RUN_FULL"
echo "  QUERY_CONCURRENCY_FULL=$QUERY_CONCURRENCY_FULL"
echo "  BASE_MODEL_ENDPOINT=$BASE_MODEL_ENDPOINT"
echo

model_index=0
for alias in "${SELECTED_MODELS[@]}"; do
  CURRENT_ALIAS="$alias"
  IFS='|' read -r target family served port candidates endpoint < <(model_info "$alias")
  IFS=':' read -r -a dirs <<< "$candidates"
  model_dir="$(resolve_existing_dir "${dirs[@]}")" || {
    echo "Skipping $alias: no model directory found in candidates: $candidates" >&2
    continue
  }
  gpu_id="${GPU_ID_LIST[$((model_index % ${#GPU_ID_LIST[@]}))]}"
  log_file="$(log_path_for_alias "$alias")"

  start_model_if_needed "$alias" "$served" "$port" "$model_dir" "$gpu_id" "$log_file"
  wait_for_health "$alias" "$port" "$log_file"

  if [[ "$RUN_SMOKE" == "1" ]]; then
    smoke_raw="$(raw_path_for_target "$SMOKE_OUTPUT" "$target")"
    run_build_reference "$target" "$SMOKE_PROBES" "$SMOKE_OUTPUT" "$QUERY_CONCURRENCY_SMOKE"
    summarize_raw "$smoke_raw"
    if [[ "$REQUIRE_SMOKE_PASS" == "1" ]]; then
      check_smoke_quality "$smoke_raw"
    fi
  fi

  if [[ "$RUN_FULL" == "1" ]]; then
    full_raw="$(raw_path_for_target "$FULL_OUTPUT" "$target")"
    run_build_reference "$target" "$FULL_PROBES" "$FULL_OUTPUT" "$QUERY_CONCURRENCY_FULL"
    summarize_raw "$full_raw"
  fi

  stop_model_if_started "$alias"
  CURRENT_ALIAS=""
  model_index=$((model_index + 1))
done

echo
echo "Done."
echo "  full output:  $FULL_OUTPUT"
echo "  smoke output: $SMOKE_OUTPUT"
echo "  config:       $CONFIG"
echo "  logs:         $LOG_DIR"

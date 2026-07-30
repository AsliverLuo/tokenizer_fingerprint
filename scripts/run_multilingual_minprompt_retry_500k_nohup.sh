#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROBES="${PROBES:-probes/en_de_fr_ar_zh_integrated_500000_probes.json}"
REF="${REF:-reference_bank_multilingual_minprompt_retry_500k}"
CONCURRENCY="${CONCURRENCY:-2}"
LOG_DIR="${LOG_DIR:-logs}"
LOG="${LOG:-$LOG_DIR/build_multilingual_minprompt_retry_500k_$(date +%Y%m%d_%H%M%S).log}"

if [[ ! -f "$PROBES" ]]; then
  echo "Probe file not found: $PROBES" >&2
  exit 1
fi

PROBE_COUNT="$(python - "$PROBES" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as fh:
    print(len(json.load(fh)))
PY
)"
if [[ "$PROBE_COUNT" -ne 500000 ]]; then
  echo "Expected 500000 probes for the 500k run, got $PROBE_COUNT from: $PROBES" >&2
  echo "Unset PROBES or set it explicitly to probes/en_de_fr_ar_zh_integrated_500000_probes.json" >&2
  exit 1
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set in this shell." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

nohup bash -lc '
set -euo pipefail

cd "$1"
PROBES="$2"
REF="$3"
CONCURRENCY="$4"
PROBE_COUNT="$5"

check_model_complete() {
  local model_name="$1"
  python - "$REF" "$model_name" "$PROBE_COUNT" <<'PY'
import json
import pathlib
import sys

ref = pathlib.Path(sys.argv[1])
model_name = sys.argv[2]
expected = int(sys.argv[3])
safe_name = model_name.replace("/", "_").replace(":", "_")
raw_path = ref / "raw_tokens" / f"{safe_name}.jsonl"
index_path = ref / "index.json"

if not raw_path.exists():
    raise SystemExit(f"Missing raw token file for {model_name}: {raw_path}")

with raw_path.open(encoding="utf-8") as fh:
    rows = sum(1 for _ in fh)
if rows != expected:
    raise SystemExit(
        f"Incomplete raw token file for {model_name}: rows={rows}, expected={expected}"
    )

if not index_path.exists():
    raise SystemExit(f"Missing reference index: {index_path}")

index = json.loads(index_path.read_text(encoding="utf-8"))
entry = next((item for item in index if item.get("model_name") == model_name), None)
if not entry:
    raise SystemExit(f"Missing model in index.json: {model_name}")
if int(entry.get("n_probes", 0)) != expected:
    raise SystemExit(
        f"Unexpected n_probes for {model_name}: {entry.get('n_probes')}, expected={expected}"
    )

print(f"Validated {model_name}: rows={rows}, n_probes={entry.get('n_probes')}")
PY
}

echo "Started at: $(date)"
echo "ROOT_DIR=$PWD"
echo "PROBES=$PROBES"
echo "PROBE_COUNT=$PROBE_COUNT"
echo "REF=$REF"
echo "CONCURRENCY=$CONCURRENCY"

python -m tokenizer_fingerprint.cli build-reference \
  --config config_bcs.yaml \
  --target DeepSeek-V4-Pro \
  --probes "$PROBES" \
  --output "$REF" \
  --concurrency "$CONCURRENCY"

check_model_complete "DeepSeek-V4-Pro"

python -m tokenizer_fingerprint.cli build-reference \
  --config config_bcs.yaml \
  --target DeepSeek-V4-Flash \
  --probes "$PROBES" \
  --output "$REF" \
  --concurrency "$CONCURRENCY"

check_model_complete "DeepSeek-V4-Flash"

cp "$PROBES" "$REF/probes_used.json"

python -m tokenizer_fingerprint.cli compare-bank \
  --reference "$REF" \
  --output "$REF/bank_compare.json" \
  --csv-output "$REF/bank_compare.csv" \
  --top-k 5

python - <<PY
import json
import pathlib
from collections import Counter

ref = pathlib.Path("$REF")
print("\\nRaw token summary")
for path in sorted((ref / "raw_tokens").glob("*.jsonl")):
    rows = 0
    empty = 0
    retry = 0
    recovered = 0
    final_empty_after_retry = 0
    empty_by_lang = Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            rows += 1
            if row.get("is_empty"):
                empty += 1
                empty_by_lang[row["probe_id"].split("_partial_")[0]] += 1
            meta = row.get("raw_response", {}).get("_empty_output_retry")
            if meta:
                retry += 1
                recovered += int(bool(meta.get("recovered_after_empty")))
                final_empty_after_retry += int(bool(meta.get("final_is_empty")))
    empty_rate = empty / rows if rows else 0.0
    print(path.name)
    print(f"  rows={rows}")
    print(f"  empty={empty} ({empty_rate:.4%})")
    print(f"  retried_results={retry}")
    print(f"  recovered_after_empty={recovered}")
    print(f"  final_empty_after_retry={final_empty_after_retry}")
    print(f"  empty_by_lang={dict(empty_by_lang)}")
PY

echo "Finished at: $(date)"
' _ "$ROOT_DIR" "$PROBES" "$REF" "$CONCURRENCY" "$PROBE_COUNT" > "$LOG" 2>&1 &

PID="$!"
echo "PID=$PID"
echo "LOG=$LOG"
echo "REF=$REF"
echo "Monitor with: tail -f \"$LOG\""

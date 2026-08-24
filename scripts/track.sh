#!/usr/bin/env bash
# Контрольная карта производительности.
#
# Запуск:  ./scripts/track.sh -m model/qwen-0.5b [-q none|q4k|q4km] [-p 128] [-n 32] [-r 5] [-t N]
#
# Делает одно: прогоняет llama-bench с фиксированным числом токенов, дописывает
# строку в bench/history.csv (с хешем коммита) и печатает дельту к предыдущему
# запуску и к baseline. Тренд, а не абсолют -- ровно то, что нужно для решения
# "стало лучше или хуже".
set -euo pipefail

cd "$(dirname "$0")/.."
HIST=bench/history.csv
NOTE=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --note) NOTE="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

mkdir -p bench
[[ -f $HIST ]] || echo "utc,commit,dirty,note,backend,threads,quant,prompt_tokens,prefill_tps,decode_tokens,decode_tps,decode_ms_per_token,load_ms" > "$HIST"

echo "==> cargo build --release" >&2
cargo build --release --bin llama-bench >&2

JSON=$(./target/release/llama-bench "${ARGS[@]}" --json)
echo "$JSON" >&2

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo none)
DIRTY=$(git diff --quiet 2>/dev/null && echo clean || echo dirty)
QUANT=none
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  [[ ${ARGS[$i]} == "-q" || ${ARGS[$i]} == "--quantize" ]] && QUANT="${ARGS[$((i + 1))]}"
done

# Разбор JSON без внешних зависимостей.
val() { echo "$JSON" | sed -n "s/.*\"$1\":\([0-9.]*\).*/\1/p"; }
sval() { echo "$JSON" | sed -n "s/.*\"$1\":\"\([^\"]*\)\".*/\1/p"; }

DEC_TPS=$(val decode_tps)
DEC_MS=$(val decode_ms)
DEC_N=$(val decode_tokens)
MSPT=$(awk -v a="$DEC_MS" -v b="$DEC_N" 'BEGIN{printf "%.3f", (b>0)?a/b:0}')

printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$COMMIT" "$DIRTY" "${NOTE//,/;}" \
  "$(sval backend)" "$(val threads)" "$QUANT" \
  "$(val prompt_tokens)" "$(val prefill_tps)" "$DEC_N" "$DEC_TPS" "$MSPT" "$(val load_ms)" >> "$HIST"

awk -F, -v cur="$DEC_TPS" -v mspt="$MSPT" '
  NR>1 && $11+0>0 { n++; prev=last; last=$11; if(base=="") base=$11 }
  END {
    printf "\n  decode: %.2f tok/s  (%.3f мс/токен)\n", cur, mspt
    if (n>1) printf "  к прошлому запуску: %+.1f%%\n", (last/prev-1)*100
    if (n>0) printf "  к baseline:         %+.1f%%  (было %.2f tok/s)\n", (last/base-1)*100, base
    printf "  история: bench/history.csv (%d записей)\n", n
  }' "$HIST"

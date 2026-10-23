#!/usr/bin/env bash
# Full test matrix for video 7 on the P100 (GPU 0 in PCI order). One model config per call.
# usage: run_all.sh <builddir> <model.gguf> <label> <outdir> [ngl=99]
# builddir has bin/llama-server and bin/llama-bench (upstream or PrismML fork).
set -uo pipefail
B=$1; MODEL=$2; LABEL=$3; OUT=$4; NGL=${5:-99}
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$OUT"; PORT=8778
log() { echo "[run_all:$LABEL] $(date -Is) $*" | tee -a "$OUT/$LABEL.run.log"; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

log "1/5 llama-bench"
bash "$HERE/../02_two_cards_vs_cloud/bench.sh" "$B/bin/llama-bench" "$MODEL" "$NGL" "${LABEL}" "$OUT" 2>&1 | grep -E 'tok/s|power:|vram:|error' | tee -a "$OUT/$LABEL.run.log"

log "2/5 memory vs context"
bash "$HERE/memctx.sh" "$B/bin/llama-server" "$MODEL" "$LABEL" "$OUT" 4096 16384 32768 2>&1 | tail -3 | tee -a "$OUT/$LABEL.run.log"

log "3/5 start server for tasks (ctx 16384)"
"$B/bin/llama-server" -m "$MODEL" -ngl "$NGL" -c 16384 -fa on --jinja --port $PORT --host 127.0.0.1 --reasoning-format deepseek > "$OUT/$LABEL.server.log" 2>&1 &
PID=$!
for i in $(seq 1 240); do curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break; kill -0 $PID 2>/dev/null || { log "FATAL server died, see $LABEL.server.log"; exit 2; }; sleep 2; done
nvidia-smi --query-gpu=timestamp,name,power.draw,memory.used,utilization.gpu --format=csv -l 5 > "$OUT/$LABEL.tasks.power.csv" & SMI=$!
log "3/5 task20"; python3 "$HERE/task20.py" "http://127.0.0.1:$PORT" "$LABEL" "$OUT" 8192 2>&1 | tail -2 | tee -a "$OUT/$LABEL.run.log"
log "4/5 humaneval40"; python3 "$HERE/humaneval40.py" "http://127.0.0.1:$PORT" "$LABEL" "$OUT" 40 2048 2>&1 | tail -2 | tee -a "$OUT/$LABEL.run.log"
log "5/5 toolcall5"; python3 "$HERE/toolcall5.py" "http://127.0.0.1:$PORT" "$LABEL" "$OUT" 2>&1 | tail -1 | tee -a "$OUT/$LABEL.run.log"
kill $SMI $PID 2>/dev/null; wait $PID 2>/dev/null
log "DONE"

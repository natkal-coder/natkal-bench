#!/usr/bin/env bash
# Memory test: start llama-server for a model at several context sizes on GPU 0 (PCI order),
# wait for health, record GPU memory used and the server's own KV/model buffer lines, stop.
# usage: memctx.sh <llama-server> <model.gguf> <label> <outdir> [ctx sizes...]
set -uo pipefail
SERVER=$1; MODEL=$2; LABEL=$3; OUT=$4; shift 4; CTXS=${@:-"4096 16384 32768"}
mkdir -p "$OUT"; PORT=8777
echo "label,ctx,fits,gpu_mem_mib,model_buf_mib,kv_mib,load_s" > "$OUT/$LABEL.memctx.csv"
for C in $CTXS; do
  LOG="$OUT/$LABEL.ctx$C.log"; T0=$(date +%s)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$SERVER" -m "$MODEL" -ngl 99 -c "$C" -fa on --port $PORT --host 127.0.0.1 > "$LOG" 2>&1 &
  PID=$!; FITS=0
  for i in $(seq 1 180); do
    if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then FITS=1; break; fi
    kill -0 $PID 2>/dev/null || break; sleep 2
  done
  LOAD=$(( $(date +%s) - T0 ))
  MEM=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  MB=$(grep -oE 'CUDA0 model buffer size = *[0-9.]+' "$LOG" | grep -oE '[0-9.]+$' | head -1)
  KV=$(grep -oE 'KV self size = *[0-9.]+|CUDA0 KV buffer size = *[0-9.]+' "$LOG" | grep -oE '[0-9.]+$' | head -1)
  echo "$LABEL,$C,$FITS,$MEM,${MB:-},${KV:-},$LOAD" | tee -a "$OUT/$LABEL.memctx.csv"
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 2
done

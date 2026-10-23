#!/usr/bin/env bash
# Same benchmark on every GPU. Usage: bench.sh <llama-bench> <model.gguf> <ngl> <label> <outdir>
# Writes <outdir>/<label>.json (llama-bench) and <outdir>/<label>.power.csv (nvidia-smi 1 Hz).
set -euo pipefail
BENCH=$1; MODEL=$2; NGL=$3; LABEL=$4; OUT=$5
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,name,power.draw,memory.used,utilization.gpu --format=csv -l 1 > "$OUT/$LABEL.power.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
echo "[bench] $LABEL start $(date -Is) model=$(basename "$MODEL") ngl=$NGL" | tee -a "$OUT/$LABEL.log"
"$BENCH" -m "$MODEL" -ngl "$NGL" -p 512 -n 128 -r 3 -o json 2>>"$OUT/$LABEL.log" > "$OUT/$LABEL.json"
echo "[bench] $LABEL done  $(date -Is)" | tee -a "$OUT/$LABEL.log"
python3 - "$OUT/$LABEL.json" "$OUT/$LABEL.power.csv" <<'EOF'
import json, sys, statistics as st
rows = json.load(open(sys.argv[1]))
for r in rows:
    print(f"  {r['model_type']:28} {r['n_prompt']:>4}pp {r['n_gen']:>4}tg  {r['avg_ts']:8.2f} tok/s  (ngl={r['n_gpu_layers']})")
rows2 = [l.split(',') for l in open(sys.argv[2]) if not l.startswith('timestamp') and l.count(',') >= 4]
pw = [float(r[2].split()[0]) for r in rows2 if r[2].strip().split()[0].replace('.','').isdigit()]
mem = [float(r[3].split()[0]) for r in rows2 if r[3].strip().split()[0].isdigit()]
if pw: print(f"  power: mean {st.mean(pw):.0f} W  max {max(pw):.0f} W  samples {len(pw)}")
if mem: print(f"  vram: max {max(mem):.0f} MiB")
EOF

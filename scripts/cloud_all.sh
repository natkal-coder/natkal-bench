#!/usr/bin/env bash
# One rented hour, every cloud measurement for videos 2-6. Runs on the vast.ai box.
# Writes out/timings.csv (step,start,end,seconds), out/<label>.json + .power.csv per bench,
# out/all.log. Each step is time-boxed against BUDGET_S so the hour is never exceeded.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p out
T0=$(date +%s); BUDGET_S=${BUDGET_S:-3300}
log() { echo "[cloud] $(date -Is) +$(( $(date +%s) - T0 ))s $*" | tee -a out/all.log; }
left() { echo $(( BUDGET_S - ( $(date +%s) - T0 ) )); }
tstep() { # tstep <name> <cmd...>
  local n=$1; shift; local s=$(date +%s); log "step $n start"
  "$@" >> out/all.log 2>&1; local rc=$?
  local e=$(date +%s); echo "$n,$s,$e,$(( e - s )),$rc" >> out/timings.csv; log "step $n done rc=$rc in $(( e - s ))s"; return $rc
}
dl() { # dl <url> <sha256> <file>
  [ -f "$3" ] && return 0
  curl -sL --retry 3 -o "$3" "$1" && echo "$2  $3" | sha256sum -c - >> out/all.log
}
BENCH=""
for c in /app/llama-bench /app/full/llama-bench "$(command -v llama-bench || true)"; do [ -n "$c" ] && [ -x "$c" ] && { BENCH=$c; break; }; done
[ -n "$BENCH" ] || { log "FATAL no llama-bench in image"; exit 2; }
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' [:punct:]')
nvidia-smi --query-gpu=name,memory.total,power.limit,power.max_limit,driver_version --format=csv > out/gpu.txt
nproc > out/nproc.txt; log "GPU=$GPU bench=$BENCH cores=$(nproc)"
echo "step,start,end,seconds,rc" > out/timings.csv

# A: 31B Q4_K_M (video 2)
H=https://huggingface.co/bartowski
tstep dl_31b dl "$H/google_gemma-4-31B-it-GGUF/resolve/main/google_gemma-4-31B-it-Q4_K_M.gguf" 2959c1b0a4f792493bb0092846ea06aec54547e47be979c282634d938d501b75 g31b-q4km.gguf
tstep bench_31b bash bench.sh "$BENCH" g31b-q4km.gguf 99 "cloud_${GPU}_g31b-Q4_K_M" out
rm -f g31b-q4km.gguf

# B: 12B quants (video 6), Q4_K_M last and kept for C and D
while read -r Q SHA; do
  [ "$(left)" -gt 400 ] || { log "skip $Q, budget"; continue; }
  F=g12b-$Q.gguf
  tstep "dl_12b_$Q" dl "$H/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-$Q.gguf" "$SHA" "$F"
  tstep "bench_12b_$Q" bash bench.sh "$BENCH" "$F" 99 "cloud_${GPU}_g12b-$Q" out
  [ "$Q" = Q4_K_M ] || rm -f "$F"
done <<'EOF'
Q8_0 929bd294cbdc59e41450488bea524a174f1c6ddc43f140fdb5905a5fd1e41969
Q6_K 53336de05c4af36dcb61daf0d984be076ddf0245e3ce7f61aa10b72cb4044e97
Q5_K_M 24fe88d9e6bba6a1805629a09e40d4d837a894d559c7708055f7a03211e2a25e
Q3_K_M b3a5c0834941b18251317f7dd1096e5d6e01b12e1b06b49670fd26467de8f6b4
Q2_K 5006b410902f522dbd71654493f567ac34b17f55815ddba56a66a1e782742d7a
Q4_K_M 3962624dcd25b947d889dc9ae1bf275b61db6cd4dbe694057f34fffef1671509
EOF

# C: power cap (video 5). Usually refused inside a container; the refusal is itself a result.
powercap() {
  nvidia-smi -q -d POWER | grep -E 'Power Limit|Max Power' | head -4
  for PL in 300 250; do
    if nvidia-smi -pl $PL; then
      bash bench.sh "$BENCH" g12b-Q4_K_M.gguf 99 "cloud_${GPU}_g12b-Q4_K_M_pl${PL}" out
    else echo "nvidia-smi -pl $PL refused"; return 1; fi
  done
  nvidia-smi -pl "$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)"
}
[ "$(left)" -gt 300 ] && tstep powercap powercap

# D: Ollama vs llama.cpp on the same card (video 4)
ollama_test() {
  curl -fsSL https://ollama.com/install.sh | sh
  OLLAMA_MODELS=$PWD/ollama ollama serve > out/ollama_serve.log 2>&1 &
  sleep 8; ollama pull gemma4:12b
  for i in 1 2 3; do
    curl -s http://127.0.0.1:11434/api/generate -d '{"model":"gemma4:12b","prompt":"Write a 200 word summary of how a transformer attention head works.","stream":false,"options":{"num_predict":128,"temperature":0}}' \
      | python3 -c "import json,sys; d=json.load(sys.stdin); print('ollama run',$i,'prompt_tok/s',round(d['prompt_eval_count']/d['prompt_eval_duration']*1e9,2),'gen_tok/s',round(d['eval_count']/d['eval_duration']*1e9,2),'eval_count',d['eval_count'])" | tee -a out/ollama.txt
  done
  ollama show --modelfile gemma4:12b | grep '^FROM' | tee -a out/ollama.txt
  pkill ollama || true
}
[ "$(left)" -gt 420 ] && tstep ollama ollama_test

# E: from-source build timing (video 3), only with budget left
build_test() {
  apt-get update -qq && apt-get install -y -qq git cmake build-essential > /dev/null
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp src
  cmake -S src -B src/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > /dev/null
  cmake --build src/build --target llama-bench -j"$(nproc)" > /dev/null
}
[ "$(left)" -gt 900 ] && tstep build_from_source build_test
log "ALL DONE total $(( $(date +%s) - T0 ))s"
echo ALL_DONE >> out/all.log

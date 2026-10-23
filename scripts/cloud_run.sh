#!/usr/bin/env bash
# Runs on the rented vast.ai box. Pulls the IDENTICAL gemma4:12b blob the P100 used
# (same sha256 from the Ollama registry), builds llama-bench with CUDA, runs bench.sh.
# Usage on the instance:  bash cloud_run.sh            (needs bench.sh next to it)
set -euo pipefail
cd "$(dirname "$0")"
T0=$(date +%s)
log() { echo "[cloud] $(date -Is) +$(( $(date +%s) - T0 ))s $*"; }

# Local P100 file is sha256 1278394b... (pulled 2 months ago). The registry has since
# rebuilt the tag as bb722270... (same model, same quant, 1.6 KB larger). If the exact
# local file was scp'd here as gemma4-12b.gguf we use it; otherwise we pull the registry
# blob and the json records which one ran.
BLOB=sha256:bb722270d54346adc198f213851dbe0207e87c3ecf2a0aff4d92262726215391
MODEL=$PWD/gemma4-12b.gguf
if [ ! -f "$MODEL" ]; then
  log "download model (7.4 GB, registry blob ${BLOB:7:12}) start"
  curl -L --retry 3 -o "$MODEL" "https://registry.ollama.ai/v2/library/gemma4/blobs/$BLOB"
  log "download done; sha256 check"
  echo "${BLOB#sha256:}  $MODEL" | sha256sum -c -
fi
sha256sum "$MODEL" | tee model.sha256

if [ ! -x llama.cpp/build/bin/llama-bench ]; then
  log "build llama.cpp start ($(nproc) cores)"
  apt-get update -qq && apt-get install -y -qq git cmake build-essential >/dev/null
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
  cmake --build llama.cpp/build --target llama-bench -j"$(nproc)" >/dev/null
  log "build done"
fi

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' | tr -d '[:punct:]' | tr -s '_')
log "bench start on $GPU"
bash bench.sh "$PWD/llama.cpp/build/bin/llama-bench" "$MODEL" 99 "cloud_${GPU}_gemma4-12b" "$PWD/out"
log "bench done; total wall $(( $(date +%s) - T0 ))s is the billable setup+run time"

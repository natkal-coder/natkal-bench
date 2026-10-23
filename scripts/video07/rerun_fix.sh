#!/usr/bin/env bash
# Re-run only the two broken stages of the video 7 matrix: llama-bench (results had been overwritten
# by task20's json) and HumanEval (typing header was dropped on full rewrites). P100 only.
set -uo pipefail
S=/media/rickeshtn/683b3557-1768-4fda-a2f5-95248804092f/scratch/youtube_agent
U=$S/llama-upstream/build; P=$S/llama-prism/build; M=$S/models; OUT=$S/42006e71/bench7; HERE=$(cd "$(dirname "$0")" && pwd)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0; PORT=8778
CFGS=("$P|$M/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf|bonsai2_PQ2_0_prismfork|99"
 "$P|$M/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf|bonsai2_PTQ1_0_prismfork|99"
 "$U|$M/bonsai2/Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf|bonsai2_Q2_0_upstream|99"
 "$U|$M/qwen38/Qwen3.8-27B-UD-IQ3_S.gguf|qwen38_IQ3_S_upstream|99"
 "$U|$M/qwen38/Qwen3.8-27B-UD-Q2_K_XL.gguf|qwen38_Q2_K_XL_upstream|99"
 "$U|$M/qwen38/Qwen3.8-27B-UD-Q4_K_M.gguf|qwen38_Q4_K_M_ngl40_upstream|40")
echo "## benches $(date -Is)"
for c in "${CFGS[@]}"; do IFS='|' read -r B MODEL LABEL NGL <<<"$c"
  bash "$HERE/../02_two_cards_vs_cloud/bench.sh" "$B/bin/llama-bench" "$MODEL" "$NGL" "${LABEL}_bench" "$OUT" 2>&1 | grep -E 'tok/s|power:|vram:|error' | sed "s/^/[$LABEL] /"
done
echo "## humaneval $(date -Is)"
for c in "${CFGS[@]}"; do IFS='|' read -r B MODEL LABEL NGL <<<"$c"
  "$B/bin/llama-server" -m "$MODEL" -ngl "$NGL" -c 16384 -fa on --jinja --port $PORT --host 127.0.0.1 --reasoning-format deepseek > "$OUT/$LABEL.server2.log" 2>&1 & PID=$!
  for i in $(seq 1 240); do curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break; kill -0 $PID 2>/dev/null || { echo "FATAL $LABEL server died"; break; }; sleep 2; done
  python3 "$HERE/humaneval40.py" "http://127.0.0.1:$PORT" "$LABEL" "$OUT" 40 2048 2>&1 | tail -1
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
done
echo "RERUN DONE $(date -Is)"

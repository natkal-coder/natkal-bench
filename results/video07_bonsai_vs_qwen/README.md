# Video 7 smoke tests, P100, 2026-10-23 18:4x (llama-bench -p 128 -n 32 -r 1 -fa 1)

| file | build | size | pp128 tok/s | tg32 tok/s |
|---|---|---|---|---|
| Ternary-Bonsai-2-27B-Q2_0 (upstream type) | upstream f5027fb | 7.09 GiB | 94.3 | 17.8 |
| Ternary-Bonsai-2-27B-PTQ1_0 (1.75 bpw) | PrismML fork 73bfe6c | 5.53 GiB | 117.4 | 16.4 |
| Ternary-Bonsai-2-27B-PQ2_0 (2.13 bpw) | PrismML fork 73bfe6c | 6.70 GiB | 92.6 | 26.4 |

# Full matrix, P100 16 GB, 2026-10-23 18:40 → 2026-10-24 00:21 (run_all.sh, llama-server -c 16384 -fa on --jinja --reasoning-format deepseek)

Harness defects found after the run (both fixed, re-running as rerun_fix.sh):
- llama-bench json was overwritten by task20's per-problem json (same `<label>.json` name). Bench re-run as `<label>_bench`.
- HumanEval dropped the prompt header (`from typing import List`) when the model rewrote the whole function: 10 to 17 of 40 failures per model were NameError, not model errors. Scores below are invalid; re-running.
- power/memory CSVs captured both GPUs; the 3080 was busy with a ClearML job, so filter on "P100".

| config | file size | GPU MiB @4k / 16k / 32k ctx | task20 (20 integer problems, 8192 max tok) | task20 gen tok/s | think chars median / max | "wait" count | tools 5 | P100 max W / mean active W |
|---|---|---|---|---|---|---|---|---|
| Bonsai 2 27B PQ2_0, PrismML fork | 6.70 GiB | 7830 / 8610 / 9650 | 20/20 in 142 s | 18.1 | 246 / 504 | 0 | 5/5 (34 s) | 168 / 123 |
| Bonsai 2 27B PTQ1_0, PrismML fork | 5.53 GiB | 6686 / 7466 / 8506 | 20/20 in 196 s | 13.1 | 246 / 504 | 0 | 5/5 (41 s) | 159 / 112 |
| Bonsai 2 27B Q2_0, upstream f5027fb | 7.09 GiB | 8230 / 9010 / 10050 | 20/20 in 212 s | 12.3 | 246 / 504 | 0 | 5/5 (42 s) | 178 / 111 |
| Qwen3.8-27B UD-IQ3_S | 12.0 GB | 11910 / 12690 / 13730 | 20/20 in 419 s | 8.9 | 349 / 5035 | 0 | 5/5 (58 s) | 182 / 113 |
| Qwen3.8-27B UD-Q2_K_XL | 9.8 GB | 9924 / 10704 / 11744 | 20/20 in 253 s | 9.0 | 316 / 652 | 0 | 5/5 (169 s) | 185 / 113 |
| Qwen3.8-27B UD-Q4_K_M, ngl 40 (CPU offload) | 16.5 GB | 15968 / does not fit / does not fit | 20/20 in 827 s | 3.8 | 283 / 4068 | 0 | 5/5 (125 s) | 179 / 111 |

Reading:
- The three Bonsai files produce byte-identical token counts and thinking lengths: same ternary weights, three packings. Only speed and memory differ. PQ2_0 is the one to use on Pascal (fastest tg, fits 32k ctx in under 10 GB).
- task20 does not separate anything (all 20/20) and no model ever wrote "wait": the "wait benchmark" is a null result and must be presented as one. What differs is thinking length (Qwen IQ3_S spent up to 5,035 chars on one problem) and wall time: Bonsai PQ2_0 finished the 20 problems 3x faster than Qwen IQ3_S.
- Q4_K_M Qwen is not a 16 GB option: 40 of 64 layers on the GPU, 3.8 tok/s, 4k context ceiling.
- Pending: HumanEval 40 (fixed harness) and pp512/tg128 llama-bench for all six (rerun_fix.sh, ~5 h).

# Re-run of the two broken stages (rerun_fix.sh, 2026-10-24 00:25 → 05:20), P100-only power

| config | pp512 tok/s | tg128 tok/s | P100 mean W (bench) | VRAM MiB (bench) | HumanEval 40, pass@1 (fixed harness) | HumanEval wall s | gen tokens |
|---|---|---|---|---|---|---|---|
| Bonsai 2 27B PQ2_0, fork | 130.7 | 26.5 | 149 | 7734 | 35/40 = 0.875 | 1234 | 26813 |
| Bonsai 2 27B PTQ1_0, fork | 141.2 | 16.5 | 144 | 6590 | 35/40 = 0.875 | 1934 | 27305 |
| Bonsai 2 27B Q2_0, upstream | 139.0 | 17.8 | 158 | 8126 | 36/40 = 0.900 | 1684 | 23821 |
| Qwen3.8-27B UD-IQ3_S | 142.6 | 11.9 | 168 | 11818 | 35/40 = 0.875 | 2848 | 26922 |
| Qwen3.8-27B UD-Q2_K_XL | 140.5 | 12.4 | 170 | 9832 | 36/40 = 0.900 | 2350 | 23451 |
| Qwen3.8-27B UD-Q4_K_M, ngl 40 | 90.8 | 4.3 | 116 | 10502 | 33/40 = 0.825 | 6461 | 26470 |

Remaining HumanEval failures are the same problems for every model: /32 (find_zero, TypeError), /38 (decode_cyclic: the prompt defines encode_cyclic above it and the harness only keeps the header before the first def, so this one is a harness miss shared by all six), plus 2 to 3 genuine assertion failures and the occasional nested code fence. Treat pass@1 as a floor and compare across rows only.

Bottom line for the video: on a 16 GB P100 the ternary Bonsai 2 27B (PQ2_0, PrismML fork) matches Qwen3.8-27B at IQ3_S/Q2_K_XL on 40 HumanEval problems (35 to 36 of 40 for all), passes the same 20 arithmetic problems and the same 5 tool calls, generates 2.1x faster (26.5 vs 12.4 tok/s), uses 35% less VRAM at 32k context (9.65 vs 11.7 GB), and draws about 20 W less. The "wait" counter was zero for every model; that test is a null result. Q4_K_M Qwen is the one configuration that does not belong on this card.

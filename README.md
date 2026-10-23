# natkal-bench: local GPU vs cloud GPU, measured

Every number behind the Natkal AI videos (https://www.youtube.com/@natkalai). llama-bench pp512/tg128, 3 repetitions; board power sampled at 1 Hz with nvidia-smi (not wall power). Model files are bartowski GGUFs or the Ollama gemma4:12b blob; sha256 in the video descriptions.

| run | GPU | layers on GPU | pp512 tok/s | tg128 tok/s | mean board W |
|---|---|---|---|---|---|
| both_g31b-Q4_K_M_ngl99 | Tesla P100-PCIE-16GB, NVIDIA GeForce RTX 3080 | 99 | 278.2 | 15.41 | 129 |
| cloud_NVIDIAGeForceRTX4090_g12b-Q4_K_M | NVIDIA GeForce RTX 4090 | 99 | 6980.7 | 98.99 | 205 |
| cloud_NVIDIAGeForceRTX4090_g31b-Q4_K_M | NVIDIA GeForce RTX 4090 | 99 | 2908.1 | 43.11 | 260 |
| cloud_rtx3090_gemma4-12b | NVIDIA GeForce RTX 3090 | 99 | 3093.6 | 83.49 |  |
| p100_g12b-Q2_K | Tesla P100-PCIE-16GB | 99 | 392.5 | 19.83 | 212 |
| p100_g12b-Q3_K_M | Tesla P100-PCIE-16GB | 99 | 386.9 | 19.11 | 208 |
| p100_g12b-Q4_K_M | Tesla P100-PCIE-16GB | 99 | 386.1 | 21.56 | 201 |
| p100_g12b-Q4_K_M_pl125 | Tesla P100-PCIE-16GB | 99 | 352.0 | 20.01 | 128 |
| p100_g12b-Q4_K_M_pl150 | Tesla P100-PCIE-16GB | 99 | 368.1 | 21.46 | 140 |
| p100_g12b-Q4_K_M_pl200 | Tesla P100-PCIE-16GB | 99 | 381.6 | 21.98 | 142 |
| p100_g12b-Q5_K_M | Tesla P100-PCIE-16GB | 99 | 391.8 | 20.05 | 218 |
| p100_g12b-Q6_K | Tesla P100-PCIE-16GB | 99 | 400.4 | 17.93 | 186 |
| p100_g12b-Q8_0 | Tesla P100-PCIE-16GB | 99 | 403.4 | 21.53 | 157 |
| p100_g31b-Q4_K_M_ngl40 | Tesla P100-PCIE-16GB | 40 | 85.4 | 3.72 | 106 |
| p100_gemma4-12b | Tesla P100-PCIE-16GB | 99 | 384.4 | 21.98 | 168 |

## Cost per million generated tokens (electricity at $0.15/kWh, substitute yours)

| layout | tok/s | $/M tokens |
|---|---|---|
| Tesla P100, Gemma 4 12B Q4_K_M | 22.0 | 0.32 |
| rented RTX 3090 at $0.195/h, 12B | 83.5 | 0.65 |
| P100 alone, 31B Q4_K_M, 40/60 layers | 3.7 | 1.19 |
| RTX 3080 alone, 31B, 45/60 layers | 6.3 | 0.57 |
| P100 + RTX 3080, 31B, 60/60 | 15.4 | 0.70 |
| rented RTX 4090 at $0.63/h, 31B | 43.1 | 4.06 |

## Scripts

`scripts/bench.sh <llama-bench> <model.gguf> <ngl> <label> <outdir>` runs the benchmark with power sampling. `scripts/cloud_all.sh` is the one-hour vast.ai batch (needs `LD_LIBRARY_PATH=/app` on the ghcr.io/ggml-org/llama.cpp:full-cuda image). `scripts/cloud_run.sh` is the single-model cloud run.

## Caveats

Board power, not wall power. One box, one P100, one 3080. Cloud hosts vary; the Chile 4090 delivered 38 MB/s on an 880 Mbps listing. Corrections welcome as issues.

# Fish Audio S2 Pro on Kaggle 2× T4 — Quantized, One T4 Each

## Context (from research)
- **Model**: Fish Audio S2 Pro = Dual-AR (4B Slow AR + 400M Fast AR + DAC codec, 10 codebooks ~21Hz). Weights ~8GB (bf16); official inference needs **24GB VRAM** (KV cache + overhead). Community reports ~21GB in practice, so it **does not fit a single 15GB T4**.
- **Kaggle free tier**: 2× Tesla T4 (15GB each, 30GB total), **Turing SM7.5**, no NVLink, PCIe-only (~15GB/s), ~73GB disk, 12h session limit. Internet must be enabled in notebook settings.
- **Native engine (`tools/api_server.py`)** has no real multi-GPU tensor parallelism. SGLang-Omni supports `--tp`, but its S2 integration **forces FlashAttention-3 (Hopper-only)** and tensor-parallel over PCIe is slow → not viable on T4.

## Decision
Run a **quantized build on a single T4**, then launch a **second identical instance on the second T4** behind a round-robin load balancer. This gives reliable real-time-ish inference AND true throughput parallelism (each GPU serves independently; SGLang-style continuous batching already handles concurrency within one instance).

## Recommended path: s2.cpp (GGUF) — best fit for T4
Pure C++/GGML + Vulkan, no Python, low VRAM, fast startup, OpenAI-compatible `/v1/tts`.
- Models (HF `mach9243/s2-pro-gguf`): need a **transformer-only + codec-only pair**.
  - `q8_0`: 5.4 + 1.0 ≈ **6.4GB** → recommended (near-lossless, headroom on 15GB).
  - `q4_k_m`: 2.8 + 1.0 ≈ **3.8GB** → max speed/headroom.
  - `f16`: 9.2 + 1.4 ≈ 12GB → reference quality if VRAM allows.

### Steps
1. Kaggle notebook: enable **Internet + GPU (Tesla T4 ×2)**.
2. Build deps + **Vulkan** for NVIDIA: `apt-get install -y vulkan-tools libvulkan1`; ensure NVIDIA Vulkan ICD is visible (`vulkaninfo` should list the T4). *(Risk: Kaggle image may need manual Vulkan ICD config — see Risks.)*
3. Clone `mach92432/s2.cpp`, build.
4. Download GGUF pair (q8_0) to `/kaggle/working` (~6.4GB; 73GB disk is fine).
5. Launch two instances:
   - GPU0: `CUDA_VISIBLE_DEVICES=0 ./s2 --model s2-pro-q8_0-transformer-only.gguf --model-codec s2-pro-q8_0-codec-only.gguf --host 0.0.0.0 --port 8080 --vulkan 0`
   - GPU1: `CUDA_VISIBLE_DEVICES=1 ./s2 ... --port 8081 --vulkan 0`
6. Load balance: round-robin client-side (or nginx stream) across `:8080` and `:8081` `/v1/tts`.
7. Validate: one request per port, check `nvidia-smi` VRAM, measure RTF.

### Alternative quantized path: groxaxo/fish-speech-int4-patch (NF4, Python)
- bitsandbytes **NF4** → ~12GB single T4, OpenAI-compatible API on `:8880`.
- Run twice (GPU0/GPU1) with same LB pattern. **Linux only for speed** (Windows/WSL is ~3–4× slower).
- Use if Vulkan setup on Kaggle proves problematic.

## Not recommended on T4
- Unquantized fp16 across both T4s via SGLang-Omni `--tp 2` (forces FA3/Hopper; slow PCIe TP).
- Pipeline-split patch (discussion #1264) puts AR on GPU0 + codec on GPU1 — fits memory but is a single instance (no throughput gain) and unquantized (slower).

## Risks / caveats
- **Vulkan on Kaggle T4**: s2.cpp uses Vulkan for the GPU transformer path. Verify `vulkaninfo` sees the device; if ICD missing, codec can fall back to CPU (`--codec-vulkan -1`) but that roughly doubles latency.
- s2.cpp is **ALPHA/experimental**; voice-clone quality slightly below official.
- 12h session limit → re-download model + restart each session (script it).
- T4 has no FP8; all work is fp16/int8/NF4.

## Validation
- `nvidia-smi`: ~6–7GB used per GPU, codec on GPU (not CPU).
- Both `:8080` and `:8081` return valid WAV for a sample text.
- Fire 2 concurrent requests → each lands on a separate GPU (≈2× single-GPU throughput).
- Spot-check audio quality vs Fish Audio hosted API.

## Community references
- `fishaudio/fish-speech` Discussions **#1264** (2×16GB pipeline-split patch)
- `fishaudio/fish-speech` Issue **#1168** (quantization for 12–16GB VRAM)
- `groxaxo/fish-speech-int4-patch` (NF4, 12GB single GPU)
- `mach92432/s2.cpp` + `mach9243/s2-pro-gguf` (GGUF, 4–12GB, Vulkan, two-GPU capable)
- `Imagilux/fishaudio-s2-pro` (INT8 fork, 16GB)
- HF `fishaudio/s2-pro` Discussion **#10** (Fish Audio: "Using tensor parallelism")
- `llcuda` KAGGLE_GUIDE (2×T4 hardware/limits reference)

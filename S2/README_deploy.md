# Fish Audio S2 Pro on Kaggle 2× T4 — load-balanced deployment

Implements the agreed plan: **quantized (`q8_0`) S2 Pro on one T4 each, two
independent servers behind a round-robin load balancer** for true throughput
parallelism.

## Why this shape
- S2 Pro (4B Slow-AR + 400M Fast-AR) needs ~24 GB unquantized → will not fit a
  single 15 GB T4. The `q8_0` GGUF (~5.6 GB) fits one T4 with headroom.
- The `s2.cpp` server (`rodrigomt/s2.cpp`) handles **one request at a time** and
  returns **HTTP 503** under concurrency. So we run **one server per GPU** and
  round-robin between them, retrying 503s on the peer.
- We build the **CUDA** backend (`--cuda N`) rather than Vulkan, because Kaggle's
  T4 image reliably has the CUDA toolkit the existing notebook uses; Vulkan ICD
  support on Kaggle is uncertain.

## Files
| File | Purpose |
|------|---------|
| `setup_s2.sh` | apt deps → clone `rodrigomt/s2.cpp` → `cpy.cu` patch → CUDA build (sm_75) → download `q8_0` GGUF + tokenizer |
| `patch_cpy.py` | defensive fix for the ggml CUDA `cpy.cu` grid-assert crash (idempotent, non-fatal) |
| `download_model.py` | downloads `s2-pro-q8_0.gguf` + `tokenizer.json` (with the HF Xet/GCP 403 workaround) |
| `serve_s2.py` | launches 2 servers (`cuda 0`→:8080, `cuda 1`→:8081) + round-robin LB on :8000 (503-retrying) |
| `test_s2.py` | fires N concurrent requests via the LB, reports throughput + per-GPU VRAM |
| `S2_Kaggle_2xT4_server.ipynb` | thin notebook that runs the above (build → serve → test → sample) |

## Quick start (Kaggle)
1. New notebook → **Settings**: enable **Internet**, accelerator **GPU (Tesla T4 ×2)**.
2. Upload these files (or `git clone` the folder) into the notebook working dir.
3. Run `S2_Kaggle_2xT4_server.ipynb` top-to-bottom:
   - Cell 1 builds + downloads (takes several minutes; ~6 GB download).
   - Cell 2 launches both servers + the LB, then prints `/health`.
   - Cell 3 runs the concurrency test (`S2_N=10`).
   - Cell 4 generates one sample and plays it inline.

## Local / non-Kaggle run
All scripts honour env overrides (defaults assume `/kaggle/working`):
```
export S2_WORKDIR=/path/to/work
export S2_BINARY=$S2_WORKDIR/s2.cpp/build/s2
export S2_MODEL=$S2_WORKDIR/models/s2-pro-q8_0.gguf
export S2_TOKENIZER=$S2_WORKDIR/models/tokenizer.json
bash setup_s2.sh
python3 serve_s2.py &          # background
S2_N=10 python3 test_s2.py
```

## Endpoints
- `POST /generate` (multipart/form-data) → forwarded to a backend. Fields:
  `text` (required), `params` (JSON), `reference`/`reference_text` (voice cloning).
- `GET /health` → JSON backend statuses.
- `POST /stop` → stop servers + LB.

## Tuning
- **More throughput:** raise `S2_N` in the test; the LB keeps both GPUs busy.
  Each server is single-flight, so effective concurrency ≈ number of GPUs here (2).
- **Tighter VRAM** (e.g. a single 8 GB T4): pass `--gpu-layers 18 --codec-cpu`
  as extra args inside `serve_s2.py` (`launch_servers`), or switch the model to
  `q4_k_m` (`S2_MODEL`).
- **Both GPUs but lower quality:** swap `q8_0` for `q6_k` / `q4_k_m` in
  `download_model.py` / `S2_MODEL`.

## Caveats
- `s2.cpp` is alpha; voice-clone quality depends on reference audio SNR/length.
- Split long text (>~800 tokens / 37 s) into sentences for best quality.
- Kaggle sessions end after 12 h — re-run `setup_s2.sh` each session (or persist
  the built binary + model in Kaggle Dataset/volume to skip rebuild).
- Weights are under the **Fish Audio Research License** (non-commercial); obtain
  a commercial license from Fish Audio for production use.

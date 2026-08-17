#!/usr/bin/env bash
# Build the CUDA s2.cpp server binary and download the q8_0 GGUF for Kaggle 2xT4.
# Run inside a Kaggle notebook cell with Internet + GPU (Tesla T4 x2) enabled:
#     !bash setup_s2.sh
set -e

WORKDIR="${S2_WORKDIR:-/kaggle/working}"
REPO="${S2_REPO:-$WORKDIR/s2.cpp}"
LOGDIR="$WORKDIR"
PAR=$(nproc)

echo "==> Workdir: $WORKDIR"

echo "==> Installing build deps"
apt-get update -qq && apt-get install -y -qq git cmake build-essential

echo "==> Cloning rodrigomt/s2.cpp (with ggml submodule)"
rm -rf "$REPO"
git clone --recurse-submodules https://github.com/rodrigomatta/s2.cpp.git "$REPO"
cd "$REPO"

echo "==> Defensive cpy.cu grid-assert patch"
python3 "$WORKDIR/patch_cpy.py" || echo "patch_cpy.py returned non-zero (continuing)"

echo "==> Locating libcuda.so for CMake's FindCUDAToolkit"
DST="/usr/lib/x86_64-linux-gnu/libcuda.so"
found=""
for d in /usr/local/nvidia/lib64 /usr/local/nvidia/lib \
         /usr/local/cuda/targets/x86_64-linux/lib/stubs \
         /usr/local/cuda/lib64/stubs /usr/lib/x86_64-linux-gnu; do
    c=$(ls $d/libcuda.so* 2>/dev/null | head -n1 || true)
    [ -n "$c" ] && { found="$c"; break; }
done
for p in $(echo "${LD_LIBRARY_PATH:-}" | tr ':' ' '); do
    c=$(ls "$p"/libcuda.so* 2>/dev/null | head -n1 || true)
    [ -n "$c" ] && { found="$c"; break; }
done
if [ -n "$found" ] && [ ! -e "$DST" ]; then
    ln -sf "$found" "$DST"
    echo "Linked $DST -> $found"
elif [ -n "$found" ]; then
    echo "$DST already present"
else
    echo "WARNING: no libcuda.so* candidate; CUDA build may fail"
fi

echo "==> Configuring + building (CUDA, sm_75 for T4)"
rm -rf build
cmake -B build -DS2_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=75 2>&1 | tail -n 20
cmake --build build --parallel "$PAR" 2>&1 | tail -n 40

if [ ! -x "$REPO/build/s2" ]; then
    echo "BUILD FAILED: build/s2 not found"
    exit 1
fi
echo "BUILD OK: $REPO/build/s2"

echo "==> Downloading model + tokenizer"
python3 "$WORKDIR/download_model.py"

echo "DONE. Binary: $REPO/build/s2"
echo "Model dir : $WORKDIR/models"

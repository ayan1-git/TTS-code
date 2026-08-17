"""Defensive fix for the ggml CUDA `cpy.cu` grid-assert crash on Kaggle T4.

Known ggml bug: when a reference clip is large enough, a CUDA copy tensor
exceeds `USHRT_MAX` grid dimensions and `ggml_cuda_cpy` hard-aborts with
`GGML_ASSERT(grid_y < USHRT_MAX) failed`. Upstream fix (ggml-org/llama.cpp
PR #25000) routes such copies through the generic 1-D scalar copy path.

This patches the vendored `ggml/src/ggml-cuda/cpy.cu` directly (style-agnostic,
idempotent, non-fatal). If the assert is already gone (newer ggml), it is a noop.
"""
import glob
import os
import re
import sys


def fix_cpy_cu(path):
    with open(path) as f:
        s = f.read()
    if "grid_y < USHRT_MAX" not in s:
        # Already fixed upstream, or a different ggml layout.
        return "noop"

    orig = s

    repls = [
        ("const int x = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;",
         "const int64_t x = (int64_t) blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;"),
        ("const int y = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;",
         "const int64_t y = (int64_t) blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;"),
        ("const int tx = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.x;  // transpose block offset",
         "const int64_t tx = (int64_t) blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.x;  // transpose block offset"),
        ("const int ty = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.y;",
         "const int64_t ty = (int64_t) blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.y;"),
    ]
    for a, b in repls:
        s = s.replace(a, b)

    s = s.replace("GGML_ASSERT(num_blocks < UINT_MAX);", "GGML_ASSERT(num_blocks <= INT_MAX);")
    s = s.replace("GGML_ASSERT(grid_x < UINT_MAX);", "GGML_ASSERT(grid_x <= INT_MAX);")

    pat = re.compile(r"GGML_ASSERT\(grid_y < USHRT_MAX\);(.*?)\}\s*else\s*\{", re.DOTALL)
    m = pat.search(s)
    if not m:
        return "unknown"
    body = m.group(1)
    body = body.replace("GGML_ASSERT(grid_z < USHRT_MAX);", "")
    body = re.sub(r"\n\s*\n", "\n", body)

    m1 = re.search(r"cpy_scalar<cpy_1_scalar<src_t, dst_t>>\s*<<<[^;]*?\);", s, re.DOTALL)
    if m1:
        one_d = "\n".join("            " + ln.strip() for ln in m1.group(0).split("\n") if ln.strip())
    else:
        one_d = ("            cpy_scalar<cpy_1_scalar<src_t, dst_t>>"
                 "<<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>\n"
                 "                (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,"
                 " ne10, ne11, ne12, nb10, nb11, nb12, nb13);")

    fallback = (
        "        if (grid_y > USHRT_MAX || grid_z > USHRT_MAX) {\n"
        "            const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;\n"
        "            GGML_ASSERT(num_blocks <= INT_MAX);\n"
        + one_d + "\n"
        "        } else {\n"
        + body.rstrip() + "\n"
        "        }\n"
        "    } else {"
    )
    s = s[:m.start()] + fallback + s[m.end():]
    if s == orig:
        return "unknown"
    with open(path, "w") as f:
        f.write(s)
    return "patched"


def main():
    repo = os.environ.get("S2_REPO", "/kaggle/working/s2.cpp")
    cands = glob.glob(os.path.join(repo, "**", "cpy.cu"), recursive=True)
    cands = [c for c in cands if "ggml-cuda" in c] or cands
    if not cands:
        print("WARNING: cpy.cu not found under", repo, "- CUDA grid-assert fix SKIPPED.")
        return 0
    cf = cands[0]
    res = fix_cpy_cu(cf)
    print("cpy.cu fix result:", res, "->", cf)
    if res == "patched":
        with open(cf) as f:
            chk = f.read()
        assert "grid_y < USHRT_MAX" not in chk, "FATAL: cpy.cu grid assert still present!"
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download the S2 Pro q8_0 GGUF + tokenizer for the CUDA s2.cpp build.

Mirrors the Xet-storage workaround used by the working Kaggle notebook:
on GCP (Kaggle) the Xet CAS CDN rejects its own signed URLs (403), so we
disable the Xet client and let huggingface_hub fetch from the working
cas-bridge endpoint.
"""
import os
import subprocess
import sys

REPO_ID = "rodrigomt/s2-pro-gguf"
FILES = ["s2-pro-q8_0.gguf", "tokenizer.json"]
MIN_SIZE = {"s2-pro-q8_0.gguf": 1_000_000_000, "tokenizer.json": 1_000_000}


def main():
    workdir = os.environ.get("S2_WORKDIR", "/kaggle/working")
    model_dir = os.path.join(workdir, "models")
    os.makedirs(model_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", "")
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        print("Loaded HF_TOKEN from Kaggle secret")
    except Exception:
        pass

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    else:
        print("No HF_TOKEN set - repo is public, continuing without auth.")

    subprocess.run("pip install -U -q 'huggingface_hub[cli]' 2>/dev/null || true", shell=True)

    from huggingface_hub import hf_hub_download

    for fn in FILES:
        dst = os.path.join(model_dir, fn)
        if os.path.exists(dst) and os.path.getsize(dst) >= MIN_SIZE.get(fn, 1):
            print("Already present (skipping):", fn, os.path.getsize(dst), "bytes")
            continue
        print("Downloading", fn, "...")
        try:
            p = hf_hub_download(
                repo_id=REPO_ID, filename=fn, token=hf_token or None,
                local_dir=model_dir, local_dir_use_symlinks=False)
            print("  ->", p, os.path.getsize(p) if os.path.exists(p) else "?")
        except Exception as e:
            print("  hf_hub_download failed:", repr(e))
            url = f"https://huggingface.co/{REPO_ID}/resolve/main/{fn}"
            subprocess.run(
                f'wget -c -L --tries=8 --timeout=60 -U "Mozilla/5.0" -O "{dst}" "{url}"',
                shell=True)
        if not (os.path.exists(dst) and os.path.getsize(dst) >= MIN_SIZE.get(fn, 1)):
            print("  WARNING: still missing or too small:", fn)

    print("Models in", model_dir)
    subprocess.run(f"ls -lh {model_dir}", shell=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

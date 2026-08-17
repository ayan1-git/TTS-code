"""Validate the 2xT4 load-balanced S2 deployment.

Sends N concurrent TTS requests through the load balancer, measures wall-clock
throughput, and prints per-GPU VRAM usage via nvidia-smi.

Usage:
  python3 test_s2.py
Env:
  S2_LB_URL   load balancer base URL (default: http://127.0.0.1:8000)
  S2_N        number of concurrent requests (default: 10)
  S2_TEXT     text to synthesize (default: a fixed sentence)
"""
import base64
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

LB_URL = os.environ.get("S2_LB_URL", "http://127.0.0.1:8000").rstrip("/")
N = int(os.environ.get("S2_N", "10"))
TEXT = os.environ.get(
    "S2_TEXT",
    "Fish Audio S2 Pro running on two Kaggle Tesla T4 GPUs behind a load balancer. "
    "This sentence is long enough to exercise real generation rather than a tiny stub.",
)


def build_multipart(fields):
    boundary = "----s2testboundary7Q3k9"
    b = b""
    for name, value in fields:
        b += (f"--{boundary}\r\n"
              f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
              f"{value}\r\n").encode()
    b += f"--{boundary}--\r\n".encode()
    return b, f"multipart/form-data; boundary={boundary}"


def one_request(i):
    body, ct = build_multipart([
        ("text", TEXT),
        ("params", json.dumps({"max_new_tokens": 512, "temperature": 0.8})),
    ])
    url = f"{LB_URL}/generate"
    req = urllib_request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", ct)
    t0 = time.time()
    try:
        with urllib_request.urlopen(req, timeout=1800) as resp:
            data = resp.read()
            dt = time.time() - t0
            return {"ok": True, "bytes": len(data), "sec": dt, "status": resp.status}
    except HTTPError as e:
        return {"ok": False, "status": e.code, "sec": time.time() - t0,
                "err": e.read()[:200].decode(errors="replace")}
    except URLError as e:
        return {"ok": False, "status": 0, "sec": time.time() - t0, "err": str(e)}


def show_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True)
        print("\nGPU VRAM (nvidia-smi):")
        print(out.strip())
    except Exception as e:
        print("nvidia-smi failed:", e)


def main():
    print(f"Load balancer: {LB_URL}")
    print(f"Concurrent requests: {N}")
    print(f"Text: {TEXT!r}\n")

    show_vram()
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        results = list(ex.map(one_request, range(N)))
    wall = time.time() - t_start

    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]
    total_audio = sum(r.get("bytes", 0) for r in ok)

    print(f"\n=== Results ({N} requests, wall {wall:.2f}s) ===")
    print(f"Success : {len(ok)}/{N}")
    print(f"Failed  : {len(fail)}")
    if ok:
        avg = sum(r['sec'] for r in ok) / len(ok)
        print(f"Avg request time : {avg:.2f}s")
        print(f"Aggregate throughput: {N / wall:.2f} requests/sec")
        print(f"Total audio bytes : {total_audio} ({total_audio/1024/1024:.1f} MB)")
    if fail:
        print("First failure:", fail[0])
    show_vram()
    return 0 if len(fail) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

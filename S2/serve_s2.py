"""Launch two s2.cpp HTTP servers (one per T4 GPU) and a round-robin
load balancer in front of them.

Why two instances? The s2.cpp `--server` handles ONE synthesis at a time and
returns HTTP 503 for concurrent requests. Spinning up one server per GPU and
round-robining across them (with 503 retry on the peer) gives true throughput
parallelism on Kaggle's 2xT4, exactly as planned.

Endpoints:
  POST /generate        -> forwarded (multipart/form-data) to a backend
  GET  /health          -> JSON: backend statuses + model info
  POST /stop            -> stop all servers + the LB (optional)

Usage:
  python3 serve_s2.py
Environment:
  S2_BINARY      path to the built `s2` binary (default: /kaggle/working/s2.cpp/build/s2)
  S2_MODEL       path to the GGUF model      (default: /kaggle/working/models/s2-pro-q8_0.gguf)
  S2_TOKENIZER   path to tokenizer.json     (default: /kaggle/working/models/tokenizer.json)
  S2_LB_PORT     load balancer port         (default: 8000)
  S2_BACKENDS    "cuda0:8080,cuda1:8081"    (default: 0:8080,1:8081)
  S2_WORKDIR     base dir                   (default: /kaggle/working)
"""
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

WORKDIR = os.environ.get("S2_WORKDIR", "/kaggle/working")
S2_BINARY = os.environ.get("S2_BINARY", os.path.join(WORKDIR, "s2.cpp", "build", "s2"))
S2_MODEL = os.environ.get("S2_MODEL", os.path.join(WORKDIR, "models", "s2-pro-q8_0.gguf"))
S2_TOKENIZER = os.environ.get("S2_TOKENIZER", os.path.join(WORKDIR, "models", "tokenizer.json"))
LB_PORT = int(os.environ.get("S2_LB_PORT", "8000"))

# Backends: list of (cuda_device, port). Default two T4s.
_raw = os.environ.get("S2_BACKENDS", "0:8080,1:8081")
BACKENDS = []
for part in _raw.split(","):
    part = part.strip()
    if not part:
        continue
    cuda_dev, _, port = part.partition(":")
    BACKENDS.append((int(cuda_dev), int(port)))

procs = []


def log(msg):
    print(f"[serve_s2] {msg}", flush=True)


def tcp_alive(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def launch_servers():
    if not os.path.exists(S2_BINARY):
        log(f"FATAL: s2 binary not found at {S2_BINARY}")
        sys.exit(1)
    if not os.path.exists(S2_MODEL):
        log(f"FATAL: model not found at {S2_MODEL}")
        sys.exit(1)
    for cuda_dev, port in BACKENDS:
        log(f"Launching backend cuda:{cuda_dev} on port {port}")
        p = subprocess.Popen(
            [S2_BINARY, "--model", S2_MODEL, "--tokenizer", S2_TOKENIZER,
             "--server", "--host", "0.0.0.0", "--port", str(port),
             "--cuda", str(cuda_dev), "--log-level", "info"],
            stdout=open(os.path.join(WORKDIR, f"s2_gpu{cuda_dev}.log"), "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append(p)
        # Wait until the port is listening (server is up; it may still 503 under load).
        for _ in range(60):
            if tcp_alive("127.0.0.1", port):
                log(f"  backend cuda:{cuda_dev} (port {port}) listening")
                break
            if p.poll() is not None:
                log(f"  backend cuda:{cuda_dev} exited early (rc={p.returncode})")
                break
            time.sleep(1.0)
        else:
            log(f"  WARNING: backend cuda:{cuda_dev} not listening after 60s")


def stop_servers():
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


atexit.register(stop_servers)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, body, content_type):
        """Round-robin backends, retrying once on 503."""
        order = list(range(len(BACKENDS)))
        # start index rotates per request for fairness
        start = int(time.time() * 1000) % len(order)
        order = order[start:] + order[:start]
        last_status = 503
        last_headers = {}
        last_body = b""
        for i in order:
            cuda_dev, port = BACKENDS[i]
            url = f"http://127.0.0.1:{port}/generate"
            if not tcp_alive("127.0.0.1", port):
                log(f"backend cuda:{cuda_dev} down, skipping")
                continue
            req = urllib_request.Request(url, data=body, method="POST")
            if content_type:
                req.add_header("Content-Type", content_type)
            try:
                with urllib_request.urlopen(req, timeout=1800) as resp:
                    last_status = resp.status
                    last_headers = dict(resp.headers)
                    last_body = resp.read()
                    if last_status != 503:
                        return last_status, last_headers, last_body
            except HTTPError as e:
                last_status = e.code
                last_headers = dict(e.headers)
                try:
                    last_body = e.read()
                except Exception:
                    last_body = b""
                if last_status != 503:
                    return last_status, last_headers, last_body
            except URLError as e:
                log(f"backend cuda:{cuda_dev} urlerror: {e}")
                continue
        return last_status, last_headers, last_body

    def do_POST(self):
        if self.path.rstrip("/") in ("/generate", "/v1/tts"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            content_type = self.headers.get("Content-Type")
            status, headers, resp_body = self._forward(body, content_type)
            self.send_response_only(status)
            for k, v in headers.items():
                if k.lower() in ("transfer-encoding", "connection", "keep-alive"):
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return
        if self.path.rstrip("/") == "/stop":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            threading.Thread(target=stop_servers, daemon=True).start()
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            payload = {
                "backends": [
                    {"cuda": cuda_dev, "port": port,
                     "alive": tcp_alive("127.0.0.1", port)}
                    for cuda_dev, port in BACKENDS
                ],
                "model": S2_MODEL,
                "lb_port": LB_PORT,
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    signal.signal(signal.SIGINT, lambda *_: (stop_servers(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (stop_servers(), sys.exit(0)))
    log(f"Binary : {S2_BINARY}")
    log(f"Model  : {S2_MODEL}")
    log(f"Backends: {BACKENDS}")
    log(f"LB port : {LB_PORT}")
    launch_servers()
    log(f"Load balancer listening on 0.0.0.0:{LB_PORT} (POST /generate, GET /health)")
    httpd = ThreadingHTTPServer(("0.0.0.0", LB_PORT), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

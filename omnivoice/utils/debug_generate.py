import json
import os
import time
from typing import Any

import numpy as np
import soundfile as sf

from omnivoice import OmniVoiceGenerationConfig


SENT_END = {".", "!", "?", "。", "！", "？"}


def split_sentences(text: str) -> list[str]:
    sents, buf = [], ""
    for ch in text:
        buf += ch
        if ch in SENT_END:
            sents.append(buf.strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    return [s for s in sents if s]


def est_seconds(text: str, cps: float = 14.0) -> float:
    return len(text.replace(" ", "")) / cps


def chunk_15_30(
    text: str,
    min_s: float = 15.0,
    max_s: float = 30.0,
) -> list[str]:
    sents = split_sentences(text)
    durs = [est_seconds(s) for s in sents]
    n, chunks = len(sents), []
    i = 0
    while i < n:
        legal, acc = [], 0.0
        for j in range(i, n):
            acc += durs[j]
            if min_s < acc < max_s:
                legal.append(j)
            if acc >= max_s:
                break
        if legal:
            j = max(legal)
        else:
            acc = 0.0
            j = i
            while j < n and acc < min_s:
                acc += durs[j]
                j += 1
            j -= 1
        chunks.append(" ".join(sents[i : j + 1]))
        i = j + 1
    return chunks


def concat_with_gap(
    wavs: list[np.ndarray],
    gap_ms: float = 250,
    sr: int = 24000,
) -> np.ndarray:
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=wavs[0].dtype)
    out = [wavs[0]]
    for w in wavs[1:]:
        out.extend([gap, w])
    return np.concatenate(out)


def generate_with_debug(
    model: Any,
    text: str,
    ref_audio: Any,
    ref_text: str | None,
    instruct: str | None = None,
    gap_ms: float = 250,
    debug_dir: str | None = None,
    mode: str = "clone",
    language: str | None = None,
    generation_config: OmniVoiceGenerationConfig | None = None,
    speed: float | None = None,
    duration: float | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    if debug_dir is None:
        from datetime import datetime

        debug_dir = f"/kaggle/working/debug_chunks/{datetime.now().strftime('run_%H%M%S')}"
    os.makedirs(debug_dir, exist_ok=True)

    chunks = chunk_15_30(text)
    rows: list[dict[str, Any]] = []
    wavs: list[np.ndarray] = []

    for k, c in enumerate(chunks):
        est = est_seconds(c)
        t0 = time.time()
        kw: dict[str, Any] = dict(
            text=c,
            generation_config=generation_config,
        )
        if language:
            kw["language"] = language
        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)
        if mode == "clone":
            if ref_audio is None:
                raise ValueError("ref_audio is required for voice cloning.")
            prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            kw["voice_clone_prompt"] = prompt
        elif mode == "design":
            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

        audio = model.generate(**kw)
        wav = np.asarray(
            audio[0] if isinstance(audio, (list, tuple)) else audio,
            dtype=np.float32,
        )
        if hasattr(wav, "numpy"):
            wav = wav.numpy()
        wav = wav.squeeze().astype(np.float32)
        actual = len(wav) / 24000
        wavs.append(wav)

        path = os.path.join(debug_dir, f"chunk_{k:02d}.wav")
        sf.write(path, wav, 24000)

        flag = "" if 15 < actual < 30 else "  WARN OUT OF RANGE"
        rows.append(
            {
                "chunk": k,
                "est_s": round(est, 1),
                "actual_s": round(actual, 2),
                "gen_time_s": round(time.time() - t0, 1),
                "text_preview": c[:60] + ("…" if len(c) > 60 else ""),
                "file": path,
                "flag": flag,
            }
        )

    final = concat_with_gap(wavs, gap_ms=gap_ms)
    final_path = os.path.join(debug_dir, "final_stitched.wav")
    sf.write(final_path, final, 24000)

    with open(os.path.join(debug_dir, "manifest.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    return final, rows, debug_dir

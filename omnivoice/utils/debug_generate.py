"""Debug utilities for sentence-level TTS generation.

This module provides a sentence-by-sentence generation approach to isolate
issues that may arise from OmniVoice's built-in long-form chunking.
Instead of sending long text to ``model.generate()`` (which triggers
internal duration-based chunking), each sentence is generated as a
separate call with ``audio_chunk_threshold`` set very high.
"""

import json
import os
import time
from typing import Any

import numpy as np
import soundfile as sf

from omnivoice import OmniVoiceGenerationConfig


SAMPLE_RATE = 24000

SENTENCE_END = {".", "!", "?", "。", "！", "？"}


def split_into_sentences(text: str) -> list[str]:
    """Split text only at sentence-ending punctuation.

    The punctuation mark remains attached to the preceding sentence.

    Example:
        >>> split_into_sentences("Hello world. Goodbye!")
        ["Hello world.", "Goodbye!"]
    """
    sentences: list[str] = []
    buffer: list[str] = []

    for char in text.strip():
        buffer.append(char)
        if char in SENTENCE_END:
            sentence = "".join(buffer).strip()
            if sentence:
                sentences.append(sentence)
            buffer = []

    remaining = "".join(buffer).strip()
    if remaining:
        sentences.append(remaining)

    return sentences


def generate_one_sentence_at_a_time(
    model: Any,
    text: str,
    ref_audio: Any,
    ref_text: str | None = None,
    instruct: str | None = None,
    postprocess_output: bool = True,
) -> list[np.ndarray]:
    """Generate audio by calling ``model.generate`` once per sentence.

    OmniVoice's built-in long-form mode uses ``audio_chunk_duration`` and
    ``audio_chunk_threshold`` for duration-based chunking.  By sending one
    sentence per call and setting ``audio_chunk_threshold`` to a very high
    value, we bypass that behaviour entirely.
    """
    sentences = split_into_sentences(text)

    if not sentences:
        raise ValueError("No usable text was provided.")

    sentence_wavs: list[np.ndarray] = []

    for index, sentence in enumerate(sentences):
        print(f"Generating sentence {index + 1}/{len(sentences)}:")
        print(sentence)

        kwargs: dict[str, Any] = {
            "text": sentence,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "audio_chunk_threshold": 1e9,
            "postprocess_output": postprocess_output,
        }

        if instruct:
            kwargs["instruct"] = instruct

        result = model.generate(**kwargs)

        # OmniVoice may return [audio] or audio depending on the version.
        if isinstance(result, (list, tuple)):
            audio = result[0]
        else:
            audio = result

        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio, dtype=np.float32).squeeze()
        sentence_wavs.append(audio)

        print(
            f"Sentence {index + 1} audio duration: "
            f"{len(audio) / SAMPLE_RATE:.2f} seconds"
        )

    return sentence_wavs


def join_sentence_audio(
    sentence_wavs: list[np.ndarray],
    sample_rate: int = SAMPLE_RATE,
    gap_ms: int = 250,
) -> np.ndarray:
    """Concatenate sentence waveforms with a silent gap between them."""
    if not sentence_wavs:
        raise ValueError("No generated audio found.")

    gap_length = int(sample_rate * gap_ms / 1000)
    gap = np.zeros(gap_length, dtype=np.float32)

    pieces: list[np.ndarray] = []

    for index, audio in enumerate(sentence_wavs):
        if index > 0:
            pieces.append(gap)
        pieces.append(audio)

    return np.concatenate(pieces)


def tts_handler(
    model: Any,
    text: str,
    ref_audio: Any,
    ref_text: str | None = None,
    instruct: str | None = None,
    gap_ms: int = 250,
) -> tuple[tuple[int, np.ndarray], list[list[Any]]]:
    """One-shot handler that generates per-sentence and stitches the result.

    Returns ``((sample_rate, audio), rows)`` where ``rows`` is a list of
     ``[sentence_number, duration_seconds, "sentence"]`` entries.
    """
    sentence_wavs = generate_one_sentence_at_a_time(
        model=model,
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        instruct=instruct,
    )

    final_audio = join_sentence_audio(
        sentence_wavs,
        sample_rate=SAMPLE_RATE,
        gap_ms=gap_ms,
    )

    rows: list[list[Any]] = []
    for index, audio in enumerate(sentence_wavs):
        rows.append(
            [
                index + 1,
                round(len(audio) / SAMPLE_RATE, 2),
                "sentence",
            ]
        )

    return (SAMPLE_RATE, final_audio), rows

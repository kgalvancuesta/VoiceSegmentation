#!/usr/bin/env python3
"""
extract_embeddings.py

Extract DNN speaker embeddings + time segments from a single WAV file,
ready for downstream clustering (NOT done here).

Pipeline:
  1) Load WAV
  2) (Optional but recommended) VAD (silero-vad) -> speech segments
  3) Split speech into fixed-size windows
  4) Extract ECAPA-TDNN speaker embeddings (SpeechBrain pretrained)
  5) Write embeddings + timestamps to disk

Inputs/Outputs (updated for your repo layout):
  - Input WAV:  data/sample.wav
  - Output dir: output/
    - output/embeddings.npz : embeddings (float32) + start/end times
    - output/embeddings.csv : start_s,end_s,duration_s,emb_0,...,emb_D-1

Usage:
  python extract_embeddings.py

Dependencies:
  pip install torch torchaudio speechbrain pandas numpy tqdm

Notes:
  - silero-vad is fetched via torch.hub on first run (internet required).
  - SpeechBrain ECAPA model is downloaded on first run (internet required).
  - Embedding extraction runs on Apple Metal (MPS) only.
  - VAD is forced to CPU for stability.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from speechbrain.inference.speaker import EncoderClassifier


INPUT_WAV_PATH = os.path.join("data", "sample.wav")
OUTPUT_DIR = "output"
OUT_PREFIX = os.path.join(OUTPUT_DIR, "embeddings")

TARGET_SR = 16000

USE_VAD = True
VAD_THRESHOLD = 0.5
MIN_SPEECH_S = 0.25
MIN_SILENCE_S = 0.20
MERGE_GAP_S = 0.15

WINDOW_S = 1.5
HOP_S = 0.75

BATCH_SIZE = 32

EMBED_DEVICE_STR = "mps"


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float


def load_audio_mono(path: str) -> Tuple[torch.Tensor, int]:
    """Load audio and convert to mono float32 tensor shaped [T]."""
    wav, sr = torchaudio.load(path)
    wav = wav.to(torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    return wav, sr


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    if sr == target_sr:
        return wav
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    return resampler(wav)


def merge_close_segments(segments: List[Segment], gap_s: float) -> List[Segment]:
    """Merge segments separated by <= gap_s."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: s.start_s)
    out = [segs[0]]
    for s in segs[1:]:
        last = out[-1]
        if s.start_s - last.end_s <= gap_s:
            out[-1] = Segment(last.start_s, max(last.end_s, s.end_s))
        else:
            out.append(s)
    return out


def silero_vad_segments(
    wav_16k: torch.Tensor,
    sr: int,
    min_speech_s: float,
    min_silence_s: float,
    vad_threshold: float,
) -> List[Segment]:
    """
    Use Silero VAD to get speech timestamps.

    Returns list of [start_s, end_s] segments in seconds.
    """
    assert sr == 16000, "Silero VAD expects 16kHz audio."

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    (get_speech_timestamps, _, _, _, _) = utils

    model = model.to(torch.device("cpu"))
    audio_cpu = wav_16k.detach().cpu()

    speech_ts = get_speech_timestamps(
        audio_cpu,
        model,
        sampling_rate=sr,
        threshold=vad_threshold,
        min_speech_duration_ms=int(min_speech_s * 1000),
        min_silence_duration_ms=int(min_silence_s * 1000),
    )
    segments = []
    for ts in speech_ts:
        start_s = ts["start"] / sr
        end_s = ts["end"] / sr
        segments.append(Segment(start_s, end_s))
    return segments


def make_sliding_windows(segments: List[Segment], window_s: float, hop_s: float) -> List[Segment]:
    """
    Convert coarse speech segments into fixed-length windows for embedding extraction.
    Each window is [t, t+window_s]. Windows shorter than window_s are skipped.
    """
    out: List[Segment] = []
    for seg in segments:
        t = seg.start_s
        while t + window_s <= seg.end_s:
            out.append(Segment(t, t + window_s))
            t += hop_s
    return out


def extract_embeddings_ecapa(
    wav_16k: torch.Tensor,
    sr: int,
    windows: List[Segment],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Extract SpeechBrain ECAPA embeddings for each window.
    Returns numpy array [N, D].
    """
    assert sr == TARGET_SR

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )

    embeddings = []
    wav = wav_16k.to(device)

    def chunk_tensor(win: Segment) -> torch.Tensor:
        a = int(round(win.start_s * sr))
        b = int(round(win.end_s * sr))
        return wav[a:b]

    for i in tqdm(range(0, len(windows), batch_size), desc="Embedding", unit="batch"):
        batch_wins = windows[i: i + batch_size]
        batch = [chunk_tensor(w) for w in batch_wins]
        max_len = max(x.numel() for x in batch)
        batch_padded = torch.stack(
            [torch.nn.functional.pad(x, (0, max_len - x.numel())) for x in batch], dim=0
        )

        with torch.no_grad():
            emb = classifier.encode_batch(batch_padded)
            emb = emb.squeeze(1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        embeddings.append(emb.detach().cpu().numpy().astype(np.float32))

    if not embeddings:
        return np.zeros((0, 192), dtype=np.float32)

    return np.concatenate(embeddings, axis=0)


def save_outputs(out_prefix: str, windows: List[Segment], emb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    starts = np.array([w.start_s for w in windows], dtype=np.float32)
    ends = np.array([w.end_s for w in windows], dtype=np.float32)
    durs = ends - starts

    np.savez_compressed(
        out_prefix + ".npz",
        start_s=starts,
        end_s=ends,
        duration_s=durs,
        embeddings=emb,
    )

    base = pd.DataFrame({"start_s": starts, "end_s": ends, "duration_s": durs})

    if emb.size > 0:
        emb_df = pd.DataFrame(
            emb,
            columns=[f"emb_{j}" for j in range(emb.shape[1])],
        )
        df = pd.concat([base, emb_df], axis=1)
    else:
        df = base

    df.to_csv(out_prefix + ".csv", index=False)

def main() -> None:
    if not torch.backends.mps.is_built():
        raise RuntimeError("PyTorch was not built with MPS support. Install a macOS/MPS-enabled PyTorch build.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine. Are you on Apple Silicon / supported macOS?")

    embed_device = torch.device(EMBED_DEVICE_STR)

    if not os.path.exists(INPUT_WAV_PATH):
        raise FileNotFoundError(f"Input WAV not found: {INPUT_WAV_PATH}")

    wav, sr = load_audio_mono(INPUT_WAV_PATH)
    wav = resample_if_needed(wav, sr, TARGET_SR)
    sr = TARGET_SR

    if USE_VAD:
        segments = silero_vad_segments(
            wav_16k=wav,
            sr=sr,
            min_speech_s=MIN_SPEECH_S,
            min_silence_s=MIN_SILENCE_S,
            vad_threshold=VAD_THRESHOLD,
        )
        segments = merge_close_segments(segments, gap_s=MERGE_GAP_S)
    else:
        total_s = wav.numel() / sr
        segments = [Segment(0.0, float(total_s))]

    windows = make_sliding_windows(segments, window_s=WINDOW_S, hop_s=HOP_S)
    if len(windows) == 0:
        raise RuntimeError(
            "No windows produced. Try lowering MIN_SPEECH_S / increasing VAD sensitivity "
            "or temporarily disable VAD (USE_VAD=False) to debug."
        )

    emb = extract_embeddings_ecapa(
        wav_16k=wav,
        sr=sr,
        windows=windows,
        device=embed_device,
        batch_size=BATCH_SIZE,
    )

    if emb.shape[0] != len(windows):
        raise RuntimeError(f"Embedding count mismatch: windows={len(windows)} emb={emb.shape[0]}")

    save_outputs(OUT_PREFIX, windows, emb)

    print("Done.")
    print(f"Input:  {INPUT_WAV_PATH}")
    print(f"Wrote:  {OUT_PREFIX}.npz")
    print(f"Wrote:  {OUT_PREFIX}.csv")
    print(f"Windows: {len(windows)} | Embedding dim: {emb.shape[1] if emb.size else 'n/a'}")


if __name__ == "__main__":
    main()

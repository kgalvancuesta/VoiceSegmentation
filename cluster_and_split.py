#!/usr/bin/env python3
"""
cluster_and_split_gmm.py

Robust clustering + separation for 2-speaker turn-taking audio using:
  - PCA -> (optional whitening)
  - Gaussian Mixture Model (2 components)
  - Viterbi smoothing with a strong self-transition bias (HMM-ish)

Inputs:
  - data/sample.wav
  - output/embeddings.npz  (from extract_embeddings.py)

Outputs:
  - output/speaker_0.wav
  - output/speaker_1.wav

Notes:
  - speaker_0 vs speaker_1 is arbitrary (cluster labels can swap).
  - If the two voices are truly extremely similar, you may need longer embedding windows
    (e.g. WINDOW_S=3.0, HOP_S=1.5 in the extraction step) — but try this first.

Deps:
  pip install scikit-learn torch torchaudio numpy
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torchaudio

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


# ---- Fixed project layout ----
INPUT_WAV_PATH = os.path.join("data", "sample.wav")
EMB_NPZ_PATH = os.path.join("output", "embeddings.npz")
OUTPUT_DIR = "output"
OUT_SPK0 = os.path.join(OUTPUT_DIR, "speaker_0.wav")
OUT_SPK1 = os.path.join(OUTPUT_DIR, "speaker_1.wav")

# ---- Feature / model params ----
N_SPEAKERS = 2
PCA_DIMS = 64              # reduce 192 -> 64 (robust for GMM)
PCA_WHITEN = False         # keep False unless you know you want whitening

GMM_N_INIT = 5
GMM_REG_COVAR = 1e-4       # stabilizes when speakers are close
GMM_COV_TYPE = "full"      # 'diag' is faster but often worse

# ---- Temporal smoothing (Viterbi) ----
# Bigger value => fewer speaker switches.
# Interpreted as log self-transition bonus added each step.
SWITCH_PENALTY = 2.0

# ---- Segment cleanup / audio write ----
MIN_SEGMENT_S = 0.20       # drop tiny fragments
PAD_S = 0.02               # pad each segment edges
FADE_MS = 8                # fade per extracted chunk to avoid clicks


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    label: int


def load_audio_mono(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)  # [C, T]
    wav = wav.to(torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    return wav, sr


def infer_window_and_hop(start_s: np.ndarray, end_s: np.ndarray) -> Tuple[float, float]:
    dur = end_s - start_s
    window_s = float(np.median(dur))
    if len(start_s) < 2:
        raise RuntimeError("Not enough windows to infer hop size.")
    ds = np.diff(np.sort(start_s))
    ds = ds[ds > 1e-6]
    if ds.size == 0:
        raise RuntimeError("Could not infer hop size from timestamps.")
    hop_s = float(np.median(ds))
    return window_s, hop_s


def fit_gmm_with_pca(emb: np.ndarray) -> Tuple[GaussianMixture, np.ndarray]:
    """
    PCA reduces dimensionality (stabilizes GMM), then fit 2-component GMM.
    Returns (gmm, emb_pca).
    """
    if emb.ndim != 2 or emb.shape[0] < 10:
        raise RuntimeError(f"Embeddings look wrong: shape={emb.shape}")

    d = min(PCA_DIMS, emb.shape[1])
    pca = PCA(n_components=d, whiten=PCA_WHITEN, random_state=0)
    z = pca.fit_transform(emb)

    gmm = GaussianMixture(
        n_components=N_SPEAKERS,
        covariance_type=GMM_COV_TYPE,
        reg_covar=GMM_REG_COVAR,
        n_init=GMM_N_INIT,
        random_state=0,
    )
    gmm.fit(z)
    return gmm, z


def viterbi_decode(log_likelihood: np.ndarray, switch_penalty: float) -> np.ndarray:
    """
    2-state Viterbi with a self-transition bonus (equivalently switch penalty).

    log_likelihood: [T, 2] where ll[t, k] = log p(obs_t | state=k)
    """
    T, K = log_likelihood.shape
    assert K == 2

    # Transition log-probs (unnormalized; we only need relative differences)
    # Staying: +switch_penalty, switching: +0
    trans = np.array([[switch_penalty, 0.0],
                      [0.0, switch_penalty]], dtype=np.float64)

    dp = np.empty((T, K), dtype=np.float64)
    bp = np.empty((T, K), dtype=np.int32)

    # init with equal priors
    dp[0] = log_likelihood[0]
    bp[0] = 0

    for t in range(1, T):
        for k in range(K):
            scores = dp[t - 1] + trans[:, k]
            j = int(np.argmax(scores))
            dp[t, k] = scores[j] + log_likelihood[t, k]
            bp[t, k] = j

    path = np.empty(T, dtype=np.int32)
    path[T - 1] = int(np.argmax(dp[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def windows_to_segments(
    start_s: np.ndarray,
    end_s: np.ndarray,
    labels: np.ndarray,
    min_segment_s: float,
) -> List[Segment]:
    """
    Merge consecutive windows of the same label into segments on the original timeline.
    Uses window boundaries (start/end) directly.
    """
    idx = np.argsort(start_s)
    start_s = start_s[idx]
    end_s = end_s[idx]
    labels = labels[idx]

    segs: List[Segment] = []
    cur_lab = int(labels[0])
    cur_start = float(start_s[0])
    cur_end = float(end_s[0])

    for s, e, lab in zip(start_s[1:], end_s[1:], labels[1:]):
        lab = int(lab)
        s = float(s)
        e = float(e)
        if lab == cur_lab and s <= cur_end + 1e-3:
            cur_end = max(cur_end, e)
        else:
            if (cur_end - cur_start) >= min_segment_s:
                segs.append(Segment(cur_start, cur_end, cur_lab))
            cur_lab = lab
            cur_start = s
            cur_end = e

    if (cur_end - cur_start) >= min_segment_s:
        segs.append(Segment(cur_start, cur_end, cur_lab))

    return segs


def apply_fade(x: torch.Tensor, sr: int, fade_ms: float) -> torch.Tensor:
    if fade_ms <= 0:
        return x
    fade_len = int(round(sr * fade_ms / 1000.0))
    if fade_len <= 0 or x.numel() < 2 * fade_len + 1:
        return x
    ramp = torch.linspace(0.0, 1.0, fade_len, dtype=x.dtype, device=x.device)
    y = x.clone()
    y[:fade_len] *= ramp
    y[-fade_len:] *= ramp.flip(0)
    return y


def extract_concat(wav: torch.Tensor, sr: int, segs: List[Segment], label: int) -> torch.Tensor:
    n = wav.numel()
    chunks: List[torch.Tensor] = []
    total_len_s = float(n) / sr

    for seg in segs:
        if seg.label != label:
            continue
        s = max(0.0, seg.start_s - PAD_S)
        e = min(total_len_s, seg.end_s + PAD_S)
        a = int(round(s * sr))
        b = int(round(e * sr))
        a = max(0, min(n, a))
        b = max(0, min(n, b))
        if b <= a:
            continue
        chunk = apply_fade(wav[a:b], sr, FADE_MS)
        chunks.append(chunk)

    if not chunks:
        return torch.zeros(0, dtype=wav.dtype)
    return torch.cat(chunks, dim=0)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_WAV_PATH):
        raise FileNotFoundError(f"Missing input WAV: {INPUT_WAV_PATH}")
    if not os.path.exists(EMB_NPZ_PATH):
        raise FileNotFoundError(f"Missing embeddings NPZ: {EMB_NPZ_PATH} (run extract_embeddings.py first)")

    # Load data
    wav, sr = load_audio_mono(INPUT_WAV_PATH)
    audio_len_s = float(wav.numel()) / sr

    d = np.load(EMB_NPZ_PATH)
    start_s = d["start_s"].astype(np.float64)
    end_s = d["end_s"].astype(np.float64)
    emb = d["embeddings"].astype(np.float32)

    window_s, hop_s = infer_window_and_hop(start_s, end_s)
    print(f"Inferred window_s={window_s:.3f}s hop_s={hop_s:.3f}s | windows={len(start_s)}")

    # Fit PCA+GMM
    gmm, z = fit_gmm_with_pca(emb)

    # Per-window log-likelihoods
    ll = gmm.score_samples(z)  # total mixture ll, not per component
    # We need per-component responsibilities -> use predict_proba with log safety:
    resp = gmm.predict_proba(z).astype(np.float64)  # [N,2]
    eps = 1e-12
    log_resp = np.log(np.clip(resp, eps, 1.0))      # proxy for per-component evidence

    # Viterbi smoothing (turn-taking prior: speakers stick for a while)
    path = viterbi_decode(log_resp, switch_penalty=SWITCH_PENALTY)

    # Quick diagnostics
    u, c = np.unique(path, return_counts=True)
    print("Post-Viterbi counts:", dict(zip(u.tolist(), c.tolist())))

    # Build segments and write WAVs
    segs = windows_to_segments(start_s, end_s, path, min_segment_s=MIN_SEGMENT_S)
    if not segs:
        raise RuntimeError("No segments after merging. Try lowering MIN_SEGMENT_S.")

    spk0 = extract_concat(wav, sr, segs, label=0)
    spk1 = extract_concat(wav, sr, segs, label=1)

    torchaudio.save(OUT_SPK0, spk0.unsqueeze(0).cpu(), sr)
    torchaudio.save(OUT_SPK1, spk1.unsqueeze(0).cpu(), sr)

    dur0 = float(spk0.numel()) / sr if spk0.numel() else 0.0
    dur1 = float(spk1.numel()) / sr if spk1.numel() else 0.0

    print("Done.")
    print(f"Wrote: {OUT_SPK0} (duration ~{dur0:.1f}s)")
    print(f"Wrote: {OUT_SPK1} (duration ~{dur1:.1f}s)")
    print("Note: speaker_0 vs speaker_1 is arbitrary; listen once and rename if needed.")


if __name__ == "__main__":
    main()
# VoiceSegmentation
Barebones 2-speaker voice separation for a single WAV file. It extracts speaker embeddings (SpeechBrain ECAPA) on speech-only windows, then clusters the windows with a 2-component GMM and smooths the labels to split the audio into two files.

## I/O
- Input: `data/sample.wav`
- Output:
  - `output/embeddings.npz` and `output/embeddings.csv` (window timestamps + embeddings)
  - `output/speaker_0.wav` and `output/speaker_1.wav` (separated voices)

## Tutorial
1) Create the environment:
```bash
conda env create -f environment.yml
conda activate voice_segmentation
```
2) Put your mono or stereo file at `data/sample.wav`.
3) Run:
```bash
python extract_embeddings.py
python cluster_and_split.py
```

Notes:
- The first run downloads Silero VAD and SpeechBrain DNN models (internet required).
- Embedding extraction is set up for Apple Silicon MPS in `extract_embeddings.py`.

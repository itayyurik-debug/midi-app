---
title: Local Transcription Pipeline
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Transcription Pipeline (Demucs + Basic Pitch)

Upload an audio file, pick an instrument, and get back a transcribed MIDI file.
Demucs separates the stem first, then Basic Pitch transcribes it with tuned
settings (onset 0.6, frame 0.5, minimum note length 127.7ms).

Runs on the free CPU tier — no GPU, so it's slower than a local machine with
a good CPU, and may be slow or time out on long tracks. Try a short clip first.

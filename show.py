import json
import os
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf
from pydub import AudioSegment
import numpy as np
import soundfile as sf

import matplotlib.pyplot as plt

audio_wav = "conversation.wav"
output_dir = os.path.join("./", 'oracle_vad')

rttm_file = os.path.join(output_dir, "pred_rttms", os.path.basename(audio_wav).replace(".wav",".rttm"))
with open(rttm_file, "r") as f:
    lines = f.readlines()

speaker_segments = {}
for line in lines:
    parts = line.strip().split()
    start = float(parts[3])
    duration = float(parts[4])
    speaker = parts[7]
    if speaker not in speaker_segments:
        speaker_segments[speaker] = []
    speaker_segments[speaker].append((start, start+duration))


waveform, sr = sf.read(audio_wav)
time = np.linspace(0, len(waveform)/sr, num=len(waveform))

fig, ax = plt.subplots(figsize=(15,4))

# waveform
ax.plot(time, waveform, color='gray', alpha=0.5)

# overlay speaker segments
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
for i, (spk, segs) in enumerate(speaker_segments.items()):
    for start, end in segs:
        ax.axvspan(start, end, ymin=0, ymax=1, color=colors[i%len(colors)], alpha=0.3, label=spk)


handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.set_title("Speaker Diarization with Audio Waveform")

plt.savefig("speaker_diarization_waveform.png", dpi=150)
print("Zapisano wykres do speaker_diarization_waveform.png")
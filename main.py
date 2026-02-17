
import soundfile as sf
import torch

signal, fs = sf.read("conversation.mp3")


import numpy as np
from speechbrain.inference import SpeakerRecognition
from speechbrain.processing.diarization import Spec_Cluster

window_size = int(fs * 1.0)  # segment 1s
segments = [signal[i:i+window_size] for i in range(0, len(signal), window_size)]

# 3) Wyodrębnij embeddingi
spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec"
)

embeddings = []
for seg in segments:
    # convert to mono and proper shape
    emb = spkrec.encode_batch(torch.from_numpy(seg).unsqueeze(0)).squeeze().detach().numpy()
    embeddings.append(emb)

embeddings = np.stack(embeddings)

# 4) Klasteryzacja spektralna (np. 2 mówców)
clusterer = Spec_Cluster(
    embed=embeddings,
    num_speakers=2
)
labels = clusterer.run()

# 5) Wypisz segmenty z etykietami
for i, label in enumerate(labels):
    start = i * 1.0
    end = start + 1.0
    print(f"Speaker {int(label)+1}: {start:.1f}s–{end:.1f}s")

import json
import os
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf
from pydub import AudioSegment

audio = AudioSegment.from_file("conversation.mp3")
audio = audio.set_channels(1)  # mono
audio.export("conversation.wav", format="wav")

# manifest
with open('conf/input_manifest.json','w') as fp:
    fp.write(json.dumps(
        {
        "audio_filepath": "conversation.wav",
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": None,
        "rttm_filepath": None,
        "uem_filepath": None
    }))

# config
config = OmegaConf.load("conf/inference/diar_infer_general.yaml")

output_dir = os.path.join("./", 'oracle_vad')
os.makedirs(output_dir,exist_ok=True)

config.diarizer.manifest_filepath = 'conf/input_manifest.json'
config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
pretrained_speaker_model = 'titanet_large'
config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5]
config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1]
config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1]

config.diarizer.oracle_vad = False
config.diarizer.clustering.parameters.oracle_num_speakers = False
config.diarizer.num_workers = 12
config.diarizer.vad.num_workers = 0

model = ClusteringDiarizer(cfg=config)

model.diarize()

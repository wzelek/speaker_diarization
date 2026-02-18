import os
import json
import tempfile
import streamlit as st
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def parse_rttm(rttm_path):
    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append((start, start + duration, speaker))
    return segments

@st.cache_resource
def get_waveform(wav_path):
    audio_data, sr = sf.read(wav_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    return audio_data, sr

st.set_page_config(page_title="Speaker Diarization", layout="centered")

st.title("Speaker Diarization")


st.sidebar.header("Upload Audio")

uploaded_file = st.sidebar.file_uploader(
    "Upload an audio file",
    type=["mp3", "mp4", "mpeg", "wav"]
)

DEFAULT_FILE = "conversation.mp3"
OUTPUT_DIR = "diarization_output"
CONFIG_DIR = "conf/inference"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dynamically list YAML config files
if os.path.exists(CONFIG_DIR):
    config_files = [
        f for f in os.listdir(CONFIG_DIR)
        if f.endswith(".yaml") or f.endswith(".yml")
    ]
else:
    config_files = []

if not config_files:
    st.sidebar.error("No config files found in conf/inference/")
    st.stop()


selected_config = st.sidebar.selectbox(
        "Select Diarization Config",
        config_files
)


if st.sidebar.button("Run Diarization"):

    with st.spinner("Processing audio...", show_time=True):

        with tempfile.TemporaryDirectory() as tmpdir:

            if uploaded_file is not None:
                input_path = os.path.join(tmpdir, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.read())
            else:
                if not os.path.exists(DEFAULT_FILE):
                    st.error("No file uploaded and default conversation.mp3 not found.")
                    st.stop()

                input_path = DEFAULT_FILE
                st.info("Using default file: conversation.mp3")


            os.system(f"rm -rf {OUTPUT_DIR}") if os.path.exists(OUTPUT_DIR) else None
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            wav_path = os.path.join(tmpdir, "input.wav")

            # Convert to mono WAV
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1)
            audio.export(wav_path, format="wav")

            # Create manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
          
            with open(manifest_path, "w") as fp:
                json.dump({
                    "audio_filepath": wav_path,
                    "offset": 0,
                    "duration": None,
                    "label": "infer",
                    "text": "-",
                    "num_speakers": None,
                    "rttm_filepath": None,
                    "uem_filepath": None
                }, fp)

            # Load NeMo config
            config_path = os.path.join(CONFIG_DIR, selected_config)
            config = OmegaConf.load(config_path)

            config.diarizer.manifest_filepath = manifest_path
            config.diarizer.out_dir = OUTPUT_DIR

            # Run diarization
            model = ClusteringDiarizer(cfg=config)
            model.diarize()

            # Show results
            rttm_file = os.path.join(
                OUTPUT_DIR,
                "pred_rttms",
                os.path.basename(wav_path).replace(".wav", ".rttm")
            )

            if rttm_file and os.path.exists(rttm_file):

                st.toast("Diarization complete ✅")

                with open(rttm_file, "r") as f:
                    rttm_content = f.read()

                audio_data, sr = get_waveform(wav_path)

                time_axis = np.linspace(0, len(audio_data)/sr, num=len(audio_data))
                segments = parse_rttm(rttm_file)
                speakers = list(set([s[2] for s in segments]))

                # Map speakers to colors
                speaker_colors = {spk: f"rgba({i*50%255},{i*80%255},{i*120%255},0.4)" 
                                for i, spk in enumerate(speakers)}

                # Plot waveform
                fig = go.Figure()

                ds = 100  # every 100th sample
                time_axis = time_axis[::ds]
                audio_display = audio_data[::ds]

                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=audio_display,
                    mode='lines',
                    line=dict(color='lightgrey'),
                    name='Waveform'
                ))

                # Add speaker regions
                for start, end, speaker in segments:
                    fig.add_vrect(
                        x0=start,
                        x1=end,
                        fillcolor=speaker_colors[speaker],
                        opacity=0.4,
                        line_width=0,
                        annotation_text=speaker,
                        annotation_position="top left"
                    )

                fig.update_layout(
                    title="🎙️ Speaker Timeline (interactive)",
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    showlegend=False,
                    height=400
                )

                plot_container = st.container()
                with plot_container:
                    st.plotly_chart(fig, use_container_width=True)

                # Audio playback
                st.audio(wav_path)

            else:
                st.error("No RTTM file generated.")

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
import pandas as pd

DEFAULT_FILE = "conversation.mp3"
OUTPUT_DIR = "diarization_output"
CONFIG_DIR = "conf/inference"

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

def rttm_to_dataframe(rttm_content):
    """
    Convert RTTM content to a Pandas DataFrame with columns:
    ['Speaker', 'Start (s)', 'End (s)', 'Duration (s)', 'Start (MM:SS)', 'End (MM:SS)']
    """
    def sec_to_mmss(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    
    rows = []
    for line in rttm_content.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 8:
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]
            rows.append({
                "Speaker": speaker,
                "Start (s)": round(start, 2),
                "Start (MM:SS)": sec_to_mmss(start),
                "End (s)": round(end, 2),
                "End (MM:SS)": sec_to_mmss(end),
                "Duration (s)": round(duration, 2),
            })
    return pd.DataFrame(rows)


@st.cache_resource
def get_waveform(wav_path):
    audio_data, sr = sf.read(wav_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    return audio_data, sr

def show_results(results):

    # Inside your diarization block after results are loaded
    df_rttm = rttm_to_dataframe(results["rttm_content"])
    st.subheader("Diarization Segments")
    st.dataframe(df_rttm)

    st.download_button(
        label="Download RTTM",
        data=results["rttm_content"],
        file_name="diarization.rttm",
        mime="text/plain"
    )

    # Show results
    rttm_file = os.path.join(
        OUTPUT_DIR,
        "pred_rttms",
        os.path.basename(results["wav_path"]).replace(".wav", ".rttm")
    )

    audio_data, sr = get_waveform(results["wav_path"])

    time_axis = np.linspace(0, len(audio_data)/sr, num=len(audio_data))
    segments = parse_rttm(rttm_file)
    speakers = list(set([s[2] for s in segments]))

    # Map speakers to colors
    speaker_colors = {spk: f"rgba({i*50%255},{i*80%255},{i*120%255},0.4)" 
                    for i, spk in enumerate(speakers)}


    # Prepare hover speaker array
    speaker_hover = [""] * len(time_axis)
    for start, end, speaker in segments:
        # find indices in time_axis that are inside this segment
        indices = np.where((time_axis >= start) & (time_axis <= end))[0]
        for idx in indices:
            speaker_hover[idx] = speaker

    fig = go.Figure()

    ds = 1000  # every 10th sample
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
        title="Speaker Timeline (interactive)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
        height=400
    )

    plot_container = st.container()
    with plot_container:
        st.plotly_chart(fig, use_container_width=True)

    # Audio playback
    st.audio(results["wav_path"])

def show():

    st.set_page_config(page_title="Speaker Diarization", layout="wide")

    st.title("Speaker Diarization")

    st.sidebar.header("Upload Audio")

    uploaded_file = st.sidebar.file_uploader(
        "Upload an audio file",
        type=["mp3", "mp4", "mpeg", "wav"]
    )

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
            config_files, index=1
    )

    clicked = st.sidebar.button("Run Diarization")
    if clicked or st.session_state.get("diarization_results") is not None:

        prev_diarization_results = st.session_state.get("diarization_results") 

        if prev_diarization_results is None or clicked:

            with st.spinner("Processing audio...", show_time=True):
                
                prev_diarization_results = st.session_state.get("diarization_results") 
                if prev_diarization_results: 
                    os.system(f"rm -rf {prev_diarization_results['tmpdir']}")

                with tempfile.TemporaryDirectory(delete=False) as tmpdir:

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

                    rttm_file = os.path.join(
                        OUTPUT_DIR, "pred_rttms",
                        os.path.basename(wav_path).replace(".wav", ".rttm")
                    )

                    if not os.path.exists(rttm_file):
                        st.error("No RTTM file generated.")
                        st.stop()

                    with open(rttm_file, "r") as f:
                        rttm_content = f.read()

                    audio_data, sr = get_waveform(wav_path)
                    segments = parse_rttm(rttm_file)

                    st.session_state.diarization_results = {
                        "wav_path": wav_path,
                        "rttm_content": rttm_content,
                        "audio_data": audio_data,
                        "sr": sr,
                        "segments": segments, 
                        "tmpdir": tmpdir
                    }

                    # Show RTTM
                    st.toast("Diarization complete ✅")


        results = st.session_state.diarization_results
        show_results(results)

show()

import os
import json
import tempfile
from typing import List, Tuple, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import soundfile as sf
import plotly.graph_objs as go
from pydub import AudioSegment
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from pathlib import Path

DEFAULT_FILE = "conversation.mp3"
OUTPUT_DIR = "diarization_output"
CONFIG_DIR = "conf"

def parse_rttm(rttm_path: str) -> List[Tuple[float, float, str]]:
    """Parse RTTM file into list of segments (start, end, speaker)."""
    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append((start, start + duration, speaker))
    return segments

import pandas as pd

def rttm_to_dataframe(rttm_content: str) -> pd.DataFrame:
    """
    Convert RTTM content to a DataFrame with dynamic speaker labeling (A, B, C...).
    Format: [segment, label, speaker, start, end]
    """
    
    def format_timestamp(seconds: float) -> str:
        """Helper to format seconds into HH:MM:SS.mmm"""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    rows = []
    for line in rttm_content.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 8:
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker_raw = parts[7] # e.g., "speaker_1"

            try:
                spk_idx = int(speaker_raw.split('_')[-1])
                label = chr(65 + spk_idx) 
            except (ValueError, IndexError):
                label = speaker_raw 
            
            segment_str = f"[ {format_timestamp(start)} --> {format_timestamp(end)}]"
            
            rows.append({
                "segment": segment_str,
                "label": label,
                "speaker": speaker_raw.upper().replace("SPEAKER_", "SPEAKER_0"), 
                "start": round(start, 6),
                "end": round(end, 6)
            })
            
    return pd.DataFrame(rows)

@st.cache_resource
def load_audio(wav_path: str) -> Tuple[np.ndarray, int]:
    """Load audio waveform."""
    audio_data, sr = sf.read(wav_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    return audio_data, sr

def plot_waveform_with_speakers(audio_data: np.ndarray, sr: int, segments: List[Tuple[float, float, str]]) -> go.Figure:
    """Create interactive waveform with speaker regions."""
    time_axis = np.linspace(0, len(audio_data)/sr, num=len(audio_data))
    speakers = list({s[2] for s in segments})
    colors = {spk: f"rgba({i*50%255},{i*80%255},{i*120%255},0.4)" for i, spk in enumerate(speakers)}

    # Downsample for plotting performance
    ds = max(len(audio_data)//5000, 1)
    time_axis_ds = time_axis[::ds]
    audio_ds = audio_data[::ds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis_ds,
        y=audio_ds,
        mode='lines',
        line=dict(color='lightgrey'),
        name='Waveform'
    ))

    # Add speaker regions
    for start, end, speaker in segments:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=colors[speaker],
            opacity=0.4,
            line_width=0,
            annotation_text=speaker,
            annotation_position="top left"
        )

    fig.update_layout(
        title="Speaker Timeline",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=False,
        height=400
    )
    return fig


def run_diarization(input_file: str, config_file: str, reference_rttm: str = None) -> Dict[str, Any]:
    """Run NeMo Clustering Diarizer and return results."""
    tmpdir = tempfile.mkdtemp()
    wav_path = os.path.join(tmpdir, "input.wav")

    # Convert audio to mono WAV
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1)
    audio.export(wav_path, format="wav")
   
    if reference_rttm:
        rttm_path_for_manifest = os.path.join(tmpdir, reference_rttm.name)
        with open(rttm_path_for_manifest, "wb") as f:
            f.write(reference_rttm.getbuffer())
            
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
            "rttm_filepath": rttm_path_for_manifest if reference_rttm else None,
            "uem_filepath": None
        }, fp)

    # Load NeMo config
    st.toast("config_file: " + config_file)
    config = OmegaConf.load(config_file)
    config.diarizer.manifest_filepath = manifest_path
    config.diarizer.out_dir = OUTPUT_DIR

    # Run diarization
    model = ClusteringDiarizer(cfg=config)
    model.diarize()

    # Load RTTM
    rttm_file = os.path.join(OUTPUT_DIR, "pred_rttms", os.path.basename(wav_path).replace(".wav", ".rttm"))
    if not os.path.exists(rttm_file):
        raise FileNotFoundError("No RTTM file generated.")
    
    with open(rttm_file, "r") as f:
        rttm_content = f.read()

    audio_data, sr = load_audio(wav_path)
    segments = parse_rttm(rttm_file)

    return {
        "input_file": input_file,
        "wav_path": wav_path,
        "rttm_content": rttm_content,
        "audio_data": audio_data,
        "sr": sr,
        "segments": segments,
        "tmpdir": tmpdir
    }

def display_results(results: Dict[str, Any]):
    """Display RTTM table, waveform, and audio player."""
    df_rttm = rttm_to_dataframe(results["rttm_content"])
    st.subheader("Diarization Segments")
    st.dataframe(df_rttm)


    rttm_filename = os.path.splitext(os.path.basename(results['input_file']))[0] + ".rttm"

    st.download_button(
        label="Download RTTM",
        data=results["rttm_content"],
        file_name=rttm_filename,
        mime="text/plain"
    )

    fig = plot_waveform_with_speakers(results["audio_data"], results["sr"], results["segments"])
    st.plotly_chart(fig, use_container_width=True)
    st.audio(results["wav_path"])


def main():
    st.set_page_config(page_title="Speaker Diarization", layout="wide")
    st.title("Speaker Diarization")

    st.sidebar.header("Upload Audio")
    uploaded_file = st.sidebar.file_uploader(
        "Upload an audio file", type=["mp3", "mp4", "mpeg", "wav"]
    )

    uploaded_rttm = st.sidebar.file_uploader("Upload reference RTTM file (optional)", type=["rttm"])

    dir = Path(CONFIG_DIR)
    # Config files dropdown
    config_files = [str(f.relative_to(dir)) for f in dir.rglob("*.yaml")]  # include .yaml
    config_files.sort()

    if not config_files:
        st.sidebar.error("No config files found in conf/inference/")
        st.stop()
    selected_config = st.sidebar.selectbox("Select Diarization Config", config_files, index=0)

    clicked = st.sidebar.button("Run Diarization")

    # Determine input file
    input_file = uploaded_file.name if uploaded_file else DEFAULT_FILE
    if uploaded_file:
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1])
        tmp_input.write(uploaded_file.read())
        tmp_input.close()
        input_file = tmp_input.name
        st.info(f"Using uploaded file: {uploaded_file.name}")
    else:
        if not os.path.exists(DEFAULT_FILE):
            st.error("Default conversation.mp3 not found.")
            st.stop()
        st.info(f"Using default file: {DEFAULT_FILE}")

    # Run diarization or fetch cached results
    if clicked or "diarization_results" in st.session_state:
        if clicked or "diarization_results" not in st.session_state:
            with st.spinner("Processing audio...", show_time=True):
                st.session_state.diarization_results = run_diarization(
                    input_file,
                    os.path.join(CONFIG_DIR, selected_config), 
                    reference_rttm=uploaded_rttm
                )
                st.toast("✅ Diarization complete!")

        display_results(st.session_state.diarization_results)

if __name__ == "__main__":
    main()

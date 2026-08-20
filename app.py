from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from src.config import load_config
from src.pipeline import AccidentTrajectoryPipeline


st.set_page_config(page_title="Accident & Trajectory Detector", layout="wide")
st.title("Road Accident Detection and Vehicle Trajectory Analysis")
st.caption("Upload a CCTV/webcam recording. Green paths are before the selected accident frame; red paths are after it.")

uploaded = st.file_uploader("Upload MP4, AVI, MOV, or MKV video", type=["mp4", "avi", "mov", "mkv"])
config_path = st.text_input("Configuration file", value="config.yaml")

if uploaded is not None and st.button("Analyse video", type="primary"):
    config = load_config(config_path)
    suffix = Path(uploaded.name).suffix or ".mp4"

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        input_path = temporary / f"input{suffix}"
        output_path = temporary / "analysed.mp4"
        report_path = temporary / "report.json"
        input_path.write_bytes(uploaded.getbuffer())

        progress = st.progress(0.0)
        status = st.empty()

        def update_progress(current: int, total: int | None) -> None:
            if total:
                progress.progress(min(1.0, current / total))
                status.write(f"Processed {current}/{total} frames")
            else:
                status.write(f"Processed {current} frames")

        try:
            pipeline = AccidentTrajectoryPipeline(config)
            report = pipeline.process(
                str(input_path),
                output_path,
                report_path,
                display=False,
                progress_callback=update_progress,
            )
        except Exception as error:
            st.exception(error)
        else:
            progress.progress(1.0)
            status.success(f"Complete. Detected {len(report['events'])} accident event(s).")
            video_bytes = output_path.read_bytes()
            report_bytes = report_path.read_bytes()

            left, right = st.columns([2, 1])
            with left:
                st.video(video_bytes)
            with right:
                st.subheader("Event report")
                st.json(report)

            st.download_button(
                "Download analysed video",
                data=video_bytes,
                file_name="analysed_accident_video.mp4",
                mime="video/mp4",
            )
            st.download_button(
                "Download JSON report",
                data=report_bytes,
                file_name="accident_report.json",
                mime="application/json",
            )
else:
    st.info("Train the CNN first, then upload a video. For live webcam mode run: python run_webcam.py")

with st.expander("Important limitation"):
    st.write(
        "The supplied Kaggle dataset contains still accident/non-accident frames. "
        "Therefore this project uses temporal probability smoothing for video. "
        "For research-grade accident timing and vehicle involvement, fine-tune on annotated accident videos."
    )

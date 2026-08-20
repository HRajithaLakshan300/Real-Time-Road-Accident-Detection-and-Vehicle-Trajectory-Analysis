from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.pipeline import AccidentTrajectoryPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live webcam accident and trajectory detection.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--save", action="store_true", help="Save the annotated webcam recording.")
    args = parser.parse_args()

    config = load_config(args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"webcam_{stamp}.mp4"
    report = output_dir / f"webcam_{stamp}.json"

    pipeline = AccidentTrajectoryPipeline(config)
    pipeline.process(args.camera, output, report, display=True)
    print("Press q in the video window to stop.")
    print(f"Recording: {output}")
    print(f"Report: {report}")
    if not args.save and output.exists():
        print("Note: the pipeline records by default so event evidence is preserved.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.pipeline import AccidentTrajectoryPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse an uploaded/recorded traffic video.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    config = load_config(args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or Path(config.project.output_dir) / f"analysed_{stamp}.mp4"
    report = output.with_suffix(".json")

    pipeline = AccidentTrajectoryPipeline(config)
    result = pipeline.process(str(args.input), output, report, display=args.show)
    print(f"Processed video: {output}")
    print(f"JSON report: {report}")
    print(f"Detected events: {len(result['events'])}")


if __name__ == "__main__":
    main()

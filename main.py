import argparse
import os
import yaml
from datetime import datetime

from src.orchestrator import run_iterations


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative Video Generation")
    parser.add_argument("--goal", type=str, help="High-level goal for the generation")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional name for this run. If omitted, uses timestamp.")
    return parser.parse_args()


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config()
    goal = args.goal or None

    # 1) Figure out a name for this run
    if args.run_name:
        run_name = args.run_name
    else:
        # Use a timestamp if no run_name was provided
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2) Build the run_directory path under your chosen base (runs/)
    runs_dir = config.get("runs_directory", "runs")  # from config.yaml
    run_directory = os.path.join(runs_dir, run_name)
    os.makedirs(run_directory, exist_ok=True)

    # 3) Override logs/frames/videos paths to live inside run_directory
    config["logs"]["directory"] = os.path.join(run_directory, "logs")
    config["frames"]["output_directory"] = os.path.join(run_directory, "frames")
    config["video_output_dir"] = os.path.join(run_directory, "videos")

    # 4) Launch the iteration loop. We’ll pass run_directory to orchestrator
    run_iterations(config, goal, run_directory)


if __name__ == "__main__":
    main()

"""Train complex YOLOv7 for 10 epochs, then evaluate on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.yolov7.build import build_model

if __name__ == "__main__":
    main("train", task="complex", version="v7", build_model=build_model)

import os

ARENA_DIAMETER = 60  # in mm
EXPERIMENT_DURATION = 1800  # experiment duration time must be in seconds
FPS = 24

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(ROOT_DIR, "data", "trackings")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")

NORMALIZATION_DIR = os.path.join(ROOT_DIR, "data", "normalization")

PIPELINE_DIR = os.path.join(ROOT_DIR, "pipeline")
RESULTS_DIR = os.path.join(ROOT_DIR, "data", "results")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

CONFIG_DIR = os.path.join(ROOT_DIR, "configs")

import os


EXPERIMENT_DURATION = 600  # experiment duration time must be in seconds
FPS = 24

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(ROOT_DIR, "data", "trackings")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")

PIPELINE_DIR = os.path.join(ROOT_DIR, "pipeline")
RESULTS_DIR = os.path.join(ROOT_DIR, "data", "results")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

CONFIG_DIR = os.path.join(ROOT_DIR, "configs")





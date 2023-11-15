import os


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(ROOT_DIR, "data", "trackings")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")

NORMALIZATION_DIR = os.path.join(ROOT_DIR, "data", "normalization")

PIPELINE_DIR = os.path.join(ROOT_DIR, "src")
RESULTS_DIR = os.path.join(ROOT_DIR, "data", "results")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

CONFIG_DIR = os.path.join(ROOT_DIR, "configs")

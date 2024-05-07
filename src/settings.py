import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = r'/srv/milky/drosophila-datasets/fragileX'
INPUT_DIR = os.path.join(DATA_DIR, "data", "trackings")
OUTPUT_DIR = os.path.join(DATA_DIR, "data", "processed")
NORMALIZATION_DIR = os.path.join(DATA_DIR, "data", "normalization")
RESULTS_DIR = os.path.join(DATA_DIR, "data", "results")

PIPELINE_DIR = os.path.join(ROOT_DIR, "src")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
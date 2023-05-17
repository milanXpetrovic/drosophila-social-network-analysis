import os
import toml

# TREATMENT = os.environ["TREATMENT"]

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

# INPUT_DIR = os.path.join(ROOT_DIR, "data", "trackings", TREATMENT)
# OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed", TREATMENT)

PIPELINE_DIR = os.path.join(ROOT_DIR, "pipeline")
RESULTS_DIR = os.path.join(ROOT_DIR, "data", "results")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

CONFIG_DIR = os.path.join(ROOT_DIR, "configs")





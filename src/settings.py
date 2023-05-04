import os

""" 'fx', 'wt', 'dBB15', dBB14 """
TREATMENT = "dBB14"

EXPERIMENT_DURATION = 600  # experiment duration time must be in seconds
FPS = 24
DATAFRAME_LEN = EXPERIMENT_DURATION * FPS

# CSm measures used from paper Automatic identification, Schneider and Levine
ANGLE = [130, 175]
DISTANCE = 2.00
TIME = [0.4, 1.0]

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(ROOT_DIR, "data", "trackings", TREATMENT)
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed", TREATMENT)

PIPELINE_DIR = os.path.join(ROOT_DIR, "pipeline")
RESULTS_DIR = os.path.join(ROOT_DIR, "data", "results")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

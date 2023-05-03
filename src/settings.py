import os

TREATMENT = "wt"

EXPERIMENT_DURATION = 600  # experiment duration time must be in seconds
FPS = 24
DATAFRAME_LEN = EXPERIMENT_DURATION * FPS

DISTANCE_BETWEEN_FLIES = 18

TOUCH_DURATION_SEC = 0.6
TOUCH_DURATION_FRAMES = int(TOUCH_DURATION_SEC * FPS)


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(ROOT_DIR, "data", "trackings", TREATMENT)
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed", TREATMENT)


LOGS_DIR = os.path.join(ROOT_DIR, "logs")

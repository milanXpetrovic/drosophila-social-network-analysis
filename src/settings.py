import os

TREATMENT = "wt"

EXPERIMENT_DURATION = 600  # experiment duration time must be in seconds
FPS = 24
DATAFRAME_LEN = EXPERIMENT_DURATION * FPS

# distance (body lengths) 2.50 (2.00–3.25)
# angle 155 (130–175)
# time 0.6 (0.4–1.0)

ANGLE = [130, 175]
DISTANCE = [2.00, 3.25]
TIME = [0.4, 1.0]

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(ROOT_DIR, "data", "trackings", TREATMENT)
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed", TREATMENT)
RESULTS_DIR = os.path.join(ROOT_DIR, "data", "results")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

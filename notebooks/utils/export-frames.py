import os
from concurrent.futures import ProcessPoolExecutor

import cv2


def save_frame(frame, frame_filename):
    resized_frame = cv2.resize(frame, None, fx=0.2, fy=0.2)
    cv2.imwrite(frame_filename, resized_frame)


def extract_frames(video_path, output_folder, frame_rate):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_frame_rate // frame_rate)
    frame_number, saved_frame_number = 0, 0

    with ProcessPoolExecutor(max_workers=12) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"{saved_frame_number+1}.png")
                executor.submit(save_frame, frame, frame_filename)
                saved_frame_number += 1

            frame_number += 1

    cap.release()
    print(f"Extracted {saved_frame_number} frames from the video.")


video_path = "./data/Cs_5DIZ/recordings/Cs_5DIZ_01_08_2023_10_10_A1.mp4"
output_folder = "./data/Cs_5DIZ/frames/Cs_5DIZ_01_08_2023_10_10_A1"
frame_rate = 24
extract_frames(video_path, output_folder, frame_rate)


import os
import re
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from src import settings
from src.utils import fileio

TREATMENT = "Cs_5DIZ"
INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "0_0_preproc_data", TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

GROUP_NAME = "Cs_5DIZ_01_08_2023_10_10_A1"
group = fileio.load_files_from_folder(treatment[GROUP_NAME], file_format=".csv")

frames_folder = "./data/Cs_5DIZ/frames/Cs_5DIZ_01_08_2023_10_10_A1"

for row in range(1000):
    image_filename = os.path.join(frames_folder, f"{row+1}.png")
    img = mpimg.imread(image_filename)
    fig, ax = plt.subplots()
    for fly_name, fly_path in group.items():
        # TODO read before and save to dict
        df = pd.read_csv(fly_path, usecols=["pos x", "pos y", "major axis len"])

        first_row = df.iloc[row].values
        img_width = img.shape[1]
        img_height = img.shape[0]
        scaled_x = first_row[0]
        scaled_y = first_row[1]
        body_len = first_row[2]
        min_x = 553.023338607595
        min_y = 167.17559769167354
        x_px_ratio = 31.839003077183513
        y_px_ratio = 32.18860843823452

        image_scaling_factor = 0.2
        x_original = ((scaled_x * x_px_ratio) + min_x) * image_scaling_factor
        y_original = ((scaled_y * y_px_ratio) + min_y) * image_scaling_factor

        bd_radius = ((body_len * ((x_px_ratio + y_px_ratio) / 2))) * image_scaling_factor
        fly_label = fly_name.replace("fly", "")
        fly_label = fly_label.replace(".csv", "")
        ax.scatter(x_original, y_original)
        # if fly_name in ["fly1.csv", "fly9.csv", "fly11.csv"]:
        # circle = plt.Circle((x_original, y_original), bd_radius * 2.5, fill=False)
        # ax.add_artist(circle)

        ax.text(x_original, y_original, fly_label, color="white", fontsize=12, fontweight="bold")

    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(pad=0)
    plt.grid(False)

    plt.savefig(f"./data/Cs_5DIZ/gif/{row+1}.png")

    # plt.show()


import os
import re

import cv2


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split("(\d+)", s)]


path = "/home/milky/drosophila-isolation/notebooks/sn_lab_visit/data/Cs_5DIZ/gif"
output_video_path = "./output_video.mp4"
frame_rate = 24

images = [img for img in sorted(os.listdir(path)) if img.endswith(".png") or img.endswith(".jpg")]
images.sort(key=natural_sort_key)

first_image_path = os.path.join(path, images[0])
first_image = cv2.imread(first_image_path)
height, width, layers = first_image.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

for image in images:
    image_path = os.path.join(path, image)
    img = cv2.imread(image_path)
    video.write(img)

video.release()
print(f"Video saved as {output_video_path}")
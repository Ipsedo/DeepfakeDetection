import argparse
import typing

from os.path import join, isfile, splitext, exists, pathsep
from os import listdir, mkdir

import numpy as np
import cv2

import json

from tqdm import tqdm


def load_sub_dataset(data_path: str, width: int, height: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    metadata_file = open(join(data_path, "metadata.json"))
    metadata_json = json.load(metadata_file)

    mp4_files = [f for f in listdir(data_path) if isfile(join(data_path, f))
                 and splitext(join(data_path, f))[1] == ".mp4"]

    nb_load = 1
    nb_frame_per_video = 300
    datas = np.zeros((nb_load * nb_frame_per_video, height, width, 3), dtype=np.uint8)
    labels = np.zeros((nb_load * nb_frame_per_video,), dtype=np.uint8)

    i = 0

    for mp4 in tqdm(mp4_files[:nb_load]):
        lbl = 1 if metadata_json[mp4]["label"] == "FAKE" else 0

        cap = cv2.VideoCapture(join(data_path, mp4))
        if not cap.isOpened():
            print("Error during reading video {}".format(join(data_path, mp4)))

        success, frame = cap.read()

        if frame.shape[0] > frame.shape[1]:
            frame = frame.transpose(1, 0, 2)

        frame = cv2.resize(frame, (width, height))

        datas[i, :, :, :] = frame
        labels[i] = lbl

        i += 1

        while success:
            success, frame = cap.read()

            if not success:
                break

            if frame.shape[0] > frame.shape[1]:
                frame = frame.transpose(1, 0, 2)

            frame = cv2.resize(frame, (width, height))

            datas[i, :, :, :] = frame
            labels[i] = lbl

            i += 1

    return datas, labels


def save_ndarrays(output_dir: str, prefix: str, frames: np.ndarray, labels: np.ndarray) -> None:
    if not exists(output_dir):
        mkdir(output_dir)

    frames_file_name = join(output_dir, "{}_frames.npy".format(prefix))
    labels_file_name = join(output_dir, "{}_labels.npy".format(prefix))

    if exists(frames_file_name):
        raise FileExistsError("{} already exists !".format(frames_file_name))
    if exists(labels_file_name):
        raise FileExistsError("{} already exists !".format(labels_file_name))

    np.save(splitext(frames_file_name)[0], frames)
    np.save(splitext(labels_file_name)[0], labels)


def main():
    parser = argparse.ArgumentParser("Load data main")
    parser.add_argument("-v", "--video-dir", dest="video_dir", required=True, type=str)
    parser.add_argument("-o", "--output-dir", dest="output_dir", required=True, type=str)

    args = parser.parse_args()

    video_dir = args.video_dir
    output_dir = args.output_dir

    frames, labels = load_sub_dataset(video_dir, 1920, 1080)

    print(frames.shape, labels.shape)

    save_ndarrays(output_dir, video_dir.split("/")[-1] + "_1920-1080_1", frames, labels)


if __name__ == "__main__":
    main()

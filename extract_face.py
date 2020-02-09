from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import cv2

import argparse

from tqdm import tqdm

import json

from os.path import join, isfile, splitext
from os import listdir


def extract_face_pytorch(mtcnn: MTCNN, image: np.ndarray, padding_ratio=0.5) -> np.ndarray:
    image = np.pad(image, (((1920 - 1080) // 2, (1920 - 1080) // 2), (0, 0), (0, 0)),
                   mode='constant', constant_values=255)
    pil_img = Image.fromarray(image)

    bboxes, _ = mtcnn.detect(pil_img)
    if bboxes is None:
        return image

    bbox = bboxes[0]

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    bbox = [int(bbox[0] - width * padding_ratio // 2),
            int(bbox[1] - height * padding_ratio // 2),
            int(bbox[2] + width * padding_ratio // 2),
            int(bbox[3] + height * padding_ratio // 2)]

    return image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]


def main():
    parser = argparse.ArgumentParser("Extract Face Main")
    parser.add_argument("-d", "--data-path", type=str, required=True, dest="data_path")
    parser.add_argument("-o", "--output-path", type=str, required=True, dest="output_path")

    args = parser.parse_args()

    data_path = args.data_path

    metadata_file = open(join(data_path, "metadata.json"))
    metadata_json = json.load(metadata_file)

    mp4_files = [f for f in listdir(data_path) if isfile(join(data_path, f))
                 and splitext(join(data_path, f))[1] == ".mp4"]

    data = np.empty((len(mp4_files),), dtype=np.ndarray)
    labels = np.zeros((len(mp4_files),))

    mtcnn = MTCNN(image_size=1920, margin=0)

    i = 0

    for mp4 in tqdm(mp4_files):
        lbl = 1 if metadata_json[mp4]["label"] == "FAKE" else 0

        cap = cv2.VideoCapture(join(data_path, mp4))
        if not cap.isOpened():
            print("Error during reading video {}".format(join(data_path, mp4)))

        success, frame = cap.read()

        if frame.shape[0] > frame.shape[1]:
            frame = frame.transpose(1, 0, 2)

        data[i] = extract_face_pytorch(mtcnn, frame)
        labels[i] = lbl

        i += 1

        """while success:
            success, frame = cap.read()

            if not success:
                break

            if frame.shape[0] > frame.shape[1]:
                frame = frame.transpose(1, 0, 2)

            data[i] = extract_face_pytorch(mtcnn, frame)
            labels[i] = lbl

            i += 1"""

    np.save(join(args.output_path, data_path.split("/")[-1] + "_faces_img"), data)
    np.save(join(args.output_path, data_path.split("/")[-1] + "_faces_lbl"), labels)


if __name__ == "__main__":
    main()

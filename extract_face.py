from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import cv2

import argparse

from tqdm import tqdm

import json

from os.path import join, isfile, splitext, isdir
from os import listdir

from typing import Tuple, Optional, List

import matplotlib.pyplot as plt


def extract_face_pytorch(mtcnn: MTCNN, image: np.ndarray,
                         size=-1, padding_ratio=0.5) -> Optional[List[np.ndarray]]:
    """img_width = image.shape[1]
    img_height = image.shape[0]

    padding_width_img = img_height - img_width \
        if img_height > img_width else 0
    padding_width_img += 1920 - (img_width + padding_width_img) \
        if img_width + padding_width_img < 1920 else 0
    padding_width_img //= 2

    padding_height_img = img_width - img_height \
        if img_width > img_height else 0
    padding_height_img += 1920 - (img_height + padding_height_img) \
        if img_height + padding_height_img < 1920 else 0
    padding_height_img //= 2

    image = np.pad(image, ((padding_height_img, padding_height_img),
                           (padding_width_img, padding_width_img),
                           (0, 0)),
                   mode='constant', constant_values=0)"""

    pil_img = Image.fromarray(image)

    bboxes, a = mtcnn.detect(pil_img)
    if bboxes is None:
        return None

    res = []

    for bbox in bboxes:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        bbox = [int(bbox[0] - width * padding_ratio // 2),
                int(bbox[1] - height * padding_ratio // 2),
                int(bbox[2] + width * padding_ratio // 2),
                int(bbox[3] + height * padding_ratio // 2)]

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        padding_width_face = (height - width if height > width else 0) // 2
        padding_height_face = (width - height if width > height else 0) // 2

        bbox = [bbox[0] - padding_width_face,
                bbox[1] - padding_height_face,
                bbox[2] + padding_width_face,
                bbox[3] + padding_height_face]

        plt.imshow(cv2.rectangle(image.copy(), (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=[255, 0, 0], thickness=3))
        plt.show()

        face = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] if size == -1 else \
            cv2.resize(image[bbox[1]:bbox[3], bbox[0]:bbox[2], :], (size, size))

        res.append(face)
    return res


def process_all_folders(root_path: str) -> Tuple[np.ndarray, np.ndarray]:
    dirs = [dir for dir in listdir(root_path) if isdir(join(root_path, dir))
            and dir.startswith("dfdc_train_part_")]

    data = np.empty(0, dtype=object)


# labels :
# 0 -> REAL, one face
# 1 -> FAKE, one face
# 2 -> REAL, more than one face
# 3 -> FAKE, more than one face
def main():
    parser = argparse.ArgumentParser("Extract Face Main")
    parser.add_argument("-d", "--data-path", type=str, required=True, dest="data_path")
    parser.add_argument("-o", "--output-path", type=str, required=True, dest="output_path")
    parser.add_argument("-s", "--size", type=int, required=True)

    args = parser.parse_args()

    data_path = args.data_path

    metadata_file = open(join(data_path, "metadata.json"))
    metadata_json = json.load(metadata_file)

    mp4_files = [f for f in listdir(data_path) if isfile(join(data_path, f))
                 and splitext(join(data_path, f))[1] == ".mp4"]

    data = np.empty((3, args.size, args.size, 3), dtype=np.uint8)
    #data = np.empty((len(mp4_files), args.size, args.size, 3), dtype=np.uint8)
    labels = np.zeros((len(mp4_files),))

    mtcnn = MTCNN()
    i = 0

    for mp4 in tqdm(mp4_files[10:11]):
        lbl = 1 if metadata_json[mp4]["label"] == "FAKE" else 0

        cap = cv2.VideoCapture(join(data_path, mp4))
        if not cap.isOpened():
            print("Error during reading video {}".format(join(data_path, mp4)))

        success, frame = cap.read()

        faces = extract_face_pytorch(mtcnn, frame, args.size)
        while faces is None:
            faces = extract_face_pytorch(mtcnn, frame, args.size)

        lbl = 2 if len(faces) > 1 and lbl == 0 else 3 if len(faces) > 1 and lbl == 1 else lbl
        for f in faces:

            data[i] = f
            labels[i] = lbl

            i += 1

        """while success:
            success, frame = cap.read()

            if not success:
                break

            data[i] = extract_face_pytorch(mtcnn, frame)
            labels[i] = lbl

            i += 1"""
    plt.imshow(data[0])
    plt.show()
    np.save(join(args.output_path, data_path.split("/")[-1] + "_1_faces_img"), data)
    np.save(join(args.output_path, data_path.split("/")[-1] + "_1_faces_lbl"), labels)


if __name__ == "__main__":
    main()

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


# https://github.com/informramiz/Face-Detection-OpenCV
def extract_face_opencv(f_cascade: cv2.CascadeClassifier,
                        image: np.ndarray, scale_factor=1.1, size=-1, padding_ratio=0.5) \
        -> List[np.ndarray]:
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images
    bboxes = f_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=5)

    res = []

    for (x, y, w, h) in bboxes:
        bbox = [x, y, x + w, y + h]

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

        left_shift = abs(bbox[0]) if bbox[0] < 0 else 0
        right_shift = bbox[2] - image.shape[1] if bbox[2] > image.shape[1] else 0
        top_shift = abs(bbox[1]) if bbox[1] < 0 else 0
        bottom_shift = bbox[3] - image.shape[0] if bbox[3] > image.shape[0] else 0

        bbox[0] += left_shift
        bbox[0] -= right_shift
        bbox[1] += top_shift
        bbox[1] -= bottom_shift

        bbox[2] += left_shift
        bbox[2] -= right_shift
        bbox[3] += top_shift
        bbox[3] -= bottom_shift

        face = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] if size == -1 else \
        cv2.resize(image[bbox[1]:bbox[3], bbox[0]:bbox[2], :], (size, size))

        res.append(face)
    return res


def process_all_folders(root_path: str, size: int) -> Tuple[np.ndarray, np.ndarray]:
    dirs = [d for d in listdir(root_path) if isdir(join(root_path, d))
            and d.startswith("dfdc_train_part_")]

    videos = [v for d in dirs for v in listdir(join(root_path, d)) if splitext(join(root_path, d, v))[1] == ".mp4"]

    data = np.empty((len(videos), 3, size, size), dtype=np.uint8)
    labels = np.empty((len(videos,)), dtype=np.uint8)

    index = 0

    video_to_process = len(videos)
    folder_to_process = len(dirs)

    f_cascade = cv2.CascadeClassifier(
        '/home/samuel/Documents/Kaggle/Face-Detection-OpenCV-master/data/haarcascade_frontalface_alt.xml')

    for d in dirs:
        metadata_file = open(join(root_path, d, "metadata.json"))
        metadata_json = json.load(metadata_file)

        folder_vid = [v for v in listdir(join(root_path, d)) if splitext(v)[1] == ".mp4"]

        for mp4 in tqdm(folder_vid):
            lbl = 1 if metadata_json[mp4]["label"] == "FAKE" else 0

            cap = cv2.VideoCapture(join(root_path, d, mp4))
            if not cap.isOpened():
                print("Error during reading video {}".format(join(root_path, d, mp4)))

            success, frame = cap.read()
            for _ in range(10):
                success, frame = cap.read()

            faces = extract_face_opencv(f_cascade, frame, size=size)

            for f in faces:
                if index < data.shape[0]:
                    data[index] = f.transpose((2, 0, 1))
                    labels[index] = lbl

                    index += 1
                else:
                    print("Limite atteinte ! {}, idx = {}, restante = {}".format(data.shape, index, video_to_process))
                    return data, labels
            video_to_process -= 1
        folder_to_process -= 1
        print("Il reste {} dossier à traiter !".format(folder_to_process))

    return data, labels


def old_main():
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

    #mtcnn = MTCNN()
    f_cascade = cv2.CascadeClassifier('/home/samuel/Documents/Kaggle/Face-Detection-OpenCV-master/data/haarcascade_frontalface_alt.xml')
    i = 0

    for mp4 in tqdm(mp4_files[10:11]):
        lbl = 1 if metadata_json[mp4]["label"] == "FAKE" else 0

        cap = cv2.VideoCapture(join(data_path, mp4))
        if not cap.isOpened():
            print("Error during reading video {}".format(join(data_path, mp4)))

        success, frame = cap.read()
        for _ in range(10):
            success, frame = cap.read()

        faces = extract_face_opencv(f_cascade, frame, size=args.size)

        for f in faces:

            data[i] = f
            labels[i] = lbl

            i += 1

    np.save(join(args.output_path, data_path.split("/")[-1] + "_1_faces_img"), data)
    np.save(join(args.output_path, data_path.split("/")[-1] + "_1_faces_lbl"), labels)


def main():
    parser = argparse.ArgumentParser("Extract face main")

    parser.add_argument("-r", "--root-path", type=str, required=True, dest="root_path")
    parser.add_argument("-s", "--size", type=int, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True, dest="output_path")

    args = parser.parse_args()

    data, labels = process_all_folders(args.root_path, args.size)
    np.save(join(args.output_path, " {}_faces_img".format(data.shape[0])), data)
    np.save(join(args.output_path, "{}_faces_lbl".format(labels.shape[0])), labels)


if __name__ == "__main__":
    main()

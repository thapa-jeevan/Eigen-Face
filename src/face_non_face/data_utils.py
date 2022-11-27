import glob
import random

import cv2
import numpy as np

from src.settings import CATEGORY_2_IDX


def divide_face_imgs():
    subjects = list(range(1, 41))
    random.shuffle(subjects)
    train_subjects = subjects[:35]

    train_img_path_list = []

    for sub_idx in train_subjects:
        img_path_list = glob.glob(f"data/atat_data/s{sub_idx}/*")
        random.shuffle(img_path_list)
        train_img_path_list += img_path_list[:8]
    test_img_path_list = list(set(glob.glob("data/atat_data/*/*")) - set(train_img_path_list))
    return train_img_path_list, test_img_path_list


def divide_non_face_imgs():
    img_path_list = glob.glob(f"data/non-faces/*")
    random.shuffle(img_path_list)
    train_img_path_list = img_path_list[:-120]
    test_img_path_list = img_path_list[-120:]
    return train_img_path_list, test_img_path_list


def read_imgs(img_path_list, IMG_HEIGHT, IMG_WIDTH, ):
    img_list = []
    for img_path in img_path_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_list.append(img.ravel())
    return np.vstack(img_list)


def get_face_cls_data():
    train_faces, test_faces = divide_face_imgs()
    train_non_faces, test_non_faces = divide_non_face_imgs()

    train_imgs = train_faces + train_non_faces
    X_train = read_imgs(train_imgs)
    y_train = np.array(
        [CATEGORY_2_IDX["face"]] * len(train_faces) + [CATEGORY_2_IDX["non-face"]] * len(train_non_faces))

    test_imgs = test_faces + test_non_faces
    X_test = read_imgs(test_imgs)
    y_test = np.array(
        [CATEGORY_2_IDX["face"]] * len(test_faces) + [CATEGORY_2_IDX["non-face"]] * len(test_non_faces))

    return X_train, X_test, y_train, y_test

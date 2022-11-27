import cv2
import glob
import numpy as np


def load_images():
    img_path_list = sorted(glob.glob("data/atat_data/*/*"))
    img_list = []
    for img_path in img_path_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_list.append(img.ravel())

    img_array = np.stack(img_list)
    return img_array
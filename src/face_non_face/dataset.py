import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .data_utils import get_face_cls_data
from src.settings import CATEGORY_2_IDX

train_img_per_class = 16000
test_img_per_class = 1000


class FaceClassification(Dataset):
    def __init__(self, img_list, labels, transform=None):
        self.img_list = img_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path) #.convert('L')
        img = self.transform(img)
        return img, self.labels[idx]


def divide_imgs(img_dir):
    img_path_list = glob.glob(os.path.join(img_dir, "*"))
    random.shuffle(img_path_list)
    train_img_path_list = img_path_list[:train_img_per_class]
    test_img_path_list = img_path_list[train_img_per_class: train_img_per_class + test_img_per_class]
    return train_img_path_list, test_img_path_list


def get_pretrain_data():
    train_face_list, test_face_list = divide_imgs("data/CelebA/img_align_celeba/")
    train_non_face_list, test_non_face_list = divide_imgs("data/VOCdevkit/VOC2012/JPEGImages/")

    train_img_list = train_face_list + train_non_face_list
    y_train = np.array(
        [CATEGORY_2_IDX["face"]] * len(train_face_list) + [CATEGORY_2_IDX["non-face"]] * len(train_non_face_list))

    test_img_list = test_face_list + test_non_face_list
    y_test = np.array(
        [CATEGORY_2_IDX["face"]] * len(test_face_list) + [CATEGORY_2_IDX["non-face"]] * len(test_non_face_list))

    return train_img_list, test_img_list, y_train, y_test


def get_dataloaders(train_transform, test_transform, batchsize):
    train_img_list, test_img_list, y_train, y_test = get_pretrain_data()
    valid_imgs, test_imgs, y_valid, y_test = get_face_cls_data(read_img=False)

    train_dataset = FaceClassification(train_img_list, y_train, train_transform)
    valid_dataset = FaceClassification(valid_imgs, y_valid, train_transform)
    test_dataset = FaceClassification(test_imgs, y_test, test_transform)

    pretrain_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, num_workers=0, pin_memory=True, shuffle=True
    )

    finetune_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batchsize, num_workers=0, pin_memory=True, shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, num_workers=0, pin_memory=True, shuffle=False
    )

    return pretrain_dataloader, finetune_dataloader, test_dataloader

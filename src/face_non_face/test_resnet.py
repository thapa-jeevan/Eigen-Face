import os

import torch

from src.face_non_face.dataset import get_dataloaders
from src.models.model_resnet import resnet18
from src.settings import CHECKPOINT_DIR
from .train_resnet import get_model_predicton, test_transform

BATCH_SIZE = 128

if __name__ == '__main__':
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "face_classification.pt")
    model = resnet18(num_classes=2)
    model.cuda()
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

    _, _, test_dataloader = get_dataloaders(
        None, test_transform, BATCH_SIZE)

    test_acc = get_model_predicton(model, test_dataloader)
    print(test_acc)

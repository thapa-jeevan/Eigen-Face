import os

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from src.models.model_resnet import resnet18
from src.settings import CHECKPOINT_DIR
from src.face_non_face.dataset import get_dataloaders
from src.settings import IMG_SHAPE
from src.utils import seed_everything

seed_everything(42)

BATCH_SIZE = 128
PRE_TRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 10

test_transform = transforms.Compose([
    transforms.Resize(IMG_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_transform = transforms.Compose([
    transforms.Resize(IMG_SHAPE),
    transforms.RandomResizedCrop(IMG_SHAPE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-15, 15)),
    transforms.RandomAutocontrast(),
    transforms.RandomAdjustSharpness(2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def train_one_epoch(model, optimizer, criterion, train_dataloader, epoch):
    train_loss = 0
    model.train()
    for i, (inp_tensor, y_true) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(inp_tensor.cuda())

        loss = criterion(outputs, y_true.cuda())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        acc = (y_pred == y_true.cpu().numpy()).sum() / len(y_true)

        print(
            f"\rEpoch: {epoch} Iteration: {i + 1}/{len(train_dataloader)} Training Loss: {loss.item():.5f} Accuracy: {acc:2.3f}",
            end=" ")

    print()
    return train_loss


def get_model_predicton(model, valid_dataloader):
    model.eval()
    print()

    y_trues = []
    y_preds = []

    for i, (inp_tensor, y_true) in enumerate(valid_dataloader):
        outputs = model(inp_tensor.cuda())
        y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)

        y_trues.append(y_true.numpy())
        y_preds.append(y_pred)

    y_trues = np.hstack(y_trues)
    y_preds = np.hstack(y_preds)
    test_acc = (y_preds == y_trues).sum() / len(y_trues)
    return test_acc


if __name__ == '__main__':
    pretrain_dataloader, finetune_dataloader, test_dataloader = get_dataloaders(
        train_transform, test_transform, BATCH_SIZE)

    model = resnet18(num_classes=2)
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "face_classification.pt")

    for epoch in tqdm(range(PRE_TRAIN_EPOCHS)):
        train_one_epoch(model, optimizer, criterion, pretrain_dataloader, epoch)
        test_acc = get_model_predicton(model, test_dataloader)
        print(test_acc)

    for epoch in tqdm(range(FINETUNE_EPOCHS)):
        train_one_epoch(model, optimizer, criterion, finetune_dataloader, epoch)
        test_acc = get_model_predicton(model, test_dataloader)
        print(test_acc)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)

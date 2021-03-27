import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from torchsummary import summary

from PIL import Image
import numpy as np

import os

from utils import UNET, dice_coef_loss, mse_dice_loss


class MRIData(Dataset):
    def __init__(self, path, transform_img=None, transform_mask=None, device="cuda"):
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.device = device
        self.data = []
        for folder in os.listdir(path):
            folder = path + folder
            for imgs in os.listdir(folder):
                imgs = folder + "/" + imgs
                if imgs[-8:-4] != "mask":
                    img = np.array(Image.open(imgs))
                    mask_path = imgs[:-4] + "_mask" + imgs[-4:]
                    mask = np.array(Image.open(mask_path).convert("L"))
                    self.data.append((img, mask))

    def __getitem__(self, idx):
        img, mask = self.data[idx]
        img = torch.tensor(img.reshape(3, 256, 256) / 255, dtype=torch.float)
        mask = torch.tensor(mask.reshape(1, 256, 256), dtype=torch.float)

        return (img.to(self.device), mask.to(self.device))

    def __len__(self):
        return len(self.data)


path = r"/tmp"
device = "cuda"
dataset = MRIData(path, device)

trainset = dataset

train_dl = DataLoader(trainset, 8, True)

model = UNET(3, 1).to(device)


def fit(model, train_dl, epochs, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for epoch in range(epochs):
        train_loss, val_loss = [], []
        model.train()
        for batch in train_dl:
            img, label = batch
            out = model(img)
            loss = bce_dice_loss(out, label)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(
            f"Epoch: {epoch+1}/{epochs} -- Train Loss: {sum(train_loss)/len(train_loss)}"
        )
        fig = plt.figure(figsize=(2, 1))
        fig.add_subplot(1, 2, 1)
        plt.imshow(np.array(label[0].cpu().detach()).reshape(256, 256), cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(np.array(out[0].cpu().detach()).reshape(256, 256), cmap="gray")
        plt.show()
        torch.save(model, r"tmp/BrainModel")


fit(model, train_dl, 20)


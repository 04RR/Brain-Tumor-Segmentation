import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import cv2
import torch
import io
import torch.nn as nn
from torchvision.models import resnext50_32x4d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as t
import streamlit as st


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()

        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.convrelu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)

        self.deconv = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )

        self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)

        return x


class ResNeXtUNet(nn.Module):
    def __init__(self, n_classes=1):
        super(ResNeXtUNet, self).__init__()

        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]
        
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.last_conv0 = ConvRelu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)

    def forward(self, x):
        
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.last_conv0(d1)
        out = self.last_conv1(out)
        out = nn.Sigmoid()(out)

        return out


model = ResNeXtUNet(1)
model = torch.load(r"BrainModel")
model = model.to('cuda')

st.title("Get instant Brain MRI Scan Results")
st.markdown("Segmentation of MRI Scans and predict chance of mortality")
img = st.file_uploader("Upload the folder with the MRI Scans")

if img is not None:
    img = bytearray(img.read())
    imgs = Image.open(io.BytesIO(img)).convert("RGB")
   
    img1 = Image.open(io.BytesIO(img)).convert("L")
    img = t.Resize((256, 256))(imgs)

    img = t.ToTensor()(img)
    img = torch.reshape(img, (1, 3, 256, 256))

    pred = model(img.to('cuda'))
    pred = pred.detach().cpu().numpy()[0, 0, :, :]

    pred_t = np.copy(pred)
    pred_t[np.nonzero(pred_t < 0.3)] = 0.0
    pred_t[np.nonzero(pred_t >= 0.3)] = 255.0  # 1.0

    im = Image.fromarray(pred_t).convert("L")

    padding = np.ones((256, 50)) * 255

    stackimgs = Image.fromarray(
        np.hstack([np.array(img1), padding, np.array(im)])
    ).convert("L")

    st.image(stackimgs, caption="MRI Scan & Predicted Mask")
    st.markdown(
        """A brain MRI is one of the most commonly performed techniques of medical imaging. It enables clinicians to focus on various parts of the brain and examine their anatomy and pathology, using different MRI sequences, such as T1w, T2w, or FLAIR.
    MRI is used to analyze the anatomy of the brain and to identify some pathological conditions such as cerebrovascular incidents, demyelinating and neurodegenerative diseases. Moreover, the MRI can be used for examining the activity of the brain under specific activities (functional MRI - fMRI). The biggest advantage of MRI is that it uses no radiation. However, it takes longer to be produced than CT for example, which is why it’s not a primary imaging choice for urgent conditions."""
    )

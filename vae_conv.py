import torch
import numpy as np
from torch import nn
from torch.nn.functional import binary_cross_entropy as bce
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import time

class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, final_relu=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x + self.block(x)
        x = self.block2(x)
        return x

class EncoderConv(nn.Module):
    
    def __init__(self, in_shape=(3,32,32), latent_dim=50):
        super().__init__()
        self.in_shape = in_shape
        self.in_channel, self.h, self.w = in_shape
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            ResBlock(self.in_channel, self.in_channel*4, final_relu=True), # 32x32
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(self.in_channel*4, self.in_channel*16, final_relu=True), # 16x16
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(self.in_channel*16, self.in_channel*64, final_relu=True), # 8x8
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(self.in_channel*64, self.in_channel*256, final_relu=True), # 4x4
            nn.AvgPool2d(kernel_size=4),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channel*256, latent_dim*2),
        )
    
    def forward(self, x) -> torch.Tensor:
        x = self.net(x)
        x = self.mlp(x.view(-1, self.in_channel*256))
        return x

class DecoderConv(nn.Module):
    
    def __init__(self, latent_dim=50, out_shape=(3,32,32)):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_shape = out_shape
        self.out_channel, self.h, self.w = out_shape
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, self.out_channel*256),
            nn.BatchNorm1d(self.out_channel*256),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.out_channel*256, self.out_channel*64, kernel_size=4, stride=4), # 4x4
            nn.Conv2d(self.out_channel*64, self.out_channel*64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(self.out_channel*64),
            nn.ReLU(),
            nn.ConvTranspose2d(self.out_channel*64, self.out_channel*4, kernel_size=4, stride=4), # 16x16
            nn.Conv2d(self.out_channel*4, self.out_channel*4, kernel_size=3, padding=1), 
            nn.BatchNorm2d(self.out_channel*4),
            nn.ConvTranspose2d(self.out_channel*4, self.out_channel, kernel_size=2, stride=2), # 32x32
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding=1), 
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding=1), 
            nn.Sigmoid(),
        )
    
    def forward(self, x) -> torch.Tensor:
        x = self.mlp(x)
        x = x.view(-1, self.out_channel*256, 1, 1)
        x = self.deconv(x)
        return x
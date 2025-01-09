import torch
import numpy as np
from torch import nn
from torch.nn.functional import binary_cross_entropy as bce
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt

device = 'xpu'

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
        return self.block2(x + self.block(x))

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
        return self.mlp(self.net(x).reshape(-1, self.in_channel*256))

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
        return self.deconv(self.mlp(x).reshape(-1, self.out_channel*256, 1, 1))

class Encoder(nn.Module):
    
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim*4),
            nn.ReLU(),
            nn.Linear(in_dim*4, latent_dim*2),
        )
    
    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)

class Decoder(nn.Module):
    
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*4),
            nn.ReLU(),
            nn.Linear(latent_dim*4, out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)

def vae_loss(mu, sigma, x, y):
    """
    mu: (b, latent_dim)
    sigma: (b, latent_dim)
    x: (b, in_dim)
    y: (b, l, out_dim)
    """
    b, l, latent_dim = y.shape
    recons_loss = torch.sum(bce(y, x.view(b, 1, -1).expand(-1, l, -1), reduction='none'), dim=(-1, -2))/l
    kl_divergence = -torch.sum((1 + torch.log(sigma**2) - mu**2 - sigma**2)/2, dim=-1)
    loss = torch.mean(recons_loss + kl_divergence)
    return loss

def train_vae(encoder, decoder, dataloader, lr, samples):
    optimizer_enc = torch.optim.AdamW(encoder.parameters(), lr=lr)
    optimizer_dec = torch.optim.AdamW(decoder.parameters(), lr=lr)
    encoder.train()
    decoder.train()
    if device == 'xpu':
        encoder, optimizer_enc = ipex.optimize(encoder, optimizer=optimizer_enc)
        decoder, optimizer_dec = ipex.optimize(decoder, optimizer=optimizer_dec)
    size = len(dataloader.dataset)
    latent_dim = encoder.latent_dim
    for batch, (X, _) in enumerate(dataloader):
        bs = X.shape[0]
        mu, sigma = encoder(X.view(bs, -1)).split(latent_dim, dim=1)
        epsilon = torch.randn(bs, samples, latent_dim, device=device, requires_grad=False)
        z = mu.view(bs, 1, -1) + epsilon * sigma.view(bs, 1, -1)
        y = decoder(z.view(-1, latent_dim)).view(bs, samples, -1)

        loss = vae_loss(mu, sigma, X.view(bs, -1), y)
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        loss.backward()
        optimizer_enc.step()
        optimizer_dec.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * bs + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def save_models(encoder, decoder):
    torch.save(encoder, f'encoder-{time.asctime()}.pth')
    torch.save(decoder, f'decoder-{time.asctime()}.pth')

to_tensor = ToTensor()

def my_transform(x):
    return to_tensor(x).to(device)

if __name__ == '__main__':
    if device == 'xpu':
        import intel_extension_for_pytorch as ipex
    # encoder = EncoderConv(latent_dim=256).to(device)
    # decoder = DecoderConv(latent_dim=256).to(device)
    encoder = Encoder(32*32*3, 256).to(device)
    decoder = Decoder(256, 32*32*3).to(device)
    training_data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=my_transform
    )
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    epochs = 5
    for i in range(epochs):
        print(f'epoch {i}')
        train_vae(encoder, decoder, train_dataloader, 0.001, 16)
    save_models(encoder, decoder)


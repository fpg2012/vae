import torch
import numpy as np
from torch import nn
from torch.nn.functional import binary_cross_entropy as bce
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import time

device = 'xpu'

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

def train_vae(encoder: Encoder, decoder: Decoder, dataloader, lr, samples):
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
        mu, sigma = encoder(X).split(latent_dim, dim=1)
        bs = X.shape[0]
        epsilon = torch.randn(bs, samples, latent_dim, device=device)
        z = mu.view(bs, 1, -1) + epsilon * sigma.view(bs, 1, -1)
        y = decoder(z.view(-1, latent_dim)).view(bs, samples, -1)

        loss = vae_loss(mu, sigma, X, y)
        loss.backward()
        optimizer_enc.step()
        optimizer_dec.step()
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * bs + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def save_models(encoder: Encoder, decoder: Decoder):
    torch.save(encoder, f'encoder-{time.asctime()}.pth')
    torch.save(decoder, f'decoder-{time.asctime()}.pth')

to_tensor = ToTensor()

def my_transform(x):
    return to_tensor(x).flatten().to(device)

if __name__ == '__main__':
    if device == 'xpu':
        import intel_extension_for_pytorch as ipex
    encoder = Encoder(784, 50).to(device)
    decoder = Decoder(50, 784).to(device)
    training_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=my_transform
    )
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    epochs = 3
    for i in range(epochs):
        print(f'epoch {i}')
        train_vae(encoder, decoder, train_dataloader, 0.001, 16)
    save_models(encoder, decoder)


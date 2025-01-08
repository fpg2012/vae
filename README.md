# A Simple PyTorch VAE Implementation
A simple PyTorch VAE implementation in ~100 lines.

```python
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
```

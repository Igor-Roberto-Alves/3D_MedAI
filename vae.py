"""
vae.py
======
Um VAE condicional (CVAE) simples: recebe a **foto** (fatia de MRI) + a **altura**
da fatia, e comprime tudo num **vetor latente** `z`.

    x [B,1,88,104] ---.
                       encoder --> mu, logvar --> z [B, latent]
    altura [B] --emb--'                             |
                                                    v
    altura [B] --emb--> decoder --> x_hat [B,1,88,104]

Por que condicionar na altura?
    Uma fatia do topo da cabeça e uma da base são MUITO diferentes. Dando a altura
    de graça pro modelo, o `z` não precisa gastar capacidade codificando "que altura
    é essa" e pode focar no que varia *entre pacientes* naquela altura.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    """VAE condicionado na altura da fatia.

    Parâmetros
    ----------
    latent : int      tamanho do vetor comprimido z
    cond_dim : int    tamanho do embedding da altura
    n_alturas : int   quantas alturas existem (176 no plano axial do OASIS)
    base : int        largura da primeira camada conv
    in_hw : (int,int) tamanho da imagem de entrada; deve ser divisível por 8
    """

    def __init__(self, latent=128, cond_dim=32, n_alturas=176, base=32, in_hw=(88, 104)):
        super().__init__()
        H, W = in_hw
        assert H % 8 == 0 and W % 8 == 0, "in_hw deve ser divisível por 8"
        self.in_hw = in_hw
        self.latent = latent
        self.fh, self.fw = H // 8, W // 8          # 88/8=11, 104/8=13
        flat = base * 4 * self.fh * self.fw

        # a altura (inteiro 0..175) vira um vetor aprendido
        self.emb = nn.Embedding(n_alturas, cond_dim)

        # encoder: 3 convs stride 2 -> H/8 x W/8
        self.enc = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(flat + cond_dim, latent)
        self.fc_lv = nn.Linear(flat + cond_dim, latent)

        # decoder: espelho do encoder
        self.fc_dec = nn.Linear(latent + cond_dim, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, 1, 4, 2, 1), nn.Sigmoid(),   # imagens estão em [0,1]
        )
        self.base = base

    def encode(self, x, altura):
        c = self.emb(altura)                       # [B, cond_dim]
        h = self.enc(x).flatten(1)                 # [B, flat]
        h = torch.cat([h, c], dim=1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma*eps  -> permite backprop através da amostragem
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, altura):
        c = self.emb(altura)
        h = self.fc_dec(torch.cat([z, c], dim=1))
        h = h.view(-1, self.base * 4, self.fh, self.fw)
        return self.dec(h)

    def forward(self, x, altura):
        mu, logvar = self.encode(x, altura)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, altura), mu, logvar


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    """Reconstrução (MSE somada nos pixels) + beta * KL. Média no batch."""
    rec = F.mse_loss(x_hat, x, reduction="none").flatten(1).sum(1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()
    return rec + beta * kl, rec, kl

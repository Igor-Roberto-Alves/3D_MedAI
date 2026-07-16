"""
complete_diffusion.py
=====================
Difusão **no espaço latente do VAE**: recebe o vetor `z` da **fatia central** e
gera os vetores `z` de todas as outras alturas, completando um bloco de MRI.

    z_central [128]  --.
                        DDPM 1D  -->  z_bloco [L, 128]  --VAE.decode-->  L fatias
    altura_central --'

Como funciona
-------------
1. O VAE (congelado) vira cada fatia num vetor `z` de 128 dims. Um bloco de L
   fatias vizinhas vira então uma **sequência** [L, 128] — que é só um sinal 1D
   de 128 canais e comprimento L. Difusão em cima disso é barata (nada de conv3D).
2. O modelo é um U-Net **1D** que roda ao longo do eixo da altura. Ele aprende a
   remover ruído da sequência inteira de uma vez, então as fatias saem coerentes
   entre si (é isso que dá o "bloco", e não L fatias independentes).
3. O condicionamento na fatia central é feito de 3 jeitos, de propósito:
     - o `z` central é concatenado como canais extras em TODA posição;
     - a altura central entra no embedding de tempo;
     - na amostragem, a posição central é **sobrescrita** com o `z` conhecido a
       cada passo (replacement/inpainting) — garante que o bloco realmente passe
       pela fatia que você deu.

Uso
---
    python complete_diffusion.py --epochs 30      # treina (usa vae.pt)
    python complete_diffusion.py --sample-only    # só gera a figura

Gera `ldm.pt` e `ldm_bloco.png` (bloco real vs bloco gerado).
"""

import os
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vae import CVAE
from train_vae import load_slices, split_by_patient


# ============================================================ 1. latentes do VAE
def load_vae(path="vae.pt", dev="cpu"):
    ck = torch.load(path, map_location=dev, weights_only=False)
    vae = CVAE(latent=ck["latent"], n_alturas=ck["n_alturas"], in_hw=ck["in_hw"])
    vae.load_state_dict(ck["model"])
    vae.eval().to(dev)
    for p in vae.parameters():          # congelado: a difusão não treina o VAE
        p.requires_grad_(False)
    return vae, ck


@torch.no_grad()
def encode_latents(vae, X, A, dev="cpu", bs=256, cache="oasis_latents.pt"):
    """Passa todas as fatias pelo encoder do VAE -> Z [N, latent].

    Usa `mu` (a média), não uma amostra: queremos o vetor determinístico.
    """
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        print(f"cache de latentes carregado: {cache}  ({tuple(d['Z'].shape)})")
        return d["Z"]
    Z = torch.empty(len(X), vae.latent, dtype=torch.float32)
    for i in tqdm(range(0, len(X), bs), desc="encodando fatias"):
        x = X[i:i + bs].to(dev)
        a = A[i:i + bs].to(dev)
        mu, _ = vae.encode(x, a)
        Z[i:i + bs] = mu.cpu()
    torch.save({"Z": Z}, cache)
    print(f"salvo {cache}  {tuple(Z.shape)}")
    return Z


def build_windows(A, P, L=32):
    """Janelas de L alturas CONTÍGUAS do mesmo paciente.

    Retorna idx [M, L] (índices em Z) e pat [M] (paciente de cada janela).
    Janelas deslizantes servem de aumento de dados: 39 pacientes viram milhares
    de blocos. Cada janela tem uma fatia central diferente.
    """
    A_np, P_np = A.numpy(), P.numpy()
    order = np.lexsort((A_np, P_np))          # ordena por paciente, depois altura
    wins, pats = [], []
    start = 0
    for i in range(1, len(order) + 1):
        # fim de um trecho contíguo (mudou de paciente ou pulou uma altura)
        fim = (i == len(order) or P_np[order[i]] != P_np[order[i - 1]]
               or A_np[order[i]] != A_np[order[i - 1]] + 1)
        if fim:
            run = order[start:i]
            for j in range(len(run) - L + 1):
                wins.append(run[j:j + L])
                pats.append(P_np[run[j]])
            start = i
    idx = torch.from_numpy(np.array(wins)).long()
    pat = torch.from_numpy(np.array(pats)).long()
    print(f"janelas de {L} fatias contíguas: {len(idx)}")
    return idx, pat


# ============================================================== 2. o U-Net 1D
def timestep_embedding(t, dim):
    """Embedding sinusoidal do passo de ruído (igual ao DDPM original)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None]          # tempo/altura entram aqui
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attn1D(nn.Module):
    """Self-attention ao longo da altura: deixa fatias distantes se enxergarem."""

    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.att = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)           # [B, L, C]
        h, _ = self.att(h, h, h)
        return x + h.transpose(1, 2)


class LatentUNet1D(nn.Module):
    """Prevê o ruído eps de uma sequência de latentes [B, latent, L].

    Entrada = [z_t ; z_central repetido] -> 2*latent canais.
    """

    def __init__(self, latent=128, L=32, base=192, n_alturas=176, emb_dim=256):
        super().__init__()
        self.latent, self.L = latent, L
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.alt_emb = nn.Embedding(n_alturas, emb_dim)

        self.conv_in = nn.Conv1d(latent * 2, base, 3, padding=1)
        # posição DENTRO do bloco (offset em relação à central) é informação real
        self.pos = nn.Parameter(torch.zeros(1, base, L))

        self.d1 = ResBlock1D(base, base, emb_dim)
        self.down1 = nn.Conv1d(base, base, 4, 2, 1)            # L -> L/2
        self.d2 = ResBlock1D(base, base * 2, emb_dim)
        self.down2 = nn.Conv1d(base * 2, base * 2, 4, 2, 1)    # L/2 -> L/4

        self.mid1 = ResBlock1D(base * 2, base * 2, emb_dim)
        self.mid_att = Attn1D(base * 2)
        self.mid2 = ResBlock1D(base * 2, base * 2, emb_dim)

        self.up2 = nn.ConvTranspose1d(base * 2, base * 2, 4, 2, 1)
        self.u2 = ResBlock1D(base * 4, base, emb_dim)           # +skip d2
        self.up1 = nn.ConvTranspose1d(base, base, 4, 2, 1)
        self.u1 = ResBlock1D(base * 2, base, emb_dim)           # +skip d1

        self.out = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(), nn.Conv1d(base, latent, 3, padding=1))

    def forward(self, z_t, t, z_cent, alt_cent):
        # condição: o z central vale em toda posição -> repete ao longo de L
        cond = z_cent[:, :, None].expand(-1, -1, self.L)
        h = self.conv_in(torch.cat([z_t, cond], dim=1)) + self.pos
        emb = self.t_mlp(timestep_embedding(t, self.emb_dim)) + self.alt_emb(alt_cent)

        h1 = self.d1(h, emb)
        h2 = self.d2(self.down1(h1), emb)
        m = self.down2(h2)
        m = self.mid2(self.mid_att(self.mid1(m, emb)), emb)
        u = self.u2(torch.cat([self.up2(m), h2], dim=1), emb)
        u = self.u1(torch.cat([self.up1(u), h1], dim=1), emb)
        return self.out(u)


# ============================================================ 3. o DDPM em si
def cosine_alphas(T):
    """Schedule cosseno (Nichol & Dhariwal) — melhor que o linear em seq. curtas."""
    s = 0.008
    x = torch.linspace(0, T, T + 1)
    f = torch.cos((x / T + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return betas, torch.cumprod(1 - betas, dim=0)


@torch.no_grad()
def sample_block(model, z_cent, alt_cent, abar, steps=50, dev="cpu", center=None,
                 x0_clip=5.0):
    """DDIM: parte de ruído puro e devolve o bloco de latentes [B, latent, L].

    A cada passo a posição central é forçada de volta ao `z_cent` conhecido.

    `x0_clip` NÃO é gambiarra, é necessário: no fim do schedule cosseno
    abar[T-1] ~ 2e-9, então `x0 = (x - sqrt(1-a)*eps)/sqrt(a)` divide por ~5e-5 e
    amplifica qualquer erro do eps em ~20000x. Sem clamp o latente sai com
    std ~1e4 (o real tem std 1) e o decoder satura em preto-e-branco. Como os
    latentes são padronizados, sabemos a priori que |z| < ~5.
    """
    model.eval()
    B, L = z_cent.shape[0], model.L
    c = L // 2 if center is None else center
    x = torch.randn(B, model.latent, L, device=dev)
    x[:, :, c] = z_cent
    ts = torch.linspace(len(abar) - 1, 0, steps).long().to(dev)
    for i, t in enumerate(ts):
        eps = model(x, t.repeat(B), z_cent, alt_cent)
        a_t = abar[t]
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-x0_clip, x0_clip)
        # recalcula o eps consistente com o x0 já clampado, senão os dois brigam
        eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        x[:, :, c] = z_cent          # a fatia dada é lei, não sugestão
    return x


# ================================================================ 4. o treino
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--block", type=int, default=32, help="L: fatias por bloco")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=50, help="passos de DDIM na amostragem")
    ap.add_argument("--base", type=int, default=192)
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L, c = args.block, args.block // 2

    # --- dados: fatias -> latentes -> janelas
    vae, _ = load_vae(dev=dev)
    X, A, P = load_slices()
    Z = encode_latents(vae, X, A, dev=dev)

    # padroniza os latentes: a difusão assume dados ~N(0,1). Sem isso o schedule
    # de ruído não bate com a escala dos dados e o treino não anda.
    z_mean, z_std = Z.mean(0), Z.std(0) + 1e-6
    Zn = (Z - z_mean) / z_std

    idx, pat = build_windows(A, P, L=L)
    tr, va = split_by_patient(pat)
    n_alturas = int(A.max().item()) + 1

    model = LatentUNet1D(latent=Z.shape[1], L=L, base=args.base,
                         n_alturas=n_alturas).to(dev)
    betas, abar = cosine_alphas(args.timesteps)
    abar = abar.to(dev)

    if args.sample_only:
        ck = torch.load("ldm.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        z_mean, z_std = ck["z_mean"], ck["z_std"]
    else:
        train_dl = DataLoader(TensorDataset(idx[tr]), batch_size=args.batch_size,
                              shuffle=True)
        val_dl = DataLoader(TensorDataset(idx[va]), batch_size=args.batch_size)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"device={dev}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
              f"L={L}  latent={Z.shape[1]}")

        def batch_loss(w):
            seq = Zn[w].transpose(1, 2).to(dev)          # [B, latent, L]
            alt = A[w[:, c]].to(dev)                     # altura da fatia central
            z_cent = seq[:, :, c]
            t = torch.randint(0, args.timesteps, (len(seq),), device=dev)
            noise = torch.randn_like(seq)
            a = abar[t][:, None, None]
            z_t = a.sqrt() * seq + (1 - a).sqrt() * noise
            z_t[:, :, c] = z_cent                        # central nunca é ruidosa
            eps = model(z_t, t, z_cent, alt)
            # não cobra perda na posição central: ela é dada, não predita
            m = torch.ones(1, 1, L, device=dev)
            m[:, :, c] = 0
            return ((eps - noise) ** 2 * m).sum() / (m.sum() * len(seq) * seq.shape[1])

        for ep in range(1, args.epochs + 1):
            model.train()
            t0, tot = time.time(), 0.0
            for (w,) in tqdm(train_dl, desc=f"época {ep}/{args.epochs}", leave=False):
                loss = batch_loss(w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item()
            model.eval()
            with torch.no_grad():
                vtot = sum(batch_loss(w).item() for (w,) in val_dl) / len(val_dl)
            print(f"ep {ep:3d} | treino mse {tot/len(train_dl):.4f} | val mse {vtot:.4f}"
                  f" | {time.time()-t0:.0f}s")

        torch.save({"model": model.state_dict(), "z_mean": z_mean, "z_std": z_std,
                    "L": L, "base": args.base, "latent": Z.shape[1],
                    "timesteps": args.timesteps}, "ldm.pt")
        print("modelo salvo em ldm.pt")

    # --- figura: bloco real vs bloco gerado só a partir da fatia central
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # janelas ao acaso: linspace cairia na ÚLTIMA janela de cada paciente (todas
    # terminam no topo da cabeça e ficam parecidas entre si)
    g = torch.Generator().manual_seed(1)
    w = idx[va][torch.randperm(len(va), generator=g)[:3]]
    seq = Zn[w].transpose(1, 2).to(dev)
    alt = A[w[:, c]].to(dev)
    gen = sample_block(model, seq[:, :, c], alt, abar, steps=args.steps, dev=dev)

    # latentes -> imagens (desfaz a padronização antes de decodificar)
    def to_img(seq_n, w_row):
        z = seq_n.transpose(1, 2).reshape(-1, Z.shape[1]).cpu() * z_std + z_mean
        a = A[w_row].reshape(-1)
        with torch.no_grad():
            return vae.decode(z.to(dev), a.to(dev)).cpu()

    # a posição central TEM que aparecer: é a única entrada do modelo
    show = torch.unique(torch.cat([torch.linspace(0, L - 1, 7).long(),
                                   torch.tensor([c])]))
    ncol = len(show)
    fig, ax = plt.subplots(6, ncol, figsize=(2 * ncol, 12))
    for b in range(3):
        real = to_img(seq[b:b + 1], w[b:b + 1])
        fake = to_img(gen[b:b + 1], w[b:b + 1])
        for j, s in enumerate(show):
            ax[2 * b, j].imshow(np.rot90(real[s, 0].numpy()), cmap="gray")
            ax[2 * b + 1, j].imshow(np.rot90(fake[s, 0].numpy()), cmap="gray")
            marca = " <- DADA" if int(s) == c else ""
            ax[2 * b, j].set_title(f"z={int(A[w[b, s]])}{marca}", fontsize=8)
            ax[2 * b, j].axis("off"); ax[2 * b + 1, j].axis("off")
    plt.suptitle("bloco real (linhas ímpares) vs gerado pela difusão a partir só da "
                 "fatia central (linhas pares) — pacientes de validação")
    plt.tight_layout(); plt.savefig("ldm_bloco.png", dpi=90)
    print("figura salva em ldm_bloco.png")
    print("bloco gerado:", tuple(gen.shape), "-> L fatias de", tuple(vae.in_hw))


if __name__ == "__main__":
    main()

"""
train_vae.py
============
Treina o CVAE (foto + altura -> vetor z) usando o índice salvo em `oasis_index.pt`.

    python train_vae.py                 # treino padrão
    python train_vae.py --epochs 5      # mais curto

Passos:
  1. `OASISDataset.load("oasis_index.pt")` -> recarrega o índice sem re-escanear.
  2. Pré-carrega TODAS as fatias na RAM uma vez (redimensionadas) e guarda num
     cache .pt. Sem isso, `shuffle=True` faz o dataset reler um volume de 12 MB
     por amostra (~4 min/época só de I/O). Com o cache, a época fica em segundos.
  3. Split treino/val POR PACIENTE (não por fatia!) — fatias vizinhas do mesmo
     paciente são quase idênticas; separar por fatia vazaria treino no val.
  4. Treina com KL warmup e salva `vae.pt` + uma figura de reconstruções.
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from oasis_dataset import OASISDataset
from vae import CVAE, vae_loss


# ---------------------------------------------------------------- 1+2. os dados
def load_slices(index_path="oasis_index.pt", size=(88, 104), cache=None):
    """Carrega todas as fatias na RAM (com cache em disco).

    Retorna X [N,1,H,W] float32 em [0,1], A [N] altura, P [N] número do paciente.
    """
    if cache is None:
        cache = f"oasis_slices_{size[0]}x{size[1]}.pt"
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        print(f"cache carregado: {cache}  ({len(d['X'])} fatias)")
        return d["X"], d["A"], d["P"]

    ds = OASISDataset.load(index_path)
    print(f"pré-carregando {len(ds)} fatias de {len(ds.volumes)} pacientes -> {cache}")
    X = torch.empty(len(ds), 1, *size, dtype=torch.float32)
    A = torch.empty(len(ds), dtype=torch.long)
    P = torch.empty(len(ds), dtype=torch.long)
    # ordem sequencial de propósito: o índice é agrupado por volume, então o
    # cache de 1 volume do OASISDataset acerta e lemos cada .img só uma vez.
    for i in tqdm(range(len(ds))):
        s = ds[i]
        img = s["image"][None]                                  # [1,1,H,W]
        img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        X[i] = img[0]
        A[i] = s["altura"]
        P[i] = s["patient"]
    torch.save({"X": X, "A": A, "P": P}, cache)
    print(f"salvo {cache}  ({X.numel() * 4 / 1e6:.0f} MB)")
    return X, A, P


# ------------------------------------------------------------- 3. split honesto
def split_by_patient(P, val_frac=0.2, seed=0):
    """Separa treino/val por paciente, para não vazar fatias vizinhas."""
    pacientes = torch.unique(P)
    g = torch.Generator().manual_seed(seed)
    perm = pacientes[torch.randperm(len(pacientes), generator=g)]
    n_val = max(1, int(round(len(pacientes) * val_frac)))
    val_pat = set(perm[:n_val].tolist())
    is_val = torch.tensor([int(p) in val_pat for p in P])
    tr = (~is_val).nonzero(as_tuple=True)[0]
    va = is_val.nonzero(as_tuple=True)[0]
    print(f"pacientes: {len(pacientes)} -> treino {len(pacientes)-n_val} / val {n_val}")
    print(f"fatias   : treino {len(tr)} / val {len(va)}")
    return tr, va


# ------------------------------------------------------------------ 4. o treino
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0, help="peso final do termo KL")
    ap.add_argument("--warmup", type=int, default=5, help="épocas subindo o beta de 0 até --beta")
    ap.add_argument("--index", default="oasis_index.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, A, P = load_slices(args.index)
    tr_idx, va_idx = split_by_patient(P)

    train_ds = TensorDataset(X[tr_idx], A[tr_idx])
    val_ds = TensorDataset(X[va_idx], A[va_idx])
    # os tensores já estão na RAM -> num_workers=0 é o mais rápido aqui
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    n_alturas = int(A.max().item()) + 1
    model = CVAE(latent=args.latent, n_alturas=n_alturas, in_hw=tuple(X.shape[-2:])).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device={dev}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  z={args.latent}")

    for ep in range(1, args.epochs + 1):
        # KL warmup: começa como autoencoder puro e vai virando VAE.
        # Sem isso o KL esmaga o z no início (posterior collapse).
        beta = args.beta * min(1.0, ep / max(1, args.warmup))

        model.train()
        t0, tot = time.time(), np.zeros(3)
        for x, alt in tqdm(train_dl, desc=f"época {ep}/{args.epochs}", leave=False):
            x, alt = x.to(dev), alt.to(dev)
            x_hat, mu, lv = model(x, alt)
            loss, rec, kl = vae_loss(x_hat, x, mu, lv, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += [loss.item(), rec.item(), kl.item()]
        tr_loss, tr_rec, tr_kl = tot / len(train_dl)

        model.eval()
        vtot = np.zeros(3)
        with torch.no_grad():
            for x, alt in val_dl:
                x, alt = x.to(dev), alt.to(dev)
                x_hat, mu, lv = model(x, alt)
                loss, rec, kl = vae_loss(x_hat, x, mu, lv, beta)
                vtot += [loss.item(), rec.item(), kl.item()]
        va_loss, va_rec, va_kl = vtot / len(val_dl)

        print(f"ep {ep:3d}  beta {beta:.2f} | treino rec {tr_rec:7.1f} kl {tr_kl:6.1f} "
              f"| val rec {va_rec:7.1f} kl {va_kl:6.1f} | {time.time()-t0:.0f}s")

    torch.save({"model": model.state_dict(),
                "latent": args.latent,
                "n_alturas": n_alturas,
                "in_hw": tuple(X.shape[-2:])}, "vae.pt")
    print("modelo salvo em vae.pt")

    # figura: original vs reconstrução (dados de validação, pacientes não vistos)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    # amostra alturas variadas do val (o val_dl não embaralha: pegaria só as
    # fatias mais baixas do primeiro paciente)
    pick = torch.linspace(0, len(val_ds) - 1, 8).long()
    x = torch.stack([val_ds[i][0] for i in pick])
    alt = torch.tensor([val_ds[i][1] for i in pick])
    with torch.no_grad():
        x_hat, mu, _ = model(x.to(dev), alt.to(dev))
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.5))
    for j in range(8):
        ax[0, j].imshow(np.rot90(x[j, 0].numpy()), cmap="gray")
        ax[0, j].set_title(f"z={int(alt[j])}", fontsize=9)
        ax[1, j].imshow(np.rot90(x_hat[j, 0].cpu().numpy()), cmap="gray")
        ax[0, j].axis("off"); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("original"); ax[1, 0].set_ylabel("recon")
    plt.suptitle("original (cima) vs reconstrução do VAE (baixo) — pacientes de validação")
    plt.tight_layout(); plt.savefig("vae_recon.png", dpi=90)
    print("figura salva em vae_recon.png")
    print("vetor comprimido z:", tuple(mu.shape))


if __name__ == "__main__":
    main()

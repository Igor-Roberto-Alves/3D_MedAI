"""
zsr.py
======
Super-resolucao no eixo Z com difusao 3D condicional, direto no espaco de voxel.

    fatias axiais ESPARSAS (a cada `gap`)  -->  volume isotropico inteiro

    z=0  [imagem]  --.
    z=8  [imagem]  --+--> UNet3D + DDPM --> bloco [D, 88, 104] completo
    z=16 [imagem]  --'                       (as alturas do meio sao INVENTADAS)

Por que esse problema, e nao "cubo inteiro a partir da fatia central"
--------------------------------------------------------------------
A informacao de paciente de uma fatia tem alcance de ~10 alturas (medido: a
correlacao do residuo cai de 1.00 para 0.14 em +-10 e para 0.01 em +-30). Uma
fatia central NAO determina o topo da cabeca -- nao por falta de modelo, por
falta de informacao. Com ancoras a cada `gap`, todo voxel fica a <=gap/2 de uma
ancora conhecida, dentro do alcance. O problema passa a ser resolvivel.

Por que no VOXEL, e nao no latente do CVAE
------------------------------------------
O teto do pipeline latente (encode+decode das fatias REAIS, ou seja, difusao
perfeita) e 19.97 dB. A interpolacao linear entre ancoras com gap=8 da 22.65 dB.
O latente flat de 128 dims perde para a baseline burra antes de comecar: o
gargalo e o VAE, nao a difusao. Aqui nao ha VAE no meio.

Por que DIFUSAO, e nao uma regressao
------------------------------------
O problema e mal-posto: muitos volumes sao consistentes com as mesmas ancoras.
Uma regressao (L2) converge para a MEDIA da posterior -> borrado, que e
exatamente o que a interpolacao linear ja faz de graga. A difusao amostra DA
posterior -> textura com estatistica de MRI de verdade. Ver `--n-samples`: as
ancoras sao as mesmas, o meio varia. Essa variacao E a incerteza do problema.

Uso
---
    python zsr.py --epochs 60 --gap 8         # treina
    python zsr.py --eval-only                 # so avalia + figura

Gera `zsr.pt` e `zsr_recon.png`. A avaliacao SEMPRE compara contra a
interpolacao linear -- se nao ganhar dela, o modelo nao serve.
"""

import os
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from train_vae import load_slices, split_by_patient


# ==================================================== 1. blocos e ancoras
def build_blocks(A, P, D=16, stride=4):
    """Blocos de D alturas CONTIGUAS do mesmo paciente -> idx [M, D].

    `stride` controla a sobreposicao entre blocos vizinhos (aumento de dados).
    stride=D daria blocos disjuntos; stride pequeno da muito mais blocos, cada
    um com um alinhamento diferente em relacao a grade de ancoras.
    """
    A_np, P_np = A.numpy(), P.numpy()
    order = np.lexsort((A_np, P_np))              # por paciente, depois altura
    blocos, pats = [], []
    ini = 0
    for i in range(1, len(order) + 1):
        # fim de um trecho contiguo (mudou de paciente ou pulou altura)
        fim = (i == len(order) or P_np[order[i]] != P_np[order[i - 1]]
               or A_np[order[i]] != A_np[order[i - 1]] + 1)
        if fim:
            run = order[ini:i]
            for j in range(0, len(run) - D + 1, stride):
                blocos.append(run[j:j + D])
                pats.append(P_np[run[j]])
            ini = i
    idx = torch.from_numpy(np.array(blocos)).long()
    pat = torch.from_numpy(np.array(pats)).long()
    print(f"blocos de {D} alturas contiguas: {len(idx)}")
    return idx, pat


def anchor_mask(D, gap, offset, dev="cpu"):
    """Mascara [D] com True nas alturas ANCORA (conhecidas).

    O `offset` desloca a grade. No treino ele e aleatorio: senao o modelo decora
    "as posicoes 0,8,16 sao faceis" em vez de aprender a interpolar, e quebra
    quando a aquisicao real cair noutro alinhamento.
    """
    m = torch.zeros(D, dtype=torch.bool, device=dev)
    m[offset % gap::gap] = True
    m[0] = True                                   # as bordas do bloco sempre
    m[D - 1] = True                               # ancoradas: sem elas o modelo
    return m                                      # extrapola em vez de interpolar


def linear_interp(x, mask):
    """Baseline honesta: interpola linearmente ao longo de Z entre as ancoras.

    x    [B, 1, D, H, W]  com as ancoras corretas (o resto e ignorado)
    mask [D] bool
    Retorna [B, 1, D, H, W]. E isto que a difusao precisa BATER.
    """
    D = x.shape[2]
    pos = mask.nonzero(as_tuple=True)[0].float()          # [K]
    tgt = torch.arange(D, device=x.device).float()
    j = torch.clamp(torch.searchsorted(pos.contiguous(), tgt.contiguous()),
                    1, len(pos) - 1)
    w = ((tgt - pos[j - 1]) / (pos[j] - pos[j - 1])).clamp(0, 1)     # [D]
    lo = x[:, :, pos[j - 1].long()]
    hi = x[:, :, pos[j].long()]
    return lo * (1 - w)[None, None, :, None, None] + hi * w[None, None, :, None, None]


# ========================================================= 2. a UNet 3D
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, cin), cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = nn.GroupNorm(min(groups, cout), cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight)         # bloco comeca como identidade:
        nn.init.zeros_(self.conv2.bias)           # treino bem mais estavel

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnZ(nn.Module):
    """Self-attention SO ao longo de Z (cada coluna (h,w) e uma sequencia de D).

    Z e o eixo do problema: a informacao precisa viajar da ancora ate o meio do
    gap. Attention 3D cheia custaria (D*H*W)^2; esta custa D^2 por coluna, e e a
    direcao que importa.
    """

    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.att = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.norm(x).permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)
        h, _ = self.att(h, h, h)
        h = h.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2)
        return x + h


class UNet3D(nn.Module):
    """Preve `v` a partir de [x_t ; x_cond ; mask] -> 3 canais de entrada.

    x_cond = o volume com as ancoras nas posicoes certas e ZERO no resto.
    mask   = 1 nas ancoras, 0 no resto. Sem esse canal o modelo nao consegue
             distinguir "ancora preta" de "altura desconhecida".
    """

    def __init__(self, base=48, emb_dim=192):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.conv_in = nn.Conv3d(3, base, 3, padding=1)
        self.d1 = ResBlock3D(base, base, emb_dim)
        self.down1 = nn.Conv3d(base, base, 4, 2, 1)               # /2 nos 3 eixos
        self.d2 = ResBlock3D(base, base * 2, emb_dim)
        self.down2 = nn.Conv3d(base * 2, base * 2, 4, 2, 1)       # /4

        self.mid1 = ResBlock3D(base * 2, base * 2, emb_dim)
        self.mid_att = AttnZ(base * 2)
        self.mid2 = ResBlock3D(base * 2, base * 2, emb_dim)

        self.up2 = nn.ConvTranspose3d(base * 2, base * 2, 4, 2, 1)
        self.u2 = ResBlock3D(base * 4, base, emb_dim)             # +skip d2
        self.up1 = nn.ConvTranspose3d(base, base, 4, 2, 1)
        self.u1 = ResBlock3D(base * 2, base, emb_dim)             # +skip d1
        self.out = nn.Sequential(
            nn.GroupNorm(min(8, base), base), nn.SiLU(),
            nn.Conv3d(base, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x_t, t, x_cond, mask):
        m = mask.float()[None, None, :, None, None].expand_as(x_t)
        h = self.conv_in(torch.cat([x_t, x_cond, m], dim=1))
        emb = self.t_mlp(timestep_embedding(t, self.emb_dim))

        h1 = self.d1(h, emb)
        h2 = self.d2(self.down1(h1), emb)
        mid = self.down2(h2)
        mid = self.mid2(self.mid_att(self.mid1(mid, emb)), emb)
        u = self.u2(torch.cat([self._fit(self.up2(mid), h2), h2], dim=1), emb)
        u = self.u1(torch.cat([self._fit(self.up1(u), h1), h1], dim=1), emb)
        return self.out(u)

    @staticmethod
    def _fit(x, ref):
        # 88->44->22 e 104->52->26 fecham; mas D impar ou outro H/W nao. Corta/
        # completa pra nunca quebrar o skip por 1 voxel de arredondamento.
        if x.shape[2:] == ref.shape[2:]:
            return x
        d = [ref.shape[i] - x.shape[i] for i in (2, 3, 4)]
        return F.pad(x, [0, d[2], 0, d[1], 0, d[0]])


# ======================================================== 3. o DDPM (v-pred)
def cosine_alphas(T):
    s = 0.008
    x = torch.linspace(0, T, T + 1)
    f = torch.cos((x / T + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return betas, torch.cumprod(1 - betas, dim=0)


# v-parametrizacao (Salimans & Ho). Trocamos eps por v de proposito:
#   v = sqrt(abar)*eps - sqrt(1-abar)*x0
# Com eps-pred, x0 = (x_t - sqrt(1-abar)*eps)/sqrt(abar) divide por sqrt(abar),
# que no fim do schedule cosseno vale ~1.5e-3 -> amplifica o erro ~650x e o x0
# satura (foi o que exigiu o `x0_clip` em complete_diffusion.py). Com v-pred,
# x0 = sqrt(abar)*x_t - sqrt(1-abar)*v e uma combinacao CONVEXA: nao ha divisao,
# nao explode, e nao precisa de clamp nenhum.
def v_from(x0, eps, a):
    return a.sqrt() * eps - (1 - a).sqrt() * x0


def x0_from_v(x_t, v, a):
    return a.sqrt() * x_t - (1 - a).sqrt() * v


def eps_from_v(x_t, v, a):
    return a.sqrt() * v + (1 - a).sqrt() * x_t


@torch.no_grad()
def sample(model, x_cond, mask, abar, steps=50, dev="cpu"):
    """DDIM com v-pred. Devolve o bloco completo [B, 1, D, H, W] em [-1, 1]."""
    model.eval()
    x = torch.randn_like(x_cond)
    ts = torch.linspace(len(abar) - 1, 0, steps).long().to(dev)
    for i, t in enumerate(ts):
        a_t = abar[t]
        v = model(x, t.repeat(len(x)), x_cond, mask)
        x0 = x0_from_v(x, v, a_t).clamp(-1, 1)     # clamp aqui e so o range real
        eps = eps_from_v(x, v, a_t)                # da imagem, nao um curativo
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    # as ancoras sao CONHECIDAS: nao ha motivo pra devolver a versao gerada delas
    x[:, :, mask] = x_cond[:, :, mask]
    return x


@torch.no_grad()
def reconstruct_volume(model, vol, gap, D, abar, steps=50, dev="cpu", overlap=None,
                       seed=None):
    """Reconstroi o volume INTEIRO (nao um bloco) costurando blocos deslizantes.

    vol : [1, Dtot, H, W] real, em [-1,1]. So as ancoras dele sao usadas.
    Retorna [1, Dtot, H, W] gerado.

    Cada bloco e uma amostra INDEPENDENTE da posterior, entao dois blocos vizinhos
    nao concordam na sobreposicao. Duas coisas seguram a costura:
      - a grade de ancoras e global, entao todo bloco ve as MESMAS ancoras (e as
        reproduz exatas) -> a discordancia fica so nas alturas interpoladas;
      - blending por janela de Hann na sobreposicao, em vez de corte seco.
    """
    if seed is not None:
        torch.manual_seed(seed)
    Dtot = vol.shape[1]
    overlap = D // 2 if overlap is None else overlap
    stride = max(1, D - overlap)
    acc = torch.zeros_like(vol)
    wsum = torch.zeros(1, Dtot, 1, 1, device=vol.device)
    # Hann vai a zero nas pontas; +1e-3 pra nenhuma altura ficar com peso 0 total
    janela = torch.hann_window(D, periodic=False, device=vol.device) + 1e-3

    inicios = list(range(0, max(1, Dtot - D + 1), stride))
    if inicios[-1] != Dtot - D:
        inicios.append(Dtot - D)                 # ultimo bloco encosta no fim
    for i in tqdm(inicios, desc="costurando blocos", leave=False):
        sl = slice(i, i + D)
        x = vol[:, sl].unsqueeze(1)              # [1,1,D,H,W]
        # a mascara e recortada da grade GLOBAL: o alinhamento das ancoras nao
        # pode depender de onde o bloco comeca, senao cada bloco ve outra coisa
        m_glob = torch.zeros(Dtot, dtype=torch.bool, device=vol.device)
        m_glob[::gap] = True
        m_glob[0] = True; m_glob[Dtot - 1] = True
        mask = m_glob[sl].clone()
        mask[0] = True; mask[D - 1] = True       # bordas do bloco, como no treino
        x_cond = torch.zeros_like(x)
        x_cond[:, :, mask] = x[:, :, mask]
        gen = sample(model, x_cond, mask, abar, steps=steps, dev=dev)[:, 0]
        acc[:, sl] += gen * janela[None, :, None, None]
        wsum[:, sl] += janela[None, :, None, None]
    out = acc / wsum
    # as ancoras globais sao conhecidas: devolve o valor real nelas
    m_glob = torch.zeros(Dtot, dtype=torch.bool, device=vol.device)
    m_glob[::gap] = True; m_glob[0] = True; m_glob[Dtot - 1] = True
    out[:, m_glob] = vol[:, m_glob]
    return out, m_glob


# ============================================================ 4. avaliacao
def psnr_g(a, b):
    """PSNR global (MSE agregado). Media de PSNR por fatia daria inf nas ancoras."""
    return -10 * math.log10(((a - b) ** 2).mean().item() + 1e-12)


@torch.no_grad()
def evaluate(model, X, idx, abar, gap, D, steps=50, dev="cpu", n_batch=8, bs=4):
    """Compara difusao vs interpolacao linear SO nas alturas nao-ancora."""
    model.eval()
    mask = anchor_mask(D, gap, 0, dev)
    nao_anc = ~mask
    dif, lin = [], []
    for i in range(0, min(n_batch * bs, len(idx)), bs):
        w = idx[i:i + bs]
        x = (X[w.reshape(-1)].reshape(len(w), D, 1, *X.shape[-2:])
             .permute(0, 2, 1, 3, 4).to(dev) * 2 - 1)             # [-1,1]
        x_cond = torch.zeros_like(x)
        x_cond[:, :, mask] = x[:, :, mask]
        gen = sample(model, x_cond, mask, abar, steps=steps, dev=dev)
        li = linear_interp(x, mask)
        dif.append(((gen[:, :, nao_anc] - x[:, :, nao_anc]) ** 2).mean().item())
        lin.append(((li[:, :, nao_anc] - x[:, :, nao_anc]) ** 2).mean().item())
    # /4 desfaz a escala [-1,1] -> [0,1], pra bater com os numeros das baselines
    d = -10 * math.log10(np.mean(dif) / 4 + 1e-12)
    l = -10 * math.log10(np.mean(lin) / 4 + 1e-12)
    return d, l


# ============================================================== 5. o treino
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=16, help="D: alturas por bloco")
    ap.add_argument("--gap", type=int, default=8, help="1 ancora a cada `gap` alturas")
    ap.add_argument("--stride", type=int, default=4, help="passo entre blocos vizinhos")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=50, help="passos de DDIM")
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--n-samples", type=int, default=3, help="amostras por bloco na figura")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = args.depth

    X, A, P = load_slices()
    idx, pat = build_blocks(A, P, D=D, stride=args.stride)
    # split derivado de P (nao de `pat`): se um paciente sumir do build_blocks,
    # `unique(pat)` muda e o randperm embaralha diferente -> paciente de val do
    # VAE viraria treino aqui, silenciosamente. Assim o split e sempre o mesmo.
    tr_s, va_s = split_by_patient(P)
    val_pats = set(P[va_s].tolist())
    is_val = torch.tensor([int(p) in val_pats for p in pat])
    tr, va = (~is_val).nonzero(as_tuple=True)[0], is_val.nonzero(as_tuple=True)[0]
    print(f"blocos: treino {len(tr)} / val {len(va)}")

    if args.eval_only:
        # a geometria vem do CHECKPOINT, nao dos args: senao um --base diferente
        # do treino quebra o load_state_dict (ou pior, carrega torto em silencio)
        ck = torch.load("zsr.pt", map_location=dev, weights_only=False)
        args.base, args.gap, args.timesteps = ck["base"], ck["gap"], ck["timesteps"]
        if ck["D"] != D:
            raise SystemExit(f"checkpoint treinado com --depth {ck['D']}, nao {D}")
        model = UNet3D(base=args.base).to(dev)
        model.load_state_dict(ck["model"])
        print(f"zsr.pt carregado: base={args.base} D={ck['D']} gap={args.gap}")

    _, abar = cosine_alphas(args.timesteps)
    abar = abar.to(dev)

    if not args.eval_only:
        model = UNet3D(base=args.base).to(dev)
        dl = DataLoader(TensorDataset(idx[tr]), batch_size=args.batch_size,
                        shuffle=True, drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"device={dev}  params={n:.1f}M  D={D}  gap={args.gap}")

        for ep in range(1, args.epochs + 1):
            model.train()
            t0, tot = time.time(), 0.0
            for (w,) in tqdm(dl, desc=f"epoca {ep}/{args.epochs}", leave=False):
                x = (X[w.reshape(-1)].reshape(len(w), D, 1, *X.shape[-2:])
                     .permute(0, 2, 1, 3, 4).to(dev) * 2 - 1)
                # offset aleatorio: o modelo tem que servir pra qualquer
                # alinhamento da grade de ancoras, nao so o do treino
                mask = anchor_mask(D, args.gap, np.random.randint(args.gap), dev)
                x_cond = torch.zeros_like(x)
                x_cond[:, :, mask] = x[:, :, mask]

                t = torch.randint(0, args.timesteps, (len(x),), device=dev)
                noise = torch.randn_like(x)
                a = abar[t][:, None, None, None, None]
                x_t = a.sqrt() * x + (1 - a).sqrt() * noise
                v = model(x_t, t, x_cond, mask)
                # a loss cobra em TODAS as alturas, ancoras inclusive: copiar a
                # ancora e facil e ancora o treino desde a primeira epoca
                loss = F.mse_loss(v, v_from(x, noise, a))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            msg = f"ep {ep:3d} | v-mse {tot/len(dl):.4f} | {time.time()-t0:.0f}s"
            if ep % 10 == 0 or ep == args.epochs:
                d, l = evaluate(model, X, idx[va], abar, args.gap, D,
                                steps=args.steps, dev=dev, n_batch=4)
                msg += f" | val PSNR difusao {d:5.2f} vs interp {l:5.2f} dB"
            print(msg)

        torch.save({"model": model.state_dict(), "D": D, "gap": args.gap,
                    "base": args.base, "timesteps": args.timesteps}, "zsr.pt")
        print("modelo salvo em zsr.pt")

    # ---- veredito + figura
    d, l = evaluate(model, X, idx[va], abar, args.gap, D, steps=args.steps,
                    dev=dev, n_batch=8)
    print(f"\n=== gap={args.gap} | difusao {d:.2f} dB | interp linear {l:.2f} dB "
          f"| ganho {d-l:+.2f} dB ===")
    if d < l:
        print("ainda NAO bate a baseline -- treine mais antes de acreditar na figura")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = anchor_mask(D, args.gap, 0, dev)
    g = torch.Generator().manual_seed(1)
    w = idx[va][torch.randperm(len(va), generator=g)[:1]]
    x = (X[w.reshape(-1)].reshape(1, D, 1, *X.shape[-2:])
         .permute(0, 2, 1, 3, 4).to(dev) * 2 - 1)
    x_cond = torch.zeros_like(x); x_cond[:, :, mask] = x[:, :, mask]
    li = linear_interp(x, mask)
    gens = [sample(model, x_cond, mask, abar, steps=args.steps, dev=dev)
            for _ in range(args.n_samples)]

    linhas = [("real", x), ("interp linear", li)] + \
             [(f"difusao #{i+1}", g_) for i, g_ in enumerate(gens)]
    fig, ax = plt.subplots(len(linhas), D, figsize=(1.5 * D, 1.8 * len(linhas)))
    for r, (nome, vol) in enumerate(linhas):
        for j in range(D):
            ax[r, j].imshow(np.rot90(((vol[0, 0, j].cpu().numpy() + 1) / 2)),
                            cmap="gray", vmin=0, vmax=1)
            ax[r, j].axis("off")
            if r == 0:
                ax[r, j].set_title(f"z={int(A[w[0, j]])}" +
                                   ("\nANCORA" if mask[j] else ""), fontsize=7,
                                   color="tab:green" if mask[j] else "tab:red")
        ax[r, 0].text(-0.35, 0.5, nome, transform=ax[r, 0].transAxes,
                      rotation=90, va="center", ha="center", fontsize=9)
    plt.suptitle(f"gap={args.gap}: verde=ancora (dada), vermelho=inventada  |  "
                 f"difusao {d:.2f} dB vs interp {l:.2f} dB  |  paciente de validacao")
    plt.tight_layout(); plt.savefig("zsr_recon.png", dpi=90)
    print("figura salva em zsr_recon.png")


if __name__ == "__main__":
    main()

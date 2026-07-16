"""
poster.py
=========
Gera as figuras do poster. Divididas em dois grupos:

  SEM MODELO (rodam ja, so precisam do cache de fatias):
    fig1_alcance.png   -- quanto uma fatia sabe sobre as outras alturas
    fig2_baselines.png -- PSNR das baselines vs gap  (+ a difusao, se zsr.pt existir)
    fig5_atlas.png     -- por que NAO fizemos superficie 3D

  COM MODELO (precisam de zsr.pt treinado):
    fig3_qualitativa.png -- axial: real / interp / difusao x N
    fig4_coronal.png     -- CORONAL: a figura que prova a super-resolucao em Z
    fig6_incerteza.png   -- desvio-padrao entre amostras = mapa de incerteza

Uso:
    python poster.py              # tudo que der pra fazer com o que existe
    python poster.py --only 1 4   # so as figuras 1 e 4
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_vae import load_slices, split_by_patient

# Z e 1 mm; o plano foi reduzido 176x208 -> 88x104, entao cada pixel do plano e
# 2 mm. Nas vistas coronal/sagital isso TEM que entrar no aspect, senao o cerebro
# aparece esticado 2x no eixo Z. (E o mesmo motivo pelo qual VOXEL_MM=(1,1,1) do
# reconstruct_3d.py esta errado para o volume gerado: la seria (1.0, 2.0, 2.0).)
DZ, DXY = 1.0, 2.0
ASPECT_ZX = DZ / DXY          # imshow de [Z, X]: altura_pixel/largura_pixel


def volumes_val(X, A, P, va):
    """Volumes contiguos [D,H,W] dos pacientes de validacao, + as alturas."""
    out = []
    for p in torch.unique(P[va]).tolist():
        m = (P == p)
        o = torch.argsort(A[m])
        out.append((X[m][o][:, 0], A[m][o], p))
    return out


# ==================================================== fig 1: alcance (sem modelo)
def fig1_alcance(X, A, P):
    pats = torch.unique(P).tolist()
    C = 85

    def na_altura(a):
        v = []
        for p in pats:
            m = (P == p) & (A == a)
            if not m.any():
                return None
            v.append(X[m][0, 0].flatten())
        return torch.stack(v).numpy()

    def resid(M):
        R = M - M.mean(0, keepdims=True)
        R -= R.mean(1, keepdims=True)
        return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)

    Rc = resid(na_altura(C))
    offs, corr, top1 = [], [], []
    for off in range(0, 51, 2):
        M = na_altura(C + off)
        if M is None:
            continue
        Ra = resid(M)
        S = Rc @ Ra.T
        offs.append(off)
        corr.append(np.diag(S).mean())
        top1.append((S.argmax(1) == np.arange(len(pats))).mean())

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(offs, corr, "o-", color="tab:blue", lw=2, label="correlação do resíduo")
    ax1.axhline(0, color="gray", lw=0.8, ls=":")
    ax1.set_xlabel("distância da fatia âncora (nº de alturas)")
    ax1.set_ylabel("corr. do resíduo de paciente", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(offs, np.array(top1) * 100, "s--", color="tab:red", lw=1.5, ms=4,
             label="identificação do paciente")
    ax2.axhline(100 / len(pats), color="tab:red", lw=0.8, ls=":")
    ax2.text(38, 100 / len(pats) + 2, "acaso", color="tab:red", fontsize=8)
    ax2.set_ylabel("identificação top-1 (%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.axvspan(0, 10, color="tab:green", alpha=0.12)
    ax1.text(4.6, max(corr) * 0.72, "alcance\nútil", ha="center", fontsize=9,
             color="tab:green", fontweight="bold")
    ax1.set_title("Uma fatia de MRI só informa ±10 alturas ao redor\n"
                  "(por isso o condicionamento é por âncoras esparsas, não pela fatia central)",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig("fig1_alcance.png", dpi=150)
    plt.close(fig)
    print("fig1_alcance.png")


# ================================================ fig 2: baselines (sem modelo)
def fig2_baselines(X, A, P, tr, va, gaps=(2, 4, 8, 16, 24, 32)):
    vols = volumes_val(X, A, P, va)
    media_alt = {int(a): X[tr][A[tr] == a].mean(0)[0] for a in torch.unique(A[tr])}

    def psnr_g(a, b):
        return -10 * np.log10(((a - b) ** 2).mean().item() + 1e-12)

    lin_c, near_c = [], []
    for gap in gaps:
        pl, pn = [], []
        for v, al, _ in vols:
            D = len(v)
            anc = torch.arange(0, D, gap)
            if anc[-1] != D - 1:
                anc = torch.cat([anc, torch.tensor([D - 1])])
            pos, tgt = anc.float(), torch.arange(D).float()
            j = torch.clamp(torch.searchsorted(pos, tgt.contiguous()), 1, len(anc) - 1)
            w = ((tgt - pos[j - 1]) / (pos[j] - pos[j - 1])).clamp(0, 1)
            lin = v[anc[j - 1]] * (1 - w)[:, None, None] + v[anc[j]] * w[:, None, None]
            near = v[anc[(tgt[:, None] - pos[None]).abs().argmin(1)]]
            nao = torch.ones(D, dtype=torch.bool); nao[anc] = False
            pl.append(psnr_g(lin[nao], v[nao])); pn.append(psnr_g(near[nao], v[nao]))
        lin_c.append(np.mean(pl)); near_c.append(np.mean(pn))

    mb = torch.stack([media_alt.get(int(a), torch.zeros_like(X[0, 0])) for a in A[va]])
    p_media = psnr_g(mb, X[va][:, 0])

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(gaps, lin_c, "o-", lw=2, label="interpolação linear (baseline a bater)")
    ax.plot(gaps, near_c, "s--", lw=1.4, color="gray", label="vizinho mais próximo")
    ax.axhline(p_media, color="tab:orange", ls=":", lw=1.6,
               label=f"média por altura, ignora o paciente ({p_media:.1f} dB)")
    ax.axhline(19.97, color="tab:red", ls="-.", lw=1.6,
               label="teto do pipeline latente (VAE) — 19.97 dB")

    if os.path.exists("zsr.pt"):
        from zsr import UNet3D, cosine_alphas, evaluate, build_blocks
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load("zsr.pt", map_location=dev, weights_only=False)
        model = UNet3D(base=ck["base"]).to(dev)
        model.load_state_dict(ck["model"])
        _, abar = cosine_alphas(ck["timesteps"]); abar = abar.to(dev)
        idx, pat = build_blocks(A, P, D=ck["D"], stride=4)
        vp = set(P[va].tolist())
        iv = torch.tensor([int(p) in vp for p in pat]).nonzero(as_tuple=True)[0]
        dg = [g for g in gaps if g < ck["D"]]
        dif = [evaluate(model, X, idx[iv], abar, g, ck["D"], dev=dev, n_batch=4)[0]
               for g in dg]
        ax.plot(dg, dif, "^-", lw=2.5, color="tab:green", ms=9,
                label="difusão 3D (zsr.py)")
    else:
        ax.text(0.5, 0.06, "treine o zsr.py para plotar a curva da difusão aqui",
                transform=ax.transAxes, ha="center", fontsize=9, style="italic",
                color="tab:green")

    ax.set_xlabel("gap entre fatias âncora (mm)")
    ax.set_ylabel("PSNR nas alturas NÃO-âncora (dB)")
    ax.set_xscale("log", base=2); ax.set_xticks(gaps)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Quão esparso dá para adquirir?  A difusão só vale se passar da reta azul",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("fig2_baselines.png", dpi=150)
    plt.close(fig)
    print("fig2_baselines.png")


# ================================================== fig 5: atlas (sem modelo)
def fig5_atlas():
    from oasis_dataset import OASISDataset, read_analyze
    ds = OASISDataset.load("oasis_index.pt")
    vols = np.stack([np.clip(read_analyze(ds.volumes[i]) / ds.norm99[i], 0, 1)
                     for i in range(6)])

    def dice(a, b):
        return 2 * (a & b).sum() / (a.sum() + b.sum())

    thrs = [0.001, 0.05, 0.1, 0.2, 0.35, 0.5]
    dd = [np.mean([dice(vols[i] > t, vols[j] > t)
                   for i in range(len(vols)) for j in range(i + 1, len(vols))])
          for t in thrs]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.3),
                                     gridspec_kw={"width_ratios": [1.25, 1, 1]})
    a1.plot(thrs, dd, "o-", lw=2, color="tab:purple")
    a1.axhline(1.0, color="gray", ls=":", lw=1)
    a1.annotate("suporte IDÊNTICO entre pacientes\n(Dice = 1.0000): máscara do atlas,\n"
                "não anatomia do paciente",
                xy=(0.05, 1.0), xytext=(0.13, 0.90), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="tab:red"), color="tab:red")
    a1.set_xlabel("limiar da isosuperfície")
    a1.set_ylabel("Dice entre dois pacientes quaisquer")
    a1.set_title("A superfície externa do OASIS t88_masked\né a mesma para todo mundo",
                 fontsize=10)
    a1.grid(alpha=0.3)

    # painel 2: os contornos do suporte dos 6 pacientes, sobrepostos. Eles caem
    # exatamente uns em cima dos outros -> so se ve UMA linha. (Um mapa de std
    # aqui seria um quadrado preto: correto, mas ilegivel num poster.)
    z = vols.shape[3] // 2
    cores = plt.cm.tab10(np.linspace(0, 1, len(vols)))
    for i in range(len(vols)):
        a2.contour(np.rot90((vols[i, :, :, z] > 0.001).astype(float)), levels=[0.5],
                   colors=[cores[i]], linewidths=1.2, alpha=0.85)
    a2.set_aspect("equal")
    a2.invert_yaxis()
    a2.set_title(f"contorno externo de 6 pacientes (z={z})\n"
                 "6 curvas sobrepostas → só se vê uma", fontsize=10)
    a2.axis("off")

    # painel 3: e aqui esta o sinal. A forma e constante, a TEXTURA varia.
    im = a3.imshow(np.rot90(vols[:, :, :, z].std(0)), cmap="inferno")
    plt.colorbar(im, ax=a3, fraction=0.046)
    a3.set_title("desvio-padrão da INTENSIDADE\ntoda a variação está DENTRO", fontsize=10)
    a3.axis("off")
    fig.suptitle("Por que o projeto modela textura interna, e não superfície 3D", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig5_atlas.png", dpi=150)
    plt.close(fig)
    print("fig5_atlas.png")


# =========================================== figs 3/4/6: precisam de zsr.pt
def figs_com_modelo(X, A, P, va, n_samples=4, steps=50):
    if not os.path.exists("zsr.pt"):
        print("zsr.pt nao existe -> pulando figs 3, 4 e 6 (treine o zsr.py primeiro)")
        return
    from zsr import UNet3D, cosine_alphas, reconstruct_volume, linear_interp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("zsr.pt", map_location=dev, weights_only=False)
    model = UNet3D(base=ck["base"]).to(dev)
    model.load_state_dict(ck["model"])
    _, abar = cosine_alphas(ck["timesteps"]); abar = abar.to(dev)
    D, gap = ck["D"], ck["gap"]

    v, al, pid = volumes_val(X, A, P, va)[0]
    vol = (v[None].to(dev) * 2 - 1)                    # [1, Dtot, H, W] em [-1,1]
    amostras = [reconstruct_volume(model, vol, gap, D, abar, steps=steps, dev=dev,
                                   seed=s)[0] for s in range(n_samples)]
    _, m_glob = reconstruct_volume(model, vol, gap, D, abar, steps=steps, dev=dev, seed=0)
    li = linear_interp(vol.unsqueeze(1), m_glob)[:, 0]
    to01 = lambda t: ((t + 1) / 2).clamp(0, 1).cpu().numpy()

    # ---- fig 4: CORONAL. e aqui que a super-resolucao em Z aparece.
    y = vol.shape[3] // 2
    linhas = [("real", to01(vol)[0, :, :, y]),
              ("interp linear", to01(li)[0, :, :, y]),
              ("difusão", to01(amostras[0])[0, :, :, y])]
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    for a, (nome, im) in zip(axs, linhas):
        a.imshow(np.rot90(im), cmap="gray", vmin=0, vmax=1, aspect=1 / ASPECT_ZX)
        a.set_title(nome, fontsize=11)
        a.axis("off")
        for zz in m_glob.nonzero(as_tuple=True)[0].cpu().numpy():
            a.axhline(im.shape[1] - 1 - 0, color="none")
        a.set_xlabel("Z")
    for zz in m_glob.nonzero(as_tuple=True)[0].cpu().numpy():
        axs[1].axvline(zz, color="tab:green", lw=0.4, alpha=0.55)
        axs[2].axvline(zz, color="tab:green", lw=0.4, alpha=0.55)
    fig.suptitle(f"Vista CORONAL (paciente {pid}, val) — linhas verdes = fatias âncora, gap={gap}\n"
                 "a interpolação borra entre as âncoras; a difusão mantém textura",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("fig4_coronal.png", dpi=150)
    plt.close(fig)
    print("fig4_coronal.png")

    # ---- fig 3: axial, no pior caso (bem no meio de dois ancoras)
    nao = (~m_glob).nonzero(as_tuple=True)[0]
    meio = nao[len(nao) // 2].item()
    zs = [meio - 2, meio - 1, meio, meio + 1, meio + 2]
    linhas = [("real", vol), ("interp linear", li)] + \
             [(f"difusão #{i+1}", a) for i, a in enumerate(amostras[:3])]
    fig, axs = plt.subplots(len(linhas), len(zs), figsize=(2.1 * len(zs), 2.3 * len(linhas)))
    for r, (nome, V) in enumerate(linhas):
        for c, zz in enumerate(zs):
            axs[r, c].imshow(np.rot90(to01(V)[0, zz]), cmap="gray", vmin=0, vmax=1)
            axs[r, c].axis("off")
            if r == 0:
                anc = bool(m_glob[zz])
                axs[r, c].set_title(f"z={int(al[zz])}" + ("\nÂNCORA" if anc else ""),
                                    fontsize=8, color="tab:green" if anc else "tab:red")
        axs[r, 0].text(-0.12, 0.5, nome, transform=axs[r, 0].transAxes, rotation=90,
                       va="center", ha="center", fontsize=9)
    fig.suptitle("Axial, no meio do gap (o caso mais difícil) — as 3 amostras têm as mesmas âncoras",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("fig3_qualitativa.png", dpi=150)
    plt.close(fig)
    print("fig3_qualitativa.png")

    # ---- fig 6: incerteza = std entre amostras. so difusao produz isto.
    S = torch.stack(amostras)                          # [N, 1, Dtot, H, W]
    std = S.std(0)[0]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    axs[0].imshow(np.rot90(to01(vol)[0, meio]), cmap="gray")
    axs[0].set_title(f"real (z={int(al[meio])}, meio do gap)", fontsize=10)
    im = axs[1].imshow(np.rot90(std[meio].cpu().numpy()), cmap="inferno")
    axs[1].set_title(f"incerteza: desvio-padrão de {n_samples} amostras", fontsize=10)
    plt.colorbar(im, ax=axs[1], fraction=0.046)
    perfil = std.mean((1, 2)).cpu().numpy()
    axs[2].plot(perfil, lw=1.5)
    for zz in m_glob.nonzero(as_tuple=True)[0].cpu().numpy():
        axs[2].axvline(zz, color="tab:green", lw=0.6, alpha=0.6)
    axs[2].set_xlabel("altura z  (verde = âncora)")
    axs[2].set_ylabel("incerteza média")
    axs[2].set_title("a incerteza zera nas âncoras\ne cresce no meio do gap", fontsize=10)
    axs[2].grid(alpha=0.3)
    for a in axs[:2]:
        a.axis("off")
    fig.suptitle("O que só a difusão te dá: um mapa de onde a aquisição esparsa perdeu informação",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("fig6_incerteza.png", dpi=150)
    plt.close(fig)
    print("fig6_incerteza.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=4)
    args = ap.parse_args()
    quer = lambda n: args.only is None or n in args.only

    X, A, P = load_slices()
    tr, va = split_by_patient(P)
    if quer(1):
        fig1_alcance(X, A, P)
    if quer(2):
        fig2_baselines(X, A, P, tr, va)
    if quer(5):
        fig5_atlas()
    if any(quer(n) for n in (3, 4, 6)):
        figs_com_modelo(X, A, P, va, n_samples=args.n_samples, steps=args.steps)


if __name__ == "__main__":
    main()

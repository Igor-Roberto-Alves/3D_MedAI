"""
oasis_dataset.py
================
Leitor do dataset OASIS-1 (cross-sectional) e um `torch.utils.data.Dataset`
que expõe TODAS as fatias (slices) de MRI como amostras de treino.

Cada amostra é um dicionário:
    {
        'image'  : Tensor float32 [1, H, W]  -> a imagem (fatia 2D normalizada 0..1)
        'altura' : int                        -> a altura (índice da fatia no eixo escolhido)
        'patient': int                        -> o número do paciente (ex: 1 para OAS1_0001)
        'patient_id': str                     -> id completo (ex: 'OAS1_0001_MR1')
    }

As imagens vêm do volume processado `*_t88_masked_gfc.img` (cérebro registrado no
atlas, sem crânio), formato Analyze 7.5 big-endian, shape 176 x 208 x 176 int16.
Não precisa de nibabel: o leitor usa só numpy.

Uso em um script de treino:
    from oasis_dataset import OASISDataset
    from torch.utils.data import DataLoader

    ds = OASISDataset("oasis_cross-sectional_disc1", plane="axial")
    ds.save("oasis_index.pt")          # salva o índice pronto (rápido de recarregar)
    # ds = OASISDataset.load("oasis_index.pt")   # recarrega sem re-escanear

    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for batch in loader:
        x = batch['image']       # [B, 1, H, W]
        h = batch['altura']      # [B]
        p = batch['patient']     # [B]
"""

import os
import re
import glob
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# datatype do header Analyze -> dtype numpy big-endian
_ANALYZE_DTYPE = {2: ">u1", 4: ">i2", 8: ">i4", 16: ">f4", 64: ">f8"}


def read_analyze(img_path):
    """Lê um volume Analyze 7.5 (.hdr + .img) big-endian usando só numpy.

    Retorna um array float32 com o singleton final removido (ex: 176x208x176).
    """
    hdr_path = img_path[:-4] + ".hdr"
    with open(hdr_path, "rb") as f:
        hdr = f.read()
    ndim = struct.unpack(">h", hdr[40:42])[0]
    dims = struct.unpack(">8h", hdr[40:56])[1:1 + ndim]
    datatype = struct.unpack(">h", hdr[70:72])[0]
    dtype = _ANALYZE_DTYPE[datatype]

    vol = np.fromfile(img_path, dtype=dtype).astype(np.float32)
    vol = vol.reshape(dims[::-1])                       # Analyze é fortran-order
    vol = vol.transpose(*range(vol.ndim)[::-1])         # volta para (X, Y, Z, ...)
    return np.squeeze(vol)                              # remove eixo de tamanho 1


# eixo do volume correspondente a cada plano anatômico
#   volume shape = (X=sagital 176, Y=coronal 208, Z=axial 176)
_PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


def _patient_number(patient_id):
    """'OAS1_0001_MR1' -> 1 (int)."""
    m = re.search(r"OAS1_(\d+)", patient_id)
    return int(m.group(1)) if m else -1


class OASISDataset(Dataset):
    """Todas as fatias 2D de MRI do OASIS-1 como amostras de treino.

    Parâmetros
    ----------
    root : str
        Pasta raiz do dataset (ex: 'oasis_cross-sectional_disc1').
    plane : {'axial', 'coronal', 'sagittal'}
        Direção de corte. 'axial' -> 'altura' é a altura da cabeça (padrão).
    min_foreground : float
        Descarta fatias com menos que essa fração de pixels de cérebro
        (evita fatias pretas/vazias nas pontas). 0.0 mantém tudo.
    normalize : bool
        Se True, cada fatia é dividida pelo percentil 99 do volume -> ~[0, 1].
    transform : callable | None
        Transform opcional aplicado ao Tensor [1, H, W].
    """

    def __init__(self, root, plane="axial", min_foreground=0.02,
                 normalize=True, transform=None):
        assert plane in _PLANE_AXIS, f"plane deve ser um de {list(_PLANE_AXIS)}"
        self.root = root
        self.plane = plane
        self.axis = _PLANE_AXIS[plane]
        self.min_foreground = min_foreground
        self.normalize = normalize
        self.transform = transform

        pattern = os.path.join(root, "**", "*t88_masked_gfc.img")
        self.volumes = sorted(glob.glob(pattern, recursive=True))
        if not self.volumes:
            raise FileNotFoundError(f"Nenhum volume encontrado em {pattern!r}")

        self.patient_ids = [self._id_from_path(p) for p in self.volumes]
        self.norm99 = [1.0] * len(self.volumes)   # percentil 99 por volume

        # índice: lista de (vol_idx, altura) para cada fatia mantida
        self.index = []
        for vi, vp in tqdm(enumerate(self.volumes), total=len(self.volumes)):
            vol = read_analyze(vp)
            if normalize:
                nz = vol[vol > 0]
                self.norm99[vi] = float(np.percentile(nz, 99)) if nz.size else 1.0
            n = vol.shape[self.axis]
            for k in range(n):
                sl = np.take(vol, k, axis=self.axis)
                if (sl > 0).mean() >= min_foreground:
                    self.index.append((vi, k))

        # cache do último volume lido (o índice é agrupado por volume)
        self._cache_vi = None
        self._cache_vol = None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _id_from_path(path):
        m = re.search(r"OAS1_\d+_MR\d+", path)
        return m.group(0) if m else os.path.basename(path)

    def _get_volume(self, vi):
        if vi != self._cache_vi:
            self._cache_vol = read_analyze(self.volumes[vi])
            self._cache_vi = vi
        return self._cache_vol

    # ------------------------------------------------------------ Dataset API
    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        vi, k = self.index[i]
        vol = self._get_volume(vi)
        sl = np.take(vol, k, axis=self.axis).astype(np.float32)
        if self.normalize:
            sl = np.clip(sl / self.norm99[vi], 0.0, 1.0)

        img = torch.from_numpy(np.ascontiguousarray(sl))[None]  # [1, H, W]
        if self.transform is not None:
            img = self.transform(img)

        patient_id = self.patient_ids[vi]
        return {
            "image": img,
            "altura": int(k),
            "patient": _patient_number(patient_id),
            "patient_id": patient_id,
        }

    # --------------------------------------------------------- salvar / carregar
    def save(self, path):
        """Salva o índice pronto (não salva os pixels; leves ~KB)."""
        torch.save({
            "root": self.root,
            "plane": self.plane,
            "min_foreground": self.min_foreground,
            "normalize": self.normalize,
            "volumes": self.volumes,
            "patient_ids": self.patient_ids,
            "norm99": self.norm99,
            "index": self.index,
        }, path)

    @classmethod
    def load(cls, path, transform=None):
        """Recarrega um dataset salvo com `save` sem re-escanear os volumes."""
        d = torch.load(path, weights_only=False)
        obj = cls.__new__(cls)
        obj.root = d["root"]
        obj.plane = d["plane"]
        obj.axis = _PLANE_AXIS[d["plane"]]
        obj.min_foreground = d["min_foreground"]
        obj.normalize = d["normalize"]
        obj.volumes = d["volumes"]
        obj.patient_ids = d["patient_ids"]
        obj.norm99 = d["norm99"]
        obj.index = d["index"]
        obj.transform = transform
        obj._cache_vi = None
        obj._cache_vol = None
        return obj


if __name__ == "__main__":
    ds = OASISDataset("oasis_cross-sectional_disc1", plane="axial")
    print(f"pacientes: {len(ds.volumes)}  |  fatias de treino: {len(ds)}")
    s = ds[len(ds) // 2]
    print("exemplo -> image:", tuple(s["image"].shape),
          "| altura:", s["altura"],
          "| patient:", s["patient"],
          "|", s["patient_id"])

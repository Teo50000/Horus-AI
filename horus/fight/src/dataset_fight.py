import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from modelo_fight import MEDIA_KINETICS, STD_KINETICS

LADO_FINAL = 112


class RWFDataset(Dataset):
    """Lee los .npz de horus/fight/data/processed/frames/<split>/<clase>/*.npz
    (uno por clip, frames=(T,128,128,3) uint8 RGB, label="fight"/"nofight")."""

    def __init__(self, carpeta, entrenamiento=False):
        self.archivos = sorted(glob.glob(os.path.join(carpeta, "*", "*.npz")))
        self.entrenamiento = entrenamiento
        self.label_map = {"nofight": 0, "fight": 1}
        self.media = torch.tensor(MEDIA_KINETICS).view(1, 1, 1, 3)
        self.std = torch.tensor(STD_KINETICS).view(1, 1, 1, 3)

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, idx):
        npz = np.load(self.archivos[idx], allow_pickle=True)
        frames = npz["frames"]  # (T, 128, 128, 3) uint8
        label = self.label_map[str(npz["label"])]

        lado = frames.shape[1]
        if self.entrenamiento:
            top = np.random.randint(0, lado - LADO_FINAL + 1)
            left = np.random.randint(0, lado - LADO_FINAL + 1)
        else:
            top = left = (lado - LADO_FINAL) // 2  # centro fijo, val reproducible
        frames = frames[:, top:top + LADO_FINAL, left:left + LADO_FINAL, :]

        if self.entrenamiento and np.random.rand() < 0.5:
            frames = frames[:, :, ::-1, :]  # flip horizontal, igual en todos los frames del clip

        clip = torch.tensor(frames.copy(), dtype=torch.float32) / 255.0  # (T,H,W,C)
        clip = (clip - self.media) / self.std
        clip = clip.permute(3, 0, 1, 2)  # (C,T,H,W), lo que espera mc3_18

        return clip, label

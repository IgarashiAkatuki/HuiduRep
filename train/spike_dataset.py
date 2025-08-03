import copy
import numpy as np
import torch
import tqdm
from torch import Tensor

from torch.utils.data import Dataset
from utils.spike_argument import SpikeAugmentation

class SpikeDataset(Dataset):
    def __init__(self, data, transform=None, use_transform=True):
        self.data = torch.tensor(data, dtype=torch.float)
        self.transform = transform
        self.use_transform = use_transform

        if use_transform and transform is None:
            self.transform = SpikeAugmentation(waveform_bank=data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seed1 = np.random.randint(0, 10000)
        seed2 = np.random.randint(0, 10000)

        spike = self.data[idx] # shape: [samples, n_channels]
        spike = torch.tensor(spike, dtype=torch.float)

        spike_1, (start, end) = self.transform(spike, seed=seed1, drop=False, noise=True, noise_prob=0.5)
        view_1 = _z_score_normalize(spike_1).permute(1, 0)

        if self.use_transform:
            spike_2, _ = self.transform(spike, seed=seed2, drop=False, noise=False, noise_prob=0.5)
            view_2 = _z_score_normalize(spike_2).permute(1, 0)
        else:
            view_2 = copy.deepcopy(spike)

        spike = _z_score_normalize(spike[start : end, :]).permute(1, 0)
        return view_1, view_2, spike


def _z_score_normalize(spike_data: Tensor, dim=1, eps=1e-6):
    # spike_datas [n, samples, channels]
    mean = spike_data.mean(dim=dim, keepdim=True)
    std = spike_data.std(dim=dim, keepdim=True)
    # return (spike_data - mean) / (std + eps)
    return spike_data

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path 


# train
class ISEEDataset_Multiple_Hnorm_Unit_Aug(Dataset):

    def __init__(self, dataset_path, b_norm, exc_noaa=None):
        self.files = list(Path(dataset_path).glob('**/input/*.npz'))
        if isinstance(exc_noaa, str):
            self.files = sorted([f for f in self.files if not exc_noaa == str(f.parent.parent.stem)])
        elif isinstance(exc_noaa, list):
            self.files = sorted([f for f in self.files if not exc_noaa[0] == str(f.parent.parent.stem)])
            for t_noaa in exc_noaa[1:]:
                self.files = sorted([f for f in self.files if not t_noaa == str(f.parent.parent.stem)])
        self.b_norm = b_norm
        self.length = len(self.files)

    def __len__(self):
        return 2*self.length

    def __getitem__(self, idx):
        is_flip = False
        if idx >= self.length:
            is_flip = True
            idx = idx - self.length
        input_file = self.files[idx]
        # NLFFF(z=0) [3, 513, 257,  1]
        inputs = torch.from_numpy(np.load(input_file, mmap_mode='r')['input'].astype(np.float32))
        # NLFFF(z=0) [3, 512, 256,  1]  remove duplicated periodic boundary
        inputs = inputs[:, :-1, :-1, :]
        inputs = inputs / self.b_norm
        
        dx = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)
        dy = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)
        dz = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)

        label_file = self.files[idx].parent.parent / 'label' / f'label_{self.files[idx].stem[6:]}.npz'
        # NLFFF     [3, 513, 257, 257]
        labels = torch.from_numpy(np.load(label_file, mmap_mode='r')['label'].astype(np.float32))
        # NLFFF     [3, 512, 256, 256]  remove duplicated periodic boundary
        labels = labels[:, :-1, :-1, :-1]
        divisor = (self.b_norm / np.arange(1, labels.shape[-1] + 1)).reshape(1, 1, 1, -1).astype(np.float32)
        labels = labels / divisor

        if is_flip:
            # rotation by 180 deg around z-axis
            inputs = torch.flip(inputs, dims=(1, 2))
            inputs[0] = -inputs[0]
            inputs[1] = -inputs[1]
            labels = torch.flip(labels, dims=(1, 2))
            labels[0] = -labels[0]
            labels[1] = -labels[1]

        samples = {'input': inputs, 'label': labels, 
                   'input_name': input_file.stem, 'label_name': label_file.stem,
                   'dx': dx, 'dy': dy, 'dz': dz}

        return samples


# validataion, test
class ISEEDataset_Hnorm_Unit_Aug(Dataset):

    def __init__(self, data_path, b_norm):
        if isinstance(data_path, str):
            files = list(Path(data_path).glob('**/input/*.npz'))
        elif isinstance(data_path, list):
            files = []
            for d_path in data_path:
                files += list(Path(d_path).glob('**/input/*.npz'))
        self.files = sorted([f for f in files])
        self.b_norm = b_norm
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_file = self.files[idx]
        # NLFFF(z=0) [3, 513, 257,  1]
        inputs = torch.from_numpy(np.load(input_file, mmap_mode='r')['input'].astype(np.float32))
        # NLFFF(z=0) [3, 512, 256,  1]  remove duplicated periodic boundary
        inputs = inputs[:, :-1, :-1, :]
        inputs = inputs / self.b_norm

        # Assume unit dx, dy, dz
        dx = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)
        dy = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)
        dz = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)

        label_file = self.files[idx].parent.parent / 'label' / f'label_{self.files[idx].stem[6:]}.npz'
        # NLFFF     [3, 513, 257, 257]
        labels = torch.from_numpy(np.load(label_file, mmap_mode='r')['label'].astype(np.float32))
        # NLFFF     [3, 512, 256, 256]  remove duplicated periodic boundary
        labels = labels[:, :-1, :-1, :-1]
        divisor = (self.b_norm / np.arange(1, labels.shape[-1] + 1)).reshape(1, 1, 1, -1).astype(np.float32)
        labels = labels / divisor

        # [3, 513, 257, 257] -> [3, 512, 256, 256]  remove duplicated periodic boundary
        potential = torch.from_numpy(np.load(label_file, mmap_mode='r')['pot'].astype(np.float32))
        potential = potential[:, :-1, :-1, :-1]
        potential = potential / divisor

        samples = {'input': inputs, 'label': labels, 'pot': potential,
                   'input_name': input_file.stem, 'label_name': label_file.stem,
                   'dx': dx, 'dy': dy, 'dz': dz}

        return samples
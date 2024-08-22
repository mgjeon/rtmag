import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path 


# train
class LowLouDataset_Multiple_Hnorm_Unit(Dataset):

    def __init__(self, dataset_path, b_norm):
        self.files = list(Path(dataset_path).glob('**/*.npz'))
        self.b_norm = b_norm
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_file = self.files[idx]
        # NLFFF(z=0) [64, 64, 64, 3]
        labels = torch.from_numpy(np.load(input_file, mmap_mode='r')['b'].astype(np.float32))
        # [x, y, z, 3] -> [3, x, y, z]
        labels = labels.permute(3, 0, 1, 2)
        # [3, x, y, z] -> [3, x, y, 1]
        inputs = labels[:, :, :, 0][:, :, :, None] / self.b_norm
        
        # Assume unit dx, dy, dz
        dx = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)
        dy = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)
        dz = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)

        # NLFFF     [3, 64, 64, 64]
        divisor = (self.b_norm / np.arange(1, labels.shape[-1] + 1)).reshape(1, 1, 1, -1).astype(np.float32)
        labels = labels / divisor

        samples = {'input': inputs, 'label': labels, 
                   'input_name': input_file.stem,
                   'dx': dx, 'dy': dy, 'dz': dz}

        return samples
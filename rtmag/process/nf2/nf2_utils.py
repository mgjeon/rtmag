import numpy as np
from rtmag.process.paper.utils import congrid

def get_bslices(input_file):
    # (3, 513, 257, 1) -> (3, 512, 256, 1) -> (512, 256, 1, 3)
    b_slices = np.load(input_file)['input'][:, :-1, :-1, :].transpose(1, 2, 3, 0)
    return b_slices

def hmi_to_bslices(hmi_b, shape=(512, 256, 3)):
    b_slices = congrid(hmi_b, shape)
    # (Nx, Ny, 3) -> (Nx, Ny, 1, 3)
    return b_slices[..., None, :]
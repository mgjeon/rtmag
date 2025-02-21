# https://github.com/RobertJaro/NF2

import numpy as np
from copy import copy
from astropy.nddata import block_reduce
from nf2.train.data_loader import SlicesDataModule
from skimage.transform import resize
import glob
import os
from sunpy.map import Map
from astropy import units as u
from rtmag.process.nf2.nf2_utils import get_bslices

class NpzCongridDataModule(SlicesDataModule):

    def __init__(self, data_path, slices=None, bin=1, use_bz=False, components=False, *args, **kwargs):
        # (Nx, Ny)
        file = np.load(data_path)
        bx = file['bx']
        by = file['by']
        bz = file['bz']

        # (Nx, Ny, 1, 3)
        b_slices = np.stack([bx, by, bz], axis=-1)
        b_slices = b_slices[:, :, None, :]

        if slices:
            b_slices = b_slices[:, :, slices]
        if bin > 1:
            b_slices = block_reduce(b_slices, (bin, bin, 1, 1), np.mean)
        if use_bz:
            b_slices[:, :, 1:, 0] = None
            b_slices[:, :, 1:, 1] = None
        if components:
            for i, c in enumerate(components):
                filter = [i for i in [0, 1, 2] if i not in c]
                b_slices[:, :, i, filter] = None
        super().__init__(b_slices, *args, **kwargs)


class NpzDataModule(SlicesDataModule):

    def __init__(self, data_path, slices=None, bin=1, use_bz=False, components=False, *args, **kwargs):
        # (Nx, Ny, Nz, 3)
        # (512, 256, 1, 3)
        b_slices = get_bslices(data_path)
        if slices:
            b_slices = b_slices[:, :, slices]
        if bin > 1:
            b_slices = block_reduce(b_slices, (bin, bin, 1, 1), np.mean)
        if use_bz:
            b_slices[:, :, 1:, 0] = None
            b_slices[:, :, 1:, 1] = None
        if components:
            for i, c in enumerate(components):
                filter = [i for i in [0, 1, 2] if i not in c]
                b_slices[:, :, i, filter] = None
        super().__init__(b_slices, *args, **kwargs)


class NpzSeriesDataModule(NpzDataModule):

    def __init__(self, file_paths, *args, **kwargs):
        self.args = args 
        self.kwargs = kwargs 
        self.file_paths = copy(file_paths)

        super().__init__(file_paths[0], *self.args, **self.kwargs)
    
    def train_dataloader(self):
        # re-initialize
        super().__init__(self.file_paths[0], *self.args, **self.kwargs)
        del self.file_paths[0]  # continue with next file in list
        return super().train_dataloader()


class ArrayDataModule(SlicesDataModule):

    def __init__(self, b_slices, *args, **kwargs):
        super().__init__(b_slices, *args, **kwargs)



class SHARPResizeDataModule(SlicesDataModule):

    def __init__(self, data_path, bin=1, slice=None, resize_size=None, *args, **kwargs):
        if isinstance(data_path, str):
            hmi_p = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))[0]  # x
            hmi_t = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))[0]  # y
            hmi_r = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))[0]  # z
            err_p = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))[0]  # x
            err_t = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))[0]  # y
            err_r = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))[0]  # z
        else:
            hmi_p, err_p, hmi_t, err_t, hmi_r, err_r = data_path
        # load maps
        p_map, t_map, r_map = Map(hmi_p),Map(hmi_t),Map(hmi_r)
        p_error_map, t_error_map, r_error_map = Map(err_p), Map(err_t), Map(err_r)

        maps = [p_map, t_map, r_map, p_error_map, t_error_map, r_error_map]
        if slice:
            maps = [m.submap(bottom_left=u.Quantity((slice[0], slice[2]), u.pixel),
                             top_right=u.Quantity((slice[1], slice[3]), u.pixel)) for m in maps]
        if bin > 1:
            maps = [m.superpixel(u.Quantity((bin, bin), u.pixel), func=np.mean) for m in maps]

        hmi_data = np.stack([maps[0].data, -maps[1].data, maps[2].data]).transpose()
        error_data = np.stack([maps[3].data, maps[4].data, maps[5].data]).transpose()

        if resize_size:
            hmi_data = resize(hmi_data, resize_size)
            error_data = resize(error_data, resize_size)

        b_slices = hmi_data[:, :, None]
        error_slices = error_data[:, :, None]
        meta_data = maps[0].meta

        super().__init__(b_slices, *args, error_slices=error_slices, meta_data=meta_data, **kwargs)



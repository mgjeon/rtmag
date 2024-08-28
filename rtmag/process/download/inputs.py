import numpy as np
from pathlib import Path
from rtmag.process.paper.utils import get_hmi, hmi_to_congrid
import gc


def make_input(hmi_path, result_path,
               congrid_shape=[512, 256]):
    
    hmi_path = Path(hmi_path)
    result_path = Path(result_path)
        
    bp_files = sorted(hmi_path.glob("*Bp.fits"))
    bt_files = sorted(hmi_path.glob("*Bt.fits"))
    br_files = sorted(hmi_path.glob("*Br.fits"))

    assert len(bp_files) == len(bt_files) == len(br_files), "Number of Bp, Bt, Br files do not match"

    nx, ny = congrid_shape

    congrid_path = result_path / 'congrid'
    congrid_path.mkdir(parents=True, exist_ok=True)

    original_path = result_path / 'original'
    original_path.mkdir(parents=True, exist_ok=True)


    for bp_file, bt_file, br_file in zip(bp_files, bt_files, br_files):
        timestamp = bp_file.name.split('.')[3]
        assert timestamp == bt_file.name.split('.')[3] == br_file.name.split('.')[3], "Timestamps do not match"

        original_filename = 'hmi_b_' + timestamp + '.npz'
        congrid_filename = 'hmi_b_congrid_' + timestamp + '.npz'

        if not (original_path / original_filename).exists() and not (congrid_path / original_filename).exists():

            hmi_b = get_hmi(bp_file, bt_file, br_file)
            hmi_b_congrid, res_o, res = hmi_to_congrid(hmi_b, nx=nx, ny=ny, isprint=False)

            np.savez(original_path / original_filename, 
                    bx=hmi_b[..., 0], by=hmi_b[..., 1], bz=hmi_b[..., 2],
                    x = res_o['x'], y = res_o['y'],
                    Lx = res_o['Lx'], Ly = res_o['Ly'],
                    nx = res_o['nx'], ny = res_o['ny'],
                    dx = res_o['dx'], dy = res_o['dy'],
                    dx_cm = res_o['dx_cm'], dy_cm = res_o['dy_cm'])
                
            np.savez(congrid_path / congrid_filename, 
                    bx=hmi_b_congrid[..., 0], by=hmi_b_congrid[..., 1], bz=hmi_b_congrid[..., 2],
                    x = res['x'], y = res['y'],
                    Lx = res['Lx'], Ly = res['Ly'],
                    nx = res['nx'], ny = res['ny'],
                    dx = res['dx'], dy = res['dy'],
                    dx_cm = res['dx_cm'], dy_cm = res['dy_cm'])
                    
            print("Processed: ", timestamp)
        
        else:
            print("Already processed: ", timestamp)
        
        del timestamp
        del original_filename
        del congrid_filename
        del hmi_b
        del hmi_b_congrid
        del res_o
        del res 

        gc.collect()
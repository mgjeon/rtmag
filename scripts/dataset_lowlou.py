import numpy as np
from rtmag.process.paper.analytical_field import get_analytic_b_field
from rtmag.utils.magplot import create_mesh, mag_plotter
import pyvista as pv
from pathlib import Path

import gc
from tqdm import tqdm

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    args = parser.parse_args()

    test_path = Path(args.test_path)
    train_path = Path(args.train_path)
    train_path_png = train_path / 'png'

    test_path.mkdir(parents=True, exist_ok=True)
    train_path.mkdir(parents=True, exist_ok=True)
    train_path_png.mkdir(parents=True, exist_ok=True)


    l = 0.3
    psi = 0.25*np.pi
    b_test = get_analytic_b_field(l=l, psi=psi)
    b_test.shape

    np.savez(test_path / 'case1.npz', b=b_test)

    bx = b_test[:, :, :, 0]
    by = b_test[:, :, :, 1]
    bz = b_test[:, :, :, 2]

    mesh = create_mesh(bx, by, bz)
    i_siz = 32
    j_siz = 32
    i_res = 8
    j_res = 8
    vmin = -200
    vmax = 200
    max_time = 10000

    b_plot = mag_plotter(mesh)
    b_tube, b_bottom, b_dargs = b_plot.create_mesh(i_siz=i_siz, j_siz=j_siz, 
                                                i_resolution=i_res, j_resolution=j_res, 
                                                vmin=vmin, vmax=vmax, 
                                                max_time=max_time)
    window_size=(800, 800)
    zoom = 0.8

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.add_mesh(b_plot.grid.outline())
    p.add_mesh(b_bottom, cmap='gray', **b_dargs)
    p.add_mesh(b_tube, lighting=False, color='blue')
    p.camera.zoom(zoom)
    p.camera_position = 'xy'
    p.add_title(f"|B| = {np.max(np.abs(b_test)):.3f} | l={l:.3f} | p={psi/np.pi:.3f}π")
    p.screenshot(test_path / f'case1_{l:.3f}_{psi/np.pi:.3f}.png')

    for _ in tqdm(range(10000)):
        # l = 0.1 + 0.3*np.random.rand(1)
        # l = l[0]
        # psi = np.random.rand(1) * np.pi
        # psi = psi[0]

        # l_check = ((l > 0.18) and (l < 0.28)) or ((l > 0.32) and (l < 0.42))
        # psi_check = ((psi > 0*np.pi) and (psi < 0.2*np.pi)) or (psi > 0.3*np.pi)

        l = np.random.uniform(0.15, 0.25) if np.random.rand() > 0.5 else np.random.uniform(0.35, 0.45)
        psi = np.random.uniform(0, 0.2*np.pi) if np.random.rand() > 0.5 else np.random.uniform(0.3*np.pi, np.pi)

        b_train = get_analytic_b_field(l=l, psi=psi)
        np.savez(train_path / f'b_{l:.3f}_{psi/np.pi:.3f}.npz', b=b_train)

        # print(np.max(np.abs(b_train)))

        bx = b_train[:, :, :, 0]
        by = b_train[:, :, :, 1]
        bz = b_train[:, :, :, 2]

        maxb = np.max(np.abs(b_train))

        mesh = create_mesh(bx, by, bz)
        i_siz = 32
        j_siz = 32
        i_res = 8
        j_res = 8
        vmin = -int(maxb)/2
        vmax = int(maxb)/2
        max_time = 10000

        b_plot = mag_plotter(mesh)
        b_tube, b_bottom, b_dargs = b_plot.create_mesh(i_siz=i_siz, j_siz=j_siz, 
                                                    i_resolution=i_res, j_resolution=j_res, 
                                                    vmin=vmin, vmax=vmax, 
                                                    max_time=max_time)
        window_size=(800, 800)
        zoom = 0.8

        p = pv.Plotter(off_screen=True, window_size=window_size)
        p.add_mesh(b_plot.grid.outline())
        p.add_mesh(b_bottom, cmap='gray', **b_dargs)
        p.add_mesh(b_tube, lighting=False, color='blue')
        p.camera.zoom(zoom)
        p.add_title(f"|B| = {maxb:.3f} | l={l:.3f} | p={psi/np.pi:.3f}π")
        p.camera_position = 'xy'
        p.screenshot(train_path_png /f'b_{l:.3f}_{psi/np.pi:.3f}.png')
        p.close()

        del l
        del psi
        del b_train
        del bx
        del by
        del bz
        del maxb
        del mesh
        del i_siz
        del j_siz
        del i_res
        del j_res
        del vmin
        del vmax
        del max_time
        del b_plot
        del b_tube
        del b_bottom
        del b_dargs
        del window_size
        del zoom
        del p

        gc.collect()

    # manually select 5000 files -> train_5000
    # manually select 1000 files -> val_1000
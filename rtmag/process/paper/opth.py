from magplot.base import create_mesh, mag_plotter
import pyvista as pv
pv.set_jupyter_backend('static')

def draw(b, radius=0.2):
    vmin = -200
    vmax = 200
    i_siz=b.shape[0] / 2
    j_siz=b.shape[1] / 2
    i_res=8
    j_res=8
    window_size=(1000, 900)
    zoom=1.0
    max_time=10000
    b_title = ''
    title_fontsize = 10

    bx = b[..., 0]
    by = b[..., 1]
    bz = b[..., 2]
    mesh = create_mesh(bx, by, bz)
    b_plot = mag_plotter(mesh)
    b_tube, b_bottoms, b_dargs = b_plot.create_mesh(i_siz=i_siz, j_siz=j_siz, i_resolution=i_res, j_resolution=j_res, vmin=vmin, vmax=vmax, max_time=max_time)
    b_tube = b_plot.strl.tube(radius=radius)

    p = pv.Plotter(off_screen=False, window_size=window_size)
    p.add_mesh(b_plot.grid.outline(), color='k')
    p.add_mesh(b_bottoms, cmap='gray', **b_dargs)
    p.add_mesh(b_tube, lighting=False, color='blue')
    p.camera_position = 'yz'
    p.camera.elevation = 20
    p.camera.azimuth = -60
    p.add_title(b_title, font_size=title_fontsize)
    p.camera.zoom(zoom)
    p.show()

import numpy as np
from rtmag.process.paper.hel import FreeEnergy_Helicity, FreeEnergy_Helicity_noA

def hel(b):
    nx, ny, nz = b.shape[:-1]
    n = np.min([nx, ny, nz])
    x = np.arange(nx) / n
    y = np.arange(ny) / n
    z = np.arange(nz) / n
    print("x, y, z", x.shape, y.shape, z.shape)
    np.savez("b.npz", b=b, x=x, y=y, z=z)
    FreeEnergy_Helicity_noA("b.npz")
    FreeEnergy_Helicity("b.npz")

def hel_xyz(b, x, y, z):
    print("x, y, z", x.shape, y.shape, z.shape)
    np.savez("b.npz", b=b, x=x, y=y, z=z)
    FreeEnergy_Helicity_noA("b.npz")
    FreeEnergy_Helicity("b.npz")
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf_file
from rtmag.process.paper.flhcart import BField

def FreeEnergy_Helicity_noA(filepath):
    files = np.load(filepath)
    b = files['b']
    bx = b[..., 0].transpose(2, 1, 0)
    by = b[..., 1].transpose(2, 1, 0)
    bz = b[..., 2].transpose(2, 1, 0)
    nz, ny, nx = bx.shape
    x1 = files['x']
    y1 = files['y']
    z1 = files['z']

    fname = 'b.nc'
    f = netcdf_file(fname, 'w')
    f.createDimension('xdim', nx)
    f.createDimension('ydim', ny)
    f.createDimension('zdim', nz)
    xv = f.createVariable('x', 'f', ('xdim',))
    yv = f.createVariable('y', 'f', ('ydim',))
    zv = f.createVariable('z', 'f', ('zdim',))
    xv[:], yv[:], zv[:] = x1, y1, z1
    bxv = f.createVariable('bx', 'f', ('zdim','ydim','xdim'))
    byv = f.createVariable('by', 'f', ('zdim','ydim','xdim'))
    bzv = f.createVariable('bz', 'f', ('zdim','ydim','xdim'))
    bxv[:,:,:] = bx
    byv[:,:,:] = by
    bzv[:,:,:] = bz
    f.close()

    b = BField('b.nc', clean=False)

    b.computePotentialField()

    x1s, y1s, z1s = np.meshgrid(x1, y1, z1, indexing='ij')
    bxs = b.bx(x1s, y1s, z1s)
    bys = b.by(x1s, y1s, z1s)
    bzs = b.bz(x1s, y1s, z1s)
    bs = np.stack([bxs, bys, bzs], axis=-1)
    bpxs = b.bpx(x1s, y1s, z1s)
    bpys = b.bpy(x1s, y1s, z1s)
    bpzs = b.bpz(x1s, y1s, z1s)
    bps = np.stack([bpxs, bpys, bpzs], axis=-1)
    print("Energy/PotEnergy", (bs**2).sum(-1).sum() / (bps**2).sum(-1).sum())

    b.computeADeVore(potential=True)
    Hr0 = b.relativeHelicity()
    print('Hr from volume integration (DeVore-Coulomb upward gauge) = %g' % Hr0)

    b.matchUniversalGauge()
    b.matchUniversalGauge(potential=True)
    Hr = b.relativeHelicity()
    print('Hr from volume integration (minimal gauge) = %g' % Hr)


def FreeEnergy_Helicity(filepath):
    files = np.load(filepath)
    b = files['b']
    bx = b[..., 0].transpose(2, 1, 0)
    by = b[..., 1].transpose(2, 1, 0)
    bz = b[..., 2].transpose(2, 1, 0)
    nz, ny, nx = bx.shape
    x1 = files['x']
    y1 = files['y']
    z1 = files['z']

    fname = 'b.nc'
    f = netcdf_file(fname, 'w')
    f.createDimension('xdim', nx)
    f.createDimension('ydim', ny)
    f.createDimension('zdim', nz)
    xv = f.createVariable('x', 'f', ('xdim',))
    yv = f.createVariable('y', 'f', ('ydim',))
    zv = f.createVariable('z', 'f', ('zdim',))
    xv[:], yv[:], zv[:] = x1, y1, z1
    bxv = f.createVariable('bx', 'f', ('zdim','ydim','xdim'))
    byv = f.createVariable('by', 'f', ('zdim','ydim','xdim'))
    bzv = f.createVariable('bz', 'f', ('zdim','ydim','xdim'))
    bxv[:,:,:] = bx
    byv[:,:,:] = by
    bzv[:,:,:] = bz
    f.close()

    b = BField('b.nc')

    b.computePotentialField()

    x1s, y1s, z1s = np.meshgrid(x1, y1, z1, indexing='ij')
    bxs = b.bx(x1s, y1s, z1s)
    bys = b.by(x1s, y1s, z1s)
    bzs = b.bz(x1s, y1s, z1s)
    bs = np.stack([bxs, bys, bzs], axis=-1)
    bpxs = b.bpx(x1s, y1s, z1s)
    bpys = b.bpy(x1s, y1s, z1s)
    bpzs = b.bpz(x1s, y1s, z1s)
    bps = np.stack([bpxs, bpys, bpzs], axis=-1)
    print("Energy/PotEnergy", (bs**2).sum(-1).sum() / (bps**2).sum(-1).sum())

    b.computeADeVore(potential=True)
    Hr0 = b.relativeHelicity()
    print('Hr from volume integration (DeVore-Coulomb upward gauge) = %g' % Hr0)

    b.matchUniversalGauge()
    b.matchUniversalGauge(potential=True)
    Hr = b.relativeHelicity()
    print('Hr from volume integration (minimal gauge) = %g' % Hr)
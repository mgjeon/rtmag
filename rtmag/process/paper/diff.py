import numpy as np
from numba import prange, njit


@njit(parallel=True, cache=True)
def Dx(f, h=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        h : float
    
    Output:
        Dx_f : [Nx, Ny, Nz]
    """
    nx, ny, nz = f.shape
    Dx_f = np.zeros((nx, ny, nz))
    
    for i in prange(1, nx-1):
        for j in prange(ny):
            for k in prange(nz):
                    Dx_f[i, j, k] = (f[i+1, j, k] - f[i-1, j, k]) / (2*h)
    
    for j in prange(ny):
        for k in prange(nz):
            Dx_f[0, j, k] = (-3*f[0, j, k] + 4*f[1, j, k] - f[2, j, k]) / (2*h)

    for j in prange(ny):
        for k in prange(nz):
            Dx_f[-1, j, k] = (3*f[-1, j, k] - 4*f[-2, j, k] + f[-3, j, k]) / (2*h)

    return Dx_f


@njit(parallel=True, cache=True)
def Dy(f, h=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        h : float
    
    Output:
        Dy_f : [Nx, Ny, Nz]
    """
    nx, ny, nz = f.shape
    Dy_f = np.zeros((nx, ny, nz))
    
    for i in prange(nx):
        for j in prange(1, ny-1):
            for k in prange(nz):
                    Dy_f[i, j, k] = (f[i, j+1, k] - f[i, j-1, k]) / (2*h)
    
    for i in prange(nx):
        for k in prange(nz):
            Dy_f[i, 0, k] = (-3*f[i, 0, k] + 4*f[i, 1, k] - f[i, 2, k]) / (2*h)

    for i in prange(nx):
        for k in prange(nz):
            Dy_f[i, -1, k] = (3*f[i, -1, k] - 4*f[i, -2, k] + f[i, -3, k]) / (2*h)

    return Dy_f


@njit(parallel=True, cache=True)
def Dz(f, h=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        h : float
    
    Output:
        Dz_f : [Nx, Ny, Nz]
    """
    nx, ny, nz = f.shape
    Dz_f = np.zeros((nx, ny, nz))
    
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(1, nz-1):
                    Dz_f[i, j, k] = (f[i, j, k+1] - f[i, j, k-1]) / (2*h)
    
    for i in prange(nx):
        for j in prange(ny):
            Dz_f[i, j, 0] = (-3*f[i, j, 0] + 4*f[i, j, 1] - f[i, j, 2]) / (2*h)

    for i in prange(nx):
        for j in prange(ny):
            Dz_f[i, j, -1] = (3*f[i, j, -1] - 4*f[i, j, -2] + f[i, j, -3]) / (2*h)

    return Dz_f


@njit(parallel=True, cache=True)
def DDx(f, h=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        h : float
    
    Output:
        DDx_f : [Nx, Ny, Nz]
    """
    nx, ny, nz = f.shape
    DDx_f = np.zeros((nx, ny, nz))
    
    for i in prange(1, nx-1):
        for j in prange(ny):
            for k in prange(nz):
                    DDx_f[i, j, k] = (f[i+1, j, k] - 2*f[i, j, k] + f[i-1, j, k]) / (h**2)
    
    for j in prange(ny):
        for k in prange(nz):
            DDx_f[0, j, k] = (2*f[0, j, k] - 5*f[1, j, k] + 4*f[2, j, k] - f[3, j, k]) / (h**2)
    
    for j in prange(ny):
        for k in prange(nz):
            DDx_f[-1, j, k] = (2*f[-1, j, k] - 5*f[-2, j, k] + 4*f[-3, j, k] - f[-4, j, k]) / (h**2)
    
    return DDx_f


@njit(parallel=True, cache=True)
def DDy(f, h=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        h : float
    
    Output:
        DDy_f : [Nx, Ny, Nz]
    """
    nx, ny, nz = f.shape
    DDy_f = np.zeros((nx, ny, nz))
    
    for i in prange(nx):
        for j in prange(1, ny-1):
            for k in prange(nz):
                    DDy_f[i, j, k] = (f[i, j+1, k] - 2*f[i, j, k] + f[i, j-1, k]) / (h**2)
    
    for i in prange(nx):
        for k in prange(nz):
            DDy_f[i, 0, k] = (2*f[i, 0, k] - 5*f[i, 1, k] + 4*f[i, 2, k] - f[i, 3, k]) / (h**2)
    
    for i in prange(nx):
        for k in prange(nz):
            DDy_f[i, -1, k] = (2*f[i, -1, k] - 5*f[i, -2, k] + 4*f[i, -3, k] - f[i, -4, k]) / (h**2)
    
    return DDy_f


@njit(parallel=True, cache=True)
def DDz(f, h=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        h : float
    
    Output:
        DDz_f : [Nx, Ny, Nz]
    """
    nx, ny, nz = f.shape
    DDz_f = np.zeros((nx, ny, nz))
    
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(1, nz-1):
                    DDz_f[i, j, k] = (f[i, j, k+1] - 2*f[i, j, k] + f[i, j, k-1]) / (h**2)
    
    for i in prange(nx):
        for j in prange(ny):
            DDz_f[i, j, 0] = (2*f[i, j, 0] - 5*f[i, j, 1] + 4*f[i, j, 2] - f[i, j, 3]) / (h**2)
    
    for i in prange(nx):
        for j in prange(ny):
            DDz_f[i, j, -1] = (2*f[i, j, -1] - 5*f[i, j, -2] + 4*f[i, j, -3] - f[i, j, -4]) / (h**2)
    
    return DDz_f


@njit(parallel=True, cache=True)
def laplacian(f, dx=1.0, dy=1.0, dz=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        dx : float
        dy : float
        dz : float

    Output:
        laplacian_f : [Nx, Ny, Nz]
    """
    laplacian_f = np.zeros(f.shape)
    laplacian_f = DDx(f, dx) + DDy(f, dy) + DDz(f, dz)
    return laplacian_f


@njit(parallel=True, cache=True)
def gradient(f, dx=1.0, dy=1.0, dz=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        dx : float
        dy : float
        dz : float

    Output:
        gradient_f : [Nx, Ny, Nz, 3]
    """

    gradient_f = np.zeros((f.shape[0], f.shape[1], f.shape[2], 3))

    gradient_f[..., 0] = Dx(f, dx)
    gradient_f[..., 1] = Dy(f, dy)
    gradient_f[..., 2] = Dz(f, dz)
    return gradient_f


@njit(parallel=True, cache=True)
def curl(F, dx=1.0, dy=1.0, dz=1.0):
    """
    Input:
        F : [Nx, Ny, Nz, 3]
        dx : float
        dy : float
        dz : float

    Output:
        curl_F : [Nx, Ny, Nz, 3]
    """

    Fx = F[..., 0]
    Fy = F[..., 1]
    Fz = F[..., 2]

    curl_F = np.zeros(F.shape)

    curl_F[..., 0] = Dy(Fz, dy) - Dz(Fy, dz)
    curl_F[..., 1] = Dz(Fx, dz) - Dx(Fz, dx)
    curl_F[..., 2] = Dx(Fy, dx) - Dy(Fx, dy)    
    return curl_F


@njit(parallel=True, cache=True)
def divergence(F, dx=1.0, dy=1.0, dz=1.0):
    """
    Input:
        F : [Nx, Ny, Nz, 3]
        dx : float
        dy : float
        dz : float

    Output:
        divergence_F : [Nx, Ny, Nz]
    """

    Fx = F[..., 0]
    Fy = F[..., 1]
    Fz = F[..., 2]

    divergence_F = np.zeros(F.shape[:-1])

    divergence_F = Dx(Fx, dx) + Dy(Fy, dy) + Dz(Fz, dz)
    return divergence_F
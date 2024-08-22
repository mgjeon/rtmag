import numpy as np

#-----------------------------------------------------------------------------------------

def Dx(f, h):
    Dx_f = np.zeros_like(f)
    Dx_f[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * h)
    Dx_f[0, :, :] = (-3 * f[0, :, :] + 4 * f[1, :, :] - f[2, :, :]) / (2 * h)
    Dx_f[-1, :, :] = (3 * f[-1, :, :] - 4 * f[-2, :, :] + f[-3, :, :]) / (2 * h)
    return Dx_f


def Dy(f, h):
    Dy_f = np.zeros_like(f)
    Dy_f[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * h)
    Dy_f[:, 0, :] = (-3 * f[:, 0, :] + 4 * f[:, 1, :] - f[:, 2, :]) / (2 * h)
    Dy_f[:, -1, :] = (3 * f[:, -1, :] - 4 * f[:, -2, :] + f[:, -3, :]) / (2 * h)
    return Dy_f


def Dz(f, h):
    Dz_f = np.zeros_like(f)
    Dz_f[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * h)
    Dz_f[:, :, 0] = (-3 * f[:, :, 0] + 4 * f[:, :, 1] - f[:, :, 2]) / (2 * h)
    Dz_f[:, :, -1] = (3 * f[:, :, -1] - 4 * f[:, :, -2] + f[:, :, -3]) / (2 * h)
    return Dz_f


def DDx(f, h):
    DDx_f = np.zeros_like(f)
    DDx_f[1:-1, :, :] = (f[2:, :, :] - 2 * f[1:-1, :, :] + f[:-2, :, :]) / (h**2)
    DDx_f[0, :, :] = (2 * f[0, :, :] - 5 * f[1, :, :] + 4 * f[2, :, :] - f[3, :, :]) / (
        h**2
    )
    DDx_f[-1, :, :] = (
        2 * f[-1, :, :] - 5 * f[-2, :, :] + 4 * f[-3, :, :] - f[-4, :, :]
    ) / (h**2)
    return DDx_f


def DDy(f, h):
    DDy_f = np.zeros_like(f)
    DDy_f[:, 1:-1, :] = (f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]) / (h**2)
    DDy_f[:, 0, :] = (2 * f[:, 0, :] - 5 * f[:, 1, :] + 4 * f[:, 2, :] - f[:, 3, :]) / (
        h**2
    )
    DDy_f[:, -1, :] = (
        2 * f[:, -1, :] - 5 * f[:, -2, :] + 4 * f[:, -3, :] - f[:, -4, :]
    ) / (h**2)
    return DDy_f


def DDz(f, h):
    DDz_f = np.zeros_like(f)
    DDz_f[:, :, 1:-1] = (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / (h**2)
    DDz_f[:, :, 0] = (2 * f[:, :, 0] - 5 * f[:, :, 1] + 4 * f[:, :, 2] - f[:, :, 3]) / (
        h**2
    )
    DDz_f[:, :, -1] = (
        2 * f[:, :, -1] - 5 * f[:, :, -2] + 4 * f[:, :, -3] - f[:, :, -4]
    ) / (h**2)
    return DDz_f

#-----------------------------------------------------------------------------------------

def laplacian(f, dx, dy, dz):
    return DDx(f, dx) + DDy(f, dy) + DDz(f, dz)

#-----------------------------------------------------------------------------------------

def gradient(f, dx, dy, dz):
    gradient_xcomp = Dx(f, dx)
    gradient_ycomp = Dy(f, dy)
    gradient_zcomp = Dz(f, dz)

    gradients = np.stack([gradient_xcomp, gradient_ycomp, gradient_zcomp], axis=-1)
    return gradients


def gradient_np(f, dx, dy, dz):
    gradient_xcomp, gradient_ycomp, gradient_zcomp = np.gradient(f, dx, dy, dz, axis=(0, 1, 2), edge_order=2)
    
    gradients = np.stack([gradient_xcomp, gradient_ycomp, gradient_zcomp], axis=-1)
    return gradients

#-----------------------------------------------------------------------------------------

def curl(F, dx, dy, dz):
    """
    F : [Nx, Ny, Nz, 3]
    """

    Fx = F[..., 0]
    Fy = F[..., 1]
    Fz = F[..., 2]

    curl_xcomp = Dy(Fz, dy) - Dz(Fy, dz)
    curl_ycomp = Dz(Fx, dz) - Dx(Fz, dx)
    curl_zcomp = Dx(Fy, dx) - Dy(Fx, dy)

    curls = np.stack([curl_xcomp, curl_ycomp, curl_zcomp], axis=-1)
    return curls


def curl_np(F, dx, dy, dz):
    """
    F : [Nx, Ny, Nz, 3]
    """

    Fx = F[..., 0]
    Fy = F[..., 1]
    Fz = F[..., 2]

    Dx_Fx, Dy_Fx, Dz_Fx = np.gradient(Fx, dx, dy, dz, axis=(0, 1, 2), edge_order=2)
    Dx_Fy, Dy_Fy, Dz_Fy = np.gradient(Fy, dx, dy, dz, axis=(0, 1, 2), edge_order=2)
    Dx_Fz, Dy_Fz, Dz_Fz = np.gradient(Fz, dx, dy, dz, axis=(0, 1, 2), edge_order=2)

    curl_xcomp = Dy_Fz - Dz_Fy
    curl_ycomp = Dz_Fx - Dx_Fz
    curl_zcomp = Dx_Fy - Dy_Fx

    curls = np.stack([curl_xcomp, curl_ycomp, curl_zcomp], axis=-1)
    return curls


def curl_np2(F):
    """
    F : [Nx, Ny, Nz, 3]
    """

    Fx = F[..., 0]
    Fy = F[..., 1]
    Fz = F[..., 2]

    Dx_Fx, Dy_Fx, Dz_Fx = np.gradient(Fx, axis=(0, 1, 2), edge_order=2)
    Dx_Fy, Dy_Fy, Dz_Fy = np.gradient(Fy, axis=(0, 1, 2), edge_order=2)
    Dx_Fz, Dy_Fz, Dz_Fz = np.gradient(Fz, axis=(0, 1, 2), edge_order=2)

    curl_xcomp = Dy_Fz - Dz_Fy
    curl_ycomp = Dz_Fx - Dx_Fz
    curl_zcomp = Dx_Fy - Dy_Fx

    curls = np.stack([curl_xcomp, curl_ycomp, curl_zcomp], axis=-1)
    return curls

#-----------------------------------------------------------------------------------------

def divergence(F, dx, dy, dz):
    """
    F : [Nx, Ny, Nz, 3]
    """

    Fx = F[..., 0]
    Fy = F[..., 1]
    Fz = F[..., 2]

    return Dx(Fx, dx) + Dy(Fy, dy) + Dz(Fz, dz)


def divergence_np(F, dx, dy, dz):
    """
    F : [Nx, Ny, Nz, 3]
    """

    Fx = F[..., 0]
    Fy = F[..., 1]
    Fz = F[..., 2]

    Dx_Fx, Dy_Fx, Dz_Fx = np.gradient(Fx, dx, dy, dz, axis=(0, 1, 2), edge_order=2)
    Dx_Fy, Dy_Fy, Dz_Fy = np.gradient(Fy, dx, dy, dz, axis=(0, 1, 2), edge_order=2)
    Dx_Fz, Dy_Fz, Dz_Fz = np.gradient(Fz, dx, dy, dz, axis=(0, 1, 2), edge_order=2)

    return Dx_Fx + Dy_Fy + Dz_Fz


def divergence_np2(F):
    """
    F : [Nx, Ny, Nz, 3]
    """

    Fx = F[..., 0]
    Fy = F[..., 1]
    Fz = F[..., 2]

    Dx_Fx, Dy_Fx, Dz_Fx = np.gradient(Fx, axis=(0, 1, 2), edge_order=2)
    Dx_Fy, Dy_Fy, Dz_Fy = np.gradient(Fy, axis=(0, 1, 2), edge_order=2)
    Dx_Fz, Dy_Fz, Dz_Fz = np.gradient(Fz, axis=(0, 1, 2), edge_order=2)

    return Dx_Fx + Dy_Fy + Dz_Fz

#-----------------------------------------------------------------------------------------

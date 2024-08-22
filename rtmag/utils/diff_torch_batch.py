import torch

#-----------------------------------------------------------------------------------------

def Dx(f, h):
    """
    f : [batch_size, Nx, Ny, Nz]
    h : [batch_size]
    """
    Dx_f = torch.zeros_like(f)
    h_inner = h[:, None, None, None]
    h_boundary = h[:, None, None]
    Dx_f[:, 1:-1, :, :] = (f[:, 2:, :, :] - f[:, :-2, :, :]) / (2 * h_inner)
    Dx_f[:, 0, :, :] = (-3 * f[:, 0, :, :] + 4 * f[:, 1, :, :] - f[:, 2, :, :]) / (2 * h_boundary)
    Dx_f[:, -1, :, :] = (3 * f[:, -1, :, :] - 4 * f[:, -2, :, :] + f[:, -3, :, :]) / (2 * h_boundary)
    return Dx_f


def Dy(f, h):
    """
    f : [batch_size, Nx, Ny, Nz]
    h : [batch_size]
    """
    Dy_f = torch.zeros_like(f)
    h_inner = h[:, None, None, None]
    h_boundary = h[:, None, None]
    Dy_f[:, :, 1:-1, :] = (f[:, :, 2:, :] - f[:, :, :-2, :]) / (2 * h_inner)
    Dy_f[:, :, 0, :] = (-3 * f[:, :, 0, :] + 4 * f[:, :, 1, :] - f[:, :, 2, :]) / (2 * h_boundary)
    Dy_f[:, :, -1, :] = (3 * f[:, :, -1, :] - 4 * f[:, :, -2, :] + f[:, :, -3, :]) / (2 * h_boundary)
    return Dy_f


def Dz(f, h):
    """
    f : [batch_size, Nx, Ny, Nz]
    h : [batch_size]
    """
    Dz_f = torch.zeros_like(f)
    h_inner = h[:, None, None, None]
    h_boundary = h[:, None, None]
    Dz_f[:, :, :, 1:-1] = (f[:, :, :, 2:] - f[:, :, :, :-2]) / (2 * h_inner)
    Dz_f[:, :, :, 0] = (-3 * f[:, :, :, 0] + 4 * f[:, :, :, 1] - f[:, :, :, 2]) / (2 * h_boundary)
    Dz_f[:, :, :, -1] = (3 * f[:, :, :, -1] - 4 * f[:, :, :, -2] + f[:, :, :, -3]) / (2 * h_boundary)
    return Dz_f


def DDx(f, h):
    """
    f : [batch_size, Nx, Ny, Nz]
    h : [batch_size]
    """
    DDx_f = torch.zeros_like(f)
    h_inner = h[:, None, None, None]
    h_boundary = h[:, None, None]
    DDx_f[:, 1:-1, :, :] = (f[:, 2:, :, :] - 2 * f[:, 1:-1, :, :] + f[:, :-2, :, :]) / (h_inner**2)
    DDx_f[:, 0, :, :] = (2 * f[:, 0, :, :] - 5 * f[:, 1, :, :] + 4 * f[:, 2, :, :] - f[:, 3, :, :]) / (h_boundary**2)
    DDx_f[:, -1, :, :] = (2 * f[:, -1, :, :] - 5 * f[:, -2, :, :] + 4 * f[:, -3, :, :] - f[:, -4, :, :]) / (h_boundary**2)
    return DDx_f


def DDy(f, h):
    """
    f : [batch_size, Nx, Ny, Nz]
    h : [batch_size]
    """
    DDy_f = torch.zeros_like(f)
    h_inner = h[:, None, None, None]
    h_boundary = h[:, None, None]
    DDy_f[:, :, 1:-1, :] = (f[:, :, 2:, :] - 2 * f[:, :, 1:-1, :] + f[:, :, :-2, :]) / (h_inner**2)
    DDy_f[:, :, 0, :] = (2 * f[:, :, 0, :] - 5 * f[:, :, 1, :] + 4 * f[:, :, 2, :] - f[:, :, 3, :]) / (h_boundary**2)
    DDy_f[:, :, -1, :] = (2 * f[:, :, -1, :] - 5 * f[:, :, -2, :] + 4 * f[:, :, -3, :] - f[:, :, -4, :]) / (h_boundary**2)
    return DDy_f


def DDz(f, h):
    """
    f : [batch_size, Nx, Ny, Nz]
    h : [batch_size]
    """
    DDz_f = torch.zeros_like(f)
    h_inner = h[:, None, None, None]
    h_boundary = h[:, None, None]
    DDz_f[:, :, :, 1:-1] = (f[:, :, :, 2:] - 2 * f[:, :, :, 1:-1] + f[:, :, :, :-2]) / (h_inner**2)
    DDz_f[:, :, :, 0] = (2 * f[:, :, :, 0] - 5 * f[:, :, :, 1] + 4 * f[:, :, :, 2] - f[:, :, :, 3]) / (h_boundary**2)
    DDz_f[:, :, :, -1] = (2 * f[:, :, :, -1] - 5 * f[:, :, :, -2] + 4 * f[:, :, :, -3] - f[:, :, :, -4]) / (h_boundary**2)
    return DDz_f

#-----------------------------------------------------------------------------------------

def gradient(f, dx, dy, dz):
    gradient_xcomp = Dx(f, dx)
    gradient_ycomp = Dy(f, dy)
    gradient_zcomp = Dz(f, dz)
    return gradient_xcomp, gradient_ycomp, gradient_zcomp


def curl(Fx, Fy, Fz, dx, dy, dz):
    curl_xcomp = Dy(Fz, dy) - Dz(Fy, dz)
    curl_ycomp = Dz(Fx, dz) - Dx(Fz, dx)
    curl_zcomp = Dx(Fy, dx) - Dy(Fx, dy)

    return curl_xcomp, curl_ycomp, curl_zcomp


def divergence(Fx, Fy, Fz, dx, dy, dz):
    return Dx(Fx, dx) + Dy(Fy, dy) + Dz(Fz, dz)


def laplacian(f, dx, dy, dz):
    return DDx(f, dx) + DDy(f, dy) + DDz(f, dz)
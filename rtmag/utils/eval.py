import numpy as np
from rtmag.utils.diff import curl_np2 as curl
from rtmag.utils.diff import divergence_np2 as divergence


def evaluate(b, B):
    # b : model solution
    # B : reference magnetic field

    result = {}
    result["C_vec"] = C_vec(b, B)
    result["C_cs"] = C_cs(b, B)
    result["E_n'"] = En_prime(b, B)
    result["E_m'"] = Em_prime(b, B)
    result['eps'] = eps(b, B)
    result['CW_sin'], result['L_f'], result['L_d'] = metrics_j(b)
    result['l2_err'] = l2_error(b, B)

    for key, value in result.items():
        print(f"{key:<10}: {value:.4f}")

    return result

#-----------------------------------------------------------------------------------------

def C_vec(b, B):
    return dot_product(B, b).sum() / np.sqrt((vector_norm(B)**2).sum() * (vector_norm(b)**2).sum())


def C_cs(b, B):
    nu = dot_product(B, b)
    de = vector_norm(B) * vector_norm(b)
    M = np.sum([de!=0.])
    return (1 / M) * np.divide(nu, de, where=de!=0.).sum()


def En_prime(b, B):
    return 1 - (vector_norm(b - B).sum() / vector_norm(B).sum())


def Em_prime(b, B):
    nu = vector_norm(b - B)
    de = vector_norm(B)
    M = np.sum([de!=0.])
    return 1 - ((1 / M) * np.sum(np.divide(nu, de, where=de!=0.)))


def eps(b, B):
    return (vector_norm(b) ** 2).sum() / (vector_norm(B) ** 2).sum()


def l2_error(u_pred, u_true):
        error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        return error


def metrics_j(b):
    j = curl(b)

    jxb = np.cross(j, b, axis=-1)
    
    nu = vector_norm(jxb)
    de = vector_norm(b)
    CW_sin = (np.divide(nu, de, where=de!=0.).sum() / vector_norm(j).sum())

    nu = vector_norm(jxb) ** 2
    de = vector_norm(b) ** 2
    L_f = np.divide(nu, de, where=de!=0.).mean()
    L_d = (divergence(b)**2).mean()

    return CW_sin, L_f, L_d

#-----------------------------------------------------------------------------------------

def vector_norm(F):
    """
    F : [Nx, Ny, Nz, 3]
    """
    return (F**2).sum(-1)**0.5


def vector_norm_np(F):
    """
    F : [Nx, Ny, Nz, 3]
    """
    return np.linalg.norm(F, axis=-1)

#-----------------------------------------------------------------------------------------

def energy(B, dV):
    """
    B  : [Nx, Ny, Nz, 3]
    dV : cm^3
    """
    return (vector_norm(B)**2).sum() * (dV/(8*np.pi))


def energy_np(B, dV):
    """
    B  : [Nx, Ny, Nz, 3]
    dV : cm^3
    """
    return (vector_norm_np(B)**2).sum() * (dV/(8*np.pi))


def energy_np2(B):
    """
    B  : [Nx, Ny, Nz, 3]
    """
    return (B ** 2).sum(-1) / (8 * np.pi)

#-----------------------------------------------------------------------------------------

def dot_product(a, b):
    """
    a : [Nx, Ny, Nz, 3]
    b : [Nx, Ny, Nz, 3]
    """
    return (a * b).sum(-1)
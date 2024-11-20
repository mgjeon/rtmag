import numpy as np
from numba import prange, njit
from rtmag.process.paper.diff import curl, divergence


#--------------------------------------------------
def evaluate_sharp(b, dV, isprint=False):
    # b : model solution  [Gauss]
    # dV: volume element  [cm^3]

    result = {}
    result['pred_E_1e33'] = energy(b, dV) / 1e33
    result['dV_1e23'] = dV / 1e23
    result['pred_E_unit'] = energy_unit(b)

    if isprint:
        for key, value in result.items():
            print(f"{key:<10}: {value:.2f}")

    return result


#--------------------------------------------------
def evaluate_energy(b, B, Bp, dV, isprint=False):
    # b : model solution
    # B : reference magnetic field
    # Bp: potential magnetic field

    result = {}
    result['pred_E_1e33'] = energy(b, dV) / 1e33
    result['ref_E_1e33'] = energy(B, dV) / 1e33
    result['pot_E_1e33'] = energy(Bp, dV) / 1e33
    result['dV_1e23'] = dV / 1e23

    result['pred_E_unit'] = energy_unit(b)
    result['ref_E_unit'] = energy_unit(B)
    result['pot_E_unit'] = energy_unit(Bp)

    result["C_vec"] = C_vec(b, B)
    result["C_cs"] = C_cs(b, B)
    result["E_n_prime"] = En_prime(b, B)
    result["E_m_prime"] = Em_prime(b, B)
    result['eps'] = eps(b, B)
    result['rel_l2_err'] = rel_l2_error(b, B)

    if isprint:
        for key, value in result.items():
            print(f"{key:<10}: {value:.2f}")

    return result


def evaluate_j(b, ret=True):
    """
    Input:
        b: model solution           [Nx, Ny, Nz, 3]
        B: reference magnetic field [Nx, Ny, Nz, 3]

    Output:
        result: dict
    """

    result = {}
    result['CW_sin'], result['L_f'], result['L_d'] = metrics_j(b)

    if ret:
        for key, value in result.items():
            print(f"{key:<10}: {value:.2f}")

    return result


#--------------------------------------------------

@njit(parallel=True, cache=False)
def current_density(b, dx, dy, dz):
    """
    Input:
        b  : [Nx, Ny, Nz, 3]     [Gauss]
        dx : float               [cm]
        dy : float               [cm]         
        dz : float               [cm] 
    
    Output:
        current_density : [Nx, Ny, Nz, 3]  [mA/m^2]
    """
    # [Gauss/cm] -> [mA/m^2]
    curlb_to_j = 1e8/(4*np.pi)

    cur_den = curl(b, dx, dy, dz)*curlb_to_j
    
    return cur_den


@njit(parallel=True, cache=True)
def dot_product(a, b):
    """
    Input:
        a : [Nx, Ny, Nz, 3]
        b : [Nx, Ny, Nz, 3]
    
    Output:
        c : [Nx, Ny, Nz]
    """
    c = np.zeros(a.shape[:-1])

    I, J, K, L = a.shape
    for i in prange(I):
        for j in prange(J):
            for k in prange(K):
                for l in prange(L):
                    c[i, j, k] += a[i, j, k, l]*b[i, j, k, l]
    
    return c


@njit(parallel=True, cache=True)
def cross_product(a, b):
    """"
    Input:
        a : [Nx, Ny, Nz, 3]
        b : [Nx, Ny, Nz, 3]

    Output:
        c : [Nx, Ny, Nz, 3]
    """
    # cross product
    I, J, K, L = b.shape
    res = np.zeros((I, J, K, L))

    for i in prange(I):
        for j in prange(J):
            for k in prange(K):
                res[i, j, k, 0] = a[i, j, k, 1]*b[i, j, k, 2] - a[i, j, k, 2]*b[i, j, k, 1]
                res[i, j, k, 1] = a[i, j, k, 2]*b[i, j, k, 0] - a[i, j, k, 0]*b[i, j, k, 2]
                res[i, j, k, 2] = a[i, j, k, 0]*b[i, j, k, 1] - a[i, j, k, 1]*b[i, j, k, 0]

    return res


@njit(parallel=True, cache=True)
def vector_norm(a):
    """
    Input:
        a : [Nx, Ny, Nz, 3]

    Output:
        c : [Nx, Ny, Nz]
    """
    c = np.zeros(a.shape[:-1])
    c = np.sqrt(dot_product(a, a))
    return c


@njit(parallel=True, cache=True)
def energy(B, dV):
    """
    Input:
        B  : [Nx, Ny, Nz, 3]   [Gauss]
        dV : float             [cm^3]

    Output:
        energy: float          [erg]
    """

    ene = (vector_norm(B)**2).sum() * (dV/(8*np.pi))
    return ene


@njit(parallel=True, cache=True)
def energy_unit(B):
    """
    Input:
        B  : [Nx, Ny, Nz, 3]   [Gauss]

    Output:
        energy: float          ... requires * (dV/(8*np.pi))
    """

    ene = (vector_norm(B)**2).sum()
    return ene

def rel_l2_error(u_pred, u_true):
    """
    relative l2 error
    """
    error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
    return error

#--------------------------------------------------
@njit(parallel=True, cache=True)
def metrics_j_exclude(b):
    j = curl(b)

    jxb = cross_product(j, b)
    
    nu = vector_norm(jxb)
    de = vector_norm(b)

    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                if de[x, y, z] != 0.:
                    res[x, y, z] = nu[x, y, z] / de[x, y, z]

    CW_sin = res.sum() / vector_norm(j).sum()

    nu = vector_norm(jxb) ** 2
    de = vector_norm(b) ** 2

    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                if de[x, y, z] != 0.:
                    res[x, y, z] = nu[x, y, z] / de[x, y, z]

    L_f = res.mean()
    L_d = (divergence(b)**2).mean()

    return CW_sin, L_f, L_d


@njit(parallel=True, cache=True)
def metrics_j(b):
    j = curl(b)

    jxb = cross_product(j, b)
    
    nu = vector_norm(jxb)
    de = vector_norm(b)

    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                res[x, y, z] = nu[x, y, z] / (de[x, y, z] + 1e-7)

    CW_sin = res.sum() / vector_norm(j).sum()

    nu = vector_norm(jxb) ** 2
    de = vector_norm(b) ** 2

    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                res[x, y, z] = nu[x, y, z] / (de[x, y, z] + 1e-7)

    L_f = res.mean()
    L_d = (divergence(b)**2).mean()

    nu = divergence(b)
    de = vector_norm(b)
    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                res[x, y, z] = np.abs(nu[x, y, z]) / (de[x, y, z] + 1e-7)

    L_d_n = res.mean()
    print("nx, ny, nz: ", nx, ny, nz)
    print("M:", len(res.flatten()))
    print("M:", nx*ny*nz)

    print("CW_sin: ", CW_sin)
    print("sigma_J*100 ", CW_sin*100)
    print("L_f: ", L_f)
    print("L_d: ", L_d)
    print("L_d_n: ", L_d_n)

    return CW_sin, CW_sin*100, L_f, L_d, L_d_n


@njit(parallel=True, cache=True)
def metrics_j_two(b):
    j = curl(b)
    jxb = cross_product(j, b)
    nu = vector_norm(jxb)
    de = vector_norm(b)
    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                res[x, y, z] = nu[x, y, z] / (de[x, y, z] + 1e-7)
    CW_sin = res.sum() / vector_norm(j).sum()

    nu = divergence(b)
    de = vector_norm(b)
    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                res[x, y, z] = np.abs(nu[x, y, z]) / (de[x, y, z] + 1e-7)
    L_d_n = res.mean()

    dx = 1  # pixel
    dy = 1  # pixel
    dz = 1  # pixel
    DeltaV = dx * dy * dz
    SurfaceA = (dx * dy + dy * dz + dz * dx) * 2

    nu = divergence(b)
    de = vector_norm(b)
    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                res[x, y, z] = np.abs(nu[x, y, z]) / (de[x, y, z] + 1e-7)
    fi = res.mean() * (DeltaV / SurfaceA)

    return CW_sin, L_d_n, fi


@njit(parallel=True, cache=True)
def metrics_j_two_dxdydz(b, dx, dy, dz):
    j = curl(b, dx, dy, dz)
    jxb = cross_product(j, b)
    nu = vector_norm(jxb)
    de = vector_norm(b)
    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                res[x, y, z] = nu[x, y, z] / (de[x, y, z] + 1e-7)
    CW_sin = res.sum() / vector_norm(j).sum()

    nu = divergence(b, dx, dy, dz)
    de = vector_norm(b)
    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                res[x, y, z] = np.abs(nu[x, y, z]) / (de[x, y, z] + 1e-7)
    L_d_n = res.mean()

    DeltaV = dx * dy * dz
    SurfaceA = (dx * dy + dy * dz + dz * dx) * 2

    nu = divergence(b, dx, dy, dz)
    de = vector_norm(b)
    nx, ny, nz = nu.shape
    res = np.zeros((nx, ny, nz))
    for x in prange(nx):
        for y in prange(ny):
            for z in prange(nz):
                res[x, y, z] = np.abs(nu[x, y, z]) / (de[x, y, z] + 1e-7)
    fi = res.mean() * (DeltaV / SurfaceA)

    return CW_sin, L_d_n, fi


@njit(cache=True)
def lorentz_force(b_field, j_field=None):
    j_field = j_field if j_field is not None else curl(b_field)
    l = cross_product(j_field, b_field)
    return l


@njit(parallel=True, cache=True)
def angle(b_field, j_field):
    norm = vector_norm(b_field) * vector_norm(j_field) + 1e-7
    j_cross_b = cross_product(j_field, b_field)
    sig = vector_norm(j_cross_b) / norm
    res = np.arcsin(np.clip(sig, -1. + 1e-7, 1. - 1e-7)) * (180 / np.pi)
    return res


@njit(parallel=True, cache=True)
def normalized_divergence(b_field):
    nu = np.abs(divergence(b_field))
    de = vector_norm(b_field) 

    res = nu / (de + 1e-7)
    return res


@njit(parallel=True, cache=True)
def weighted_theta(b, j=None):
    j = j if j is not None else curl(b)
    sigma = vector_norm(lorentz_force(b, j)) / vector_norm(b) / vector_norm(j)
    w_sigma = np.average((sigma), weights=vector_norm(j))
    theta_j = np.arcsin(w_sigma) * (180 / np.pi)
    return theta_j


#--------------------------------------------------
def evaluate(b, B, ret=True):
    """
    Input:
        b: model solution           [Nx, Ny, Nz, 3]
        B: reference magnetic field [Nx, Ny, Nz, 3]

    Output:
        result: dict
    """

    result = {}
    result["C_vec"] = C_vec(b, B)
    result["C_cs"] = C_cs(b, B)
    result["E_n'"] = En_prime(b, B)
    result["E_m'"] = Em_prime(b, B)
    result['eps'] = eps(b, B)

    result['sigmaJ_b'], result['div_b'], result['fi_b'] = metrics_j_two(b)
    result['sigmaJ_B'], result['div_B'], result['fi_B'] = metrics_j_two(B)

    result["sig100_b"] = result['sigmaJ_b']*100
    result["div100_b"] = result['div_b']*100

    result["sig100_B"] = result['sigmaJ_B']*100
    result["div100_B"] = result['div_B']*100 

    if ret:
        for key, value in result.items():
            print(f"{key:<10}: {value:.2f}")

        print()
        for key, value in result.items():
            print(f"{key:<10}: {value:.2g}")

    return result


def evaluate_dxdydz(b, B, dx, dy, dz, ret=True):
    """
    Input:
        b: model solution           [Nx, Ny, Nz, 3]
        B: reference magnetic field [Nx, Ny, Nz, 3]
        dx, dy, dz : Mm

    Output:
        result: dict
    """

    result = {}
    result["C_vec"] = C_vec(b, B)
    result["C_cs"] = C_cs(b, B)
    result["E_n'"] = En_prime(b, B)
    result["E_m'"] = Em_prime(b, B)
    result['eps'] = eps(b, B)

    result['sigmaJ_b'], result['div_b'], result['fi_b'] = metrics_j_two_dxdydz(b, dx, dy, dz)
    result['sigmaJ_B'], result['div_B'], result['fi_B'] = metrics_j_two_dxdydz(B, dx, dy, dz)

    result["sig100_b"] = result['sigmaJ_b']*100
    result["div100_b"] = result['div_b']*100

    result["sig100_B"] = result['sigmaJ_B']*100
    result["div100_B"] = result['div_B']*100 

    if ret:
        for key, value in result.items():
            print(f"{key:<10}: {value:.2f}")

        print()
        for key, value in result.items():
            print(f"{key:<10}: {value:.2g}")

    return result




def evaluate_exclude(b, B):
    """
    Input:
        b: model solution           [Nx, Ny, Nz, 3]
        B: reference magnetic field [Nx, Ny, Nz, 3]

    Output:
        result: dict
    """

    result = {}
    result["C_vec"] = C_vec(b, B)
    result["C_cs"] = C_cs_exclude(b, B)
    result["E_n'"] = En_prime(b, B)
    result["E_m'"] = Em_prime_exclude(b, B)
    result['eps'] = eps(b, B)

    for key, value in result.items():
        print(f"{key:<10}: {value:.2f}")

    return result


@njit(parallel=True, cache=True)
def C_vec(b, B):
    """
    Input:
        b: [Nx, Ny, Nz, 3]
        B: [Nx, Ny, Nz, 3]

    Output:
        res: float
    """

    nu = dot_product(B, b).sum()
    de = np.sqrt( (vector_norm(B)**2).sum() * (vector_norm(b)**2).sum() )

    res = nu / de
    return res


@njit(parallel=True, cache=True)
def C_cs_exclude(b, B):
    """
    Input:
        b: [Nx, Ny, Nz, 3]
        B: [Nx, Ny, Nz, 3]

    Output:
        res: float
    """

    nu = dot_product(B, b)
    de = vector_norm(B) * vector_norm(b)

    nx, ny, nz = nu.shape

    M = 0
    res = np.zeros((nx, ny, nz))

    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                if de[i, j, k] != 0.:
                    res[i, j, k] = nu[i, j, k] / de[i, j, k]
                    M += 1

    res = np.sum(res)
    res = res / M
    return res


@njit(parallel=True, cache=True)
def C_cs(b, B):
    """
    Input:
        b: [Nx, Ny, Nz, 3]
        B: [Nx, Ny, Nz, 3]

    Output:
        res: float
    """

    nu = dot_product(B, b)
    de = vector_norm(B) * vector_norm(b)

    nx, ny, nz = nu.shape

    M = 0
    res = np.zeros((nx, ny, nz))

    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                res[i, j, k] = nu[i, j, k] / (de[i, j, k] + 1e-7)
                M += 1

    res = np.sum(res)
    res = res / M
    return res


@njit(parallel=True, cache=True)
def En_prime(b, B):
    """
    Input:
        b: [Nx, Ny, Nz, 3]
        B: [Nx, Ny, Nz, 3]

    Output:
        res: float
    """

    nu = vector_norm(b - B).sum()
    de = vector_norm(B).sum()

    res = nu / de
    res = 1 - res
    return res


@njit(parallel=True, cache=True)
def Em_prime_exclude(b, B):
    """
    Input:
        b: [Nx, Ny, Nz, 3]
        B: [Nx, Ny, Nz, 3]

    Output:
        res: float
    """

    nu = vector_norm(b - B)
    de = vector_norm(B)

    nx, ny, nz = nu.shape

    M = 0
    res = np.zeros((nx, ny, nz))

    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                if de[i, j, k] != 0.:
                    res[i, j, k] = nu[i, j, k] / de[i, j, k]
                    M += 1

    res = np.sum(res)
    res = res / M
    res = 1 - res
    return res


@njit(parallel=True, cache=True)
def Em_prime(b, B):
    """
    Input:
        b: [Nx, Ny, Nz, 3]
        B: [Nx, Ny, Nz, 3]

    Output:
        res: float
    """

    nu = vector_norm(b - B)
    de = vector_norm(B)

    nx, ny, nz = nu.shape

    M = 0
    res = np.zeros((nx, ny, nz))

    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                res[i, j, k] = nu[i, j, k] / (de[i, j, k] + 1e-7)
                M += 1

    res = np.sum(res)
    res = res / M
    res = 1 - res
    return res


@njit(parallel=True, cache=True)
def eps(b, B):
    """
    Input:
        b: [Nx, Ny, Nz, 3]
        B: [Nx, Ny, Nz, 3]

    Output:
        res: float
    """

    nu = (vector_norm(b) ** 2).sum()
    de = (vector_norm(B) ** 2).sum()

    res = nu / de

    return res
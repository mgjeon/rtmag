import torch

def Dx(f, h=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        h : float
    
    Output:
        Dx_f : [Nx, Ny, Nz]
    """
    Dx_f = torch.zeros_like(f)
    Dx_f[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2*h)
    Dx_f[0, :, :] = (-3*f[0, :, :] + 4*f[1, :, :] - f[2, :, :]) / (2*h)
    Dx_f[-1, :, :] = (3*f[-1, :, :] - 4*f[-2, :, :] + f[-3, :, :]) / (2*h)
    return Dx_f


def Dy(f, h=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        h : float
    
    Output:
        Dy_f : [Nx, Ny, Nz]
    """
    Dy_f = torch.zeros_like(f)
    Dy_f[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2*h)
    Dy_f[:, 0, :] = (-3*f[:, 0, :] + 4*f[:, 1, :] - f[:, 2, :]) / (2*h)
    Dy_f[:, -1, :] = (3*f[:, -1, :] - 4*f[:, -2, :] + f[:, -3, :]) / (2*h)
    return Dy_f


def Dz(f, h=1.0):
    """
    Input:
        f : [Nx, Ny, Nz]
        h : float
    
    Output:
        Dz_f : [Nx, Ny, Nz]
    """
    Dz_f = torch.zeros_like(f)
    Dz_f[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2*h)
    Dz_f[:, :, 0] = (-3*f[:, :, 0] + 4*f[:, :, 1] - f[:, :, 2]) / (2*h)
    Dz_f[:, :, -1] = (3*f[:, :, -1] - 4*f[:, :, -2] + f[:, :, -3]) / (2*h)
    return Dz_f

def calculateL(bx, by, bz, dx, dy, dz):
    """
    Input:
        bx, by, bz: [nx, ny, nz]
        dx, dy, dz: float
    
    Output:
        Fx, Fy, Fz: [nx, ny, nz]
        helpL, helpL1, helpL2: float
    """

    b2 = bx**2 + by**2 + bz**2

    cbx = Dy(bz, dy) - Dz(by, dz)
    cby = Dz(bx, dx) - Dx(bz, dx)
    cbz = Dx(by, dx) - Dy(bx, dy)

    fx = cby*bz - cbz*by
    fy = cbz*bx - cbx*bz
    fz = cbx*by - cby*bx

    divB = Dx(bx, dx) + Dy(by, dy) + Dz(bz, dz)

    oxa = (1/b2) * fx
    oya = (1/b2) * fy
    oza = (1/b2) * fz
    oxb = (1/b2) * (divB * bx)
    oyb = (1/b2) * (divB * by)
    ozb = (1/b2) * (divB * bz)

    o2a = oxa**2 + oya**2 + oza**2
    o2b = oxb**2 + oyb**2 + ozb**2

    oxbx = oya*bz - oza*by
    oxby = oza*bx - oxa*bz
    oxbz = oxa*by - oya*bx

    odotb = oxb*bx + oyb*by + ozb*bz

    oxjx = oya*cbz - oza*cby
    oxjy = oza*cbx - oxa*cbz
    oxjz = oxa*cby - oya*cbx

    #===============================================
    term1x = Dy(oxbz, dy) - Dz(oxby, dz)
    term1y = Dz(oxbx, dz) - Dx(oxbz, dx)
    term1z = Dx(oxby, dx) - Dy(oxbx, dy)

    term2x = oxjx 
    term2y = oxjy
    term2z = oxjz

    term3x = Dx(odotb, dx)
    term3y = Dy(odotb, dy)
    term3z = Dz(odotb, dz)

    term4x = oxb * divB
    term4y = oyb * divB
    term4z = ozb * divB

    o2a = oxa**2 + oya**2 + oza**2
    o2b = oxb**2 + oyb**2 + ozb**2

    term5ax = bx*o2a
    term5ay = by*o2a
    term5az = bz*o2a

    term5bx = bx*o2b
    term5by = by*o2b
    term5bz = bz*o2b

    term6x = oxby - oxbz
    term6y = oxbz - oxbx
    term6z = oxbx - oxby

    term7x = odotb
    term7y = odotb
    term7z = odotb

    #===============================================
    Fx = (term1x - term2x + term5ax) + (term3x - term4x + term5bx) + term6x + term7x 
    Fy = (term1y - term2y + term5ay) + (term3y - term4y + term5by) + term6y + term7y
    Fz = (term1z - term2z + term5az) + (term3z - term4z + term5bz) + term6z + term7z

    #===============================================
    helpL = b2*o2a + b2*o2b
    helpL1 = b2*o2a
    helpL2 = b2*o2b
    helpL = helpL.sum() * dx*dy*dz
    helpL1 = helpL1.sum() * dx*dy*dz
    helpL2 = helpL2.sum() * dx*dy*dz

    return Fx, Fy, Fz, helpL, helpL1, helpL2


import numpy as np 
from time import perf_counter
import pandas as pd

def relax(bp, filename, maxit=10000, mue=None, device=None):
    """
    Input:
        bp : [Nx, Ny, Nz, 3]
    
    Output:
        b : [Nx, Ny, Nz, 3]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nx, ny, nz, _ = bp.shape
    bottom = bp[:, :, 0, :]

    nd = 0
    nave = np.sqrt(1.0*(nx - 2 * nd - 1)*(ny - 2 * nd - 1))
    Lx = 1.0*(nx - 1) / nave
    Ly = 1.0*(ny - 1) / nave
    Lz = 1.0*(nz - 1) / nave
    dx = dy = dz = Lx / (nx - 1)
    Bave = np.sqrt((bottom**2).sum(-1)).sum() / (nx - 2 * nd) / (ny - 2 * nd)
    print("nx, ny, nz = ", nx, ny, nz)
    print("Lx, Ly, Lz = ", Lx, Ly, Lz)
    print("dx, dy, dz = ", dx, dy, dz)
    print("Bave = ", Bave)

    if mue is None:
        mue = 0.1*dx*dx

    print("maxit = ", maxit)
    print("mue = ", mue)

    b = torch.from_numpy(bp / Bave).to(device)

    Bx = torch.clone(b[..., 0])
    By = torch.clone(b[..., 1])
    Bz = torch.clone(b[..., 2])

    Bx1 = torch.clone(Bx)
    By1 = torch.clone(By)
    Bz1 = torch.clone(Bz)

    Bx2 = torch.clone(Bx)
    By2 = torch.clone(By)
    Bz2 = torch.clone(Bz)

    it = -1
    restore = 0
    oldL = 0
    statcount = 0

    ls = []
    fs = []
    lis = []
    fis = []

    start_time = perf_counter()

    while (it < maxit and statcount < 10 and mue > 1e-7*dx*dx):
        it = it + 1
        
        Fx, Fy, Fz, L, _, _ = calculateL(Bx, By, Bz, dx, dy, dz)

        if it == 0:
            oldL = L
        
        if restore == 1:
            L = oldL

        if it > 0 and L > oldL:
            restore = 1
            mue = 0.5*mue
            it = it - 1
            Bx1 = torch.clone(Bx2)
            By1 = torch.clone(By2)
            Bz1 = torch.clone(Bz2)

            Bx = torch.clone(Bx1)
            By = torch.clone(By1)
            Bz = torch.clone(Bz1)
        
        else: 
            mue = 1.01*mue
            restore = 0
            oldL = L

    
        diagstep = 10
        if it % diagstep == 0:
            Fxi, Fyi, Fzi, Li, _, _ = calculateL(Bx[1:-1, 1:-1, 1:-1], By[1:-1, 1:-1, 1:-1], Bz[1:-1, 1:-1, 1:-1], dx, dy, dz)

            F = torch.sqrt((Fx**2 + Fy**2 + Fz**2)).mean()
            Fi = torch.sqrt((Fxi**2 + Fyi**2 + Fzi**2)).mean()

            print("it = ", it, "L = ", L.item(), "F = ", F.item(), "Li = ", Li.item(), "Fi = ", Fi.item())

            ls.append(L.item())
            fs.append(F.item())
            lis.append(Li.item())
            fis.append(Fi.item())

            if it == 0:
                prevL = 2.0*L 
                newL = L
            else: 
                prevL = newL
                newL = L
            
            gradL = torch.abs((newL - prevL) / newL)
            if gradL < 0.00001:
                statcount = statcount + 1
                print("STATIONARY COUNT = ", statcount, "grad L/L", gradL.item())
            if gradL > 0.00001:
                statcount = 0
                print("grad L/L", gradL.item())

        if oldL >= L:
            Bx1[1:-1, 1:-1, 1:-1] = Bx[1:-1, 1:-1, 1:-1] + mue*Fx[1:-1, 1:-1, 1:-1]
            By1[1:-1, 1:-1, 1:-1] = By[1:-1, 1:-1, 1:-1] + mue*Fy[1:-1, 1:-1, 1:-1]
            Bz1[1:-1, 1:-1, 1:-1] = Bz[1:-1, 1:-1, 1:-1] + mue*Fz[1:-1, 1:-1, 1:-1]
        
        Bx2 = torch.clone(Bx)
        By2 = torch.clone(By)
        Bz2 = torch.clone(Bz)

        Bx = torch.clone(Bx1)
        By = torch.clone(By1)
        Bz = torch.clone(Bz1)

    end_time = perf_counter()

    total_time = end_time - start_time

    df = pd.DataFrame({
        'L': ls,
        'F': fs,
        'Li': lis,
        'Fi': fis,
    })
    df.to_csv(f"{filename}.csv", index=False)

    with open(f"{filename}.txt", "w") as f:
        print(f"total_it: {it}", file=f)
        print(f"total_time: {total_time} seconds", file=f)

    print(f"total_it: {it}")
    print(f"total_time: {total_time} seconds")

    b = np.stack([Bx.detach().cpu().numpy(), 
                   By.detach().cpu().numpy(), 
                   Bz.detach().cpu().numpy()], axis=-1)*Bave

    return b


def relax_withoutprint(bp, filename, path, maxit=10000, mue=None, device=None):
    """
    Input:
        bp : [Nx, Ny, Nz, 3]
    
    Output:
        b : [Nx, Ny, Nz, 3]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nx, ny, nz, _ = bp.shape
    bottom = bp[:, :, 0, :]

    nd = 0
    nave = np.sqrt(1.0*(nx - 2 * nd - 1)*(ny - 2 * nd - 1))
    Lx = 1.0*(nx - 1) / nave
    Ly = 1.0*(ny - 1) / nave
    Lz = 1.0*(nz - 1) / nave
    dx = dy = dz = Lx / (nx - 1)
    Bave = np.sqrt((bottom**2).sum(-1)).sum() / (nx - 2 * nd) / (ny - 2 * nd)
    if mue is None:
        mue = 0.1*dx*dx
    b = torch.from_numpy(bp / Bave).to(device)

    Bx = torch.clone(b[..., 0])
    By = torch.clone(b[..., 1])
    Bz = torch.clone(b[..., 2])

    Bx1 = torch.clone(Bx)
    By1 = torch.clone(By)
    Bz1 = torch.clone(Bz)

    Bx2 = torch.clone(Bx)
    By2 = torch.clone(By)
    Bz2 = torch.clone(Bz)

    it = -1
    restore = 0
    oldL = 0
    statcount = 0

    ls = []
    fs = []
    lis = []
    fis = []

    start_time = perf_counter()

    while (it < maxit and statcount < 10 and mue > 1e-7*dx*dx):
        it = it + 1
        
        Fx, Fy, Fz, L, _, _ = calculateL(Bx, By, Bz, dx, dy, dz)

        if it == 0:
            oldL = L
        
        if restore == 1:
            L = oldL

        if it > 0 and L > oldL:
            restore = 1
            mue = 0.5*mue
            it = it - 1
            Bx1 = torch.clone(Bx2)
            By1 = torch.clone(By2)
            Bz1 = torch.clone(Bz2)

            Bx = torch.clone(Bx1)
            By = torch.clone(By1)
            Bz = torch.clone(Bz1)
        
        else: 
            mue = 1.01*mue
            restore = 0
            oldL = L

    
        diagstep = 10
        if it % diagstep == 0:
            Fxi, Fyi, Fzi, Li, _, _ = calculateL(Bx[1:-1, 1:-1, 1:-1], By[1:-1, 1:-1, 1:-1], Bz[1:-1, 1:-1, 1:-1], dx, dy, dz)

            F = torch.sqrt((Fx**2 + Fy**2 + Fz**2)).mean()
            Fi = torch.sqrt((Fxi**2 + Fyi**2 + Fzi**2)).mean()

            # print("it = ", it, "L = ", L, "F = ", F, "Li = ", Li, "Fi = ", Fi)

            ls.append(L.item())
            fs.append(F.item())
            lis.append(Li.item())
            fis.append(Fi.item())

            if it == 0:
                prevL = 2.0*L 
                newL = L
            else: 
                prevL = newL
                newL = L
            
            gradL = torch.abs((newL - prevL) / newL)
            if gradL < 0.00001:
                statcount = statcount + 1
                # print("STATIONARY COUNT = ", statcount, "grad L/L", gradL)
            if gradL > 0.00001:
                statcount = 0
                # print("grad L/L", gradL)

        if oldL >= L:
            Bx1[1:-1, 1:-1, 1:-1] = Bx[1:-1, 1:-1, 1:-1] + mue*Fx[1:-1, 1:-1, 1:-1]
            By1[1:-1, 1:-1, 1:-1] = By[1:-1, 1:-1, 1:-1] + mue*Fy[1:-1, 1:-1, 1:-1]
            Bz1[1:-1, 1:-1, 1:-1] = Bz[1:-1, 1:-1, 1:-1] + mue*Fz[1:-1, 1:-1, 1:-1]
        
        Bx2 = torch.clone(Bx)
        By2 = torch.clone(By)
        Bz2 = torch.clone(Bz)

        Bx = torch.clone(Bx1)
        By = torch.clone(By1)
        Bz = torch.clone(Bz1)

    end_time = perf_counter()

    total_time = end_time - start_time

    df = pd.DataFrame({
        'L': ls,
        'F': fs,
        'Li': lis,
        'Fi': fis,
    })
    df.to_csv(path / f"{filename}.csv", index=False)

    with open(path / f"{filename}.txt", "w") as f:
        print(f"total_it: {it}", file=f)
        print(f"total_time: {total_time} seconds", file=f)

    b = np.stack([Bx.detach().cpu().numpy(), 
                   By.detach().cpu().numpy(), 
                   Bz.detach().cpu().numpy()], axis=-1)*Bave

    return b
from rtmag.utils.diff_torch_batch import Dx, Dy, Dz

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



def calculateLonly(bx, by, bz, dx, dy, dz):
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

    #===============================================
    helpL1 = b2*o2a
    helpL2 = b2*o2b
    helpL1 = helpL1.sum() * dx*dy*dz
    helpL2 = helpL2.sum() * dx*dy*dz

    return helpL1, helpL2
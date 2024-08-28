import numpy as np 

def read_bin(filename, shape=(3, 512, 256, 256)):
    """
    shape = (3, Nx, Ny, Nz)

    Bx = f[0]
    By = f[1]
    Bz = f[2]

    """
    f = np.fromfile(filename, dtype=np.float64)
    f = f.reshape(shape)
    return f


def save_bin_from_b(b, filename):
    """
    b : [Nx, Ny, Nz, 3]
    """

    b = b.astype(np.float64)
    bx = b[..., 0]
    by = b[..., 1]
    bz = b[..., 2]
    
    with open(filename, 'wb') as f:
        f.write(bytearray(bx))
        f.write(bytearray(by))
        f.write(bytearray(bz))
    
    print(f"Saved to {filename}")


def save_npz_from_bottom(bottom, filename):
    """
    bottom : [Nx, Ny, 3]
    """

    bx = bottom[..., 0]
    by = bottom[..., 1]
    bz = bottom[..., 2]

    np.savez(filename, bx=bx, by=by, bz=bz)
    
    print(f"Saved to {filename}")
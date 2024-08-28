import numpy as np
from skimage.transform import resize

def get_input(hmi_data, nx=512, ny=256):

    onx, ony, _ = hmi_data.shape
    odx = 696 * 0.03 * np.pi / 180
    ody = odx

    dx = ((onx - 1)/(nx - 1))*odx
    dy = ((ony - 1)/(ny - 1))*ody
    dz = dy
    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dy, ny)
    z = y

    print(f"dx = {dx} Mm")
    print(f"dy = {dy} Mm")
    print(f"dz = {dz} Mm")

    model_input = resize(hmi_data, (nx, ny, 3))
    model_input = model_input[None, :, :, None, :]
    model_input = model_input.transpose(0, 3, 2, 1, 4)
    return model_input, x, y, z, dx, dy, dz
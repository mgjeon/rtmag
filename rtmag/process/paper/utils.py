import yaml
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import readsav
from sunpy.map import Map

def energy(b):
    return ((b ** 2).sum(-1) / (8 * np.pi)).sum()

# IDL congrid
def congrid(array, reshape=(512, 256, 3), minus_one=False):
    """
    Resample an array to a new shape using IDL congrid algorithm.
    """
    if isinstance(reshape, int):
        reshape = (reshape, reshape, reshape)
    elif isinstance(reshape, tuple):
        reshape = reshape
    else:
        raise ValueError("reshape must be an integer or a tuple of integers")

    return _congrid(array, new_x=reshape[0], new_y=reshape[1], new_z=reshape[2], minus_one=minus_one)

def _congrid(array, new_x, new_y=None, new_z=None, minus_one=False):
    old_shape = array.shape
    dim = len(old_shape)

    if dim == 1:
        if new_y is not None or new_z is not None:
            raise ValueError("For 1D array, new_y and new_z must be None")
        return _congrid_1d(array, new_x, minus_one)
    elif dim == 2:
        if new_z is not None:
            raise ValueError("For 2D array, new_z must be None")
        if new_y is None:
            raise ValueError("For 2D array, new_y must be specified")
        return _congrid_2d(array, new_x, new_y, minus_one)
    elif dim == 3:
        if new_y is None or new_z is None:
            raise ValueError("For 3D array, new_y and new_z must be specified")
        return _congrid_3d(array, new_x, new_y, new_z, minus_one)
    else:
        raise ValueError("Only 1D, 2D, and 3D arrays are supported")

def _congrid_1d(array, new_x, minus_one):
    old_x = array.shape[0]
    scale_x = (old_x - int(minus_one)) / (new_x - int(minus_one))
    indices = (np.arange(new_x) * scale_x).astype(int)
    return array[indices]

def _congrid_2d(array, new_x, new_y, minus_one):
    old_x, old_y = array.shape
    scale_x = (old_x - int(minus_one)) / (new_x - int(minus_one))
    scale_y = (old_y - int(minus_one)) / (new_y - int(minus_one))
    indices_x = (np.arange(new_x) * scale_x).astype(int)
    indices_y = (np.arange(new_y) * scale_y).astype(int)
    return array[np.ix_(indices_x, indices_y)]

def _congrid_3d(array, new_x, new_y, new_z, minus_one):
    old_x, old_y, old_z = array.shape
    scale_x = (old_x - int(minus_one)) / (new_x - int(minus_one))
    scale_y = (old_y - int(minus_one)) / (new_y - int(minus_one))
    scale_z = (old_z - int(minus_one)) / (new_z - int(minus_one))
    indices_x = (np.arange(new_x) * scale_x).astype(int)
    indices_y = (np.arange(new_y) * scale_y).astype(int)
    indices_z = (np.arange(new_z) * scale_z).astype(int)
    return array[np.ix_(indices_x, indices_y, indices_z)]


def plot_model_input(model_input, figsize=(12, 8)):
    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.title("Bx")
    plt.imshow(model_input[0, 0, :, :, 0], cmap='gray', origin='lower',
               vmin=-1000, vmax=1000)

    plt.subplot(1, 3, 2)
    plt.title("By")
    plt.imshow(model_input[0, 0, :, :, 1], cmap='gray', origin='lower',
               vmin=-1000, vmax=1000)

    plt.subplot(1, 3, 3)
    plt.title("Bz")
    plt.imshow(model_input[0, 0, :, :, 2], cmap='gray', origin='lower',
               vmin=-1000, vmax=1000)
    
    plt.tight_layout()
    plt.show()


def plot_image(data):
    """
    data: [Nx, Ny, 3]
    """

    fig = plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.title("Bx")
    plt.imshow(data[..., 0].T, cmap='gray', origin='lower',
            vmin=-1000, vmax=1000)

    plt.subplot(1, 3, 2)
    plt.title("By")
    plt.imshow(data[..., 1].T, cmap='gray', origin='lower',
            vmin=-1000, vmax=1000)

    plt.subplot(1, 3, 3)
    plt.title("Bz")
    plt.imshow(data[..., 2].T, cmap='gray', origin='lower',
            vmin=-1000, vmax=1000)


def load_prt_sav(file_path):
    """
    Load the .sav file
    """
    b = readsav(file_path)['data']
    bx = b[0].T
    by = -b[1].T
    bz = b[2].T
    b = np.stack([bx, by, bz], axis=-1)
    return b


def get_label(label_path, clip=None):
    B = np.load(label_path)["label"][:, :-1, :-1, :-1].astype(np.float32)
    B = B.transpose(1, 2, 3, 0)
    if clip is not None:
        B = np.clip(B, -clip, clip)

    return B.astype(np.float32)

def get_label_input(label_path):
    B = np.load(label_path)["label"][:, :-1, :-1, :-1].astype(np.float32)
    B = B.transpose(1, 2, 3, 0)
    
    model_input = B[:, :, 0, :].transpose(1, 0, 2)[None, None, ...]

    return B, model_input

def get_label_xyz(label_path, clip=None):
    labels = np.load(label_path)
    B = labels["label"][:, :-1, :-1, :-1].astype(np.float32)
    B = B.transpose(1, 2, 3, 0)
    if clip is not None:
        B = np.clip(B, -clip, clip)

    x = labels["x"][:-1]
    y = labels["y"][:-1]
    z = labels["z"][:-1]

    return B, x, y, z

def get_pot(label_path, clip=None):
    B = np.load(label_path)["pot"][:, :-1, :-1, :-1].astype(np.float32)
    B = B.transpose(1, 2, 3, 0)
    if clip is not None:
        B = np.clip(B, -clip, clip)

    return B.astype(np.float32)


def get_input(input_path):
    B = np.load(input_path)['input'][..., 0].transpose(1, 2, 0)
    B = B[:-1, :-1, :]

    # (Nx, Ny, 3)
    return B.astype(np.float32)

def input_to_model_input(input_path):
    # (Nx, Ny, 3)
    inputs = get_input(input_path)

    # (Nx, Ny, 3) -> (1, 1, Ny, Nx, 3)
    model_input = inputs.transpose(1, 0, 2)[None, None, ...]
    return model_input


# Function to load YAML file and return it as a dictionary
def load_yaml_to_dict(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def get_hmi(bp_path, bt_path, br_path):
    bx = Map(bp_path).data.T
    by = -Map(bt_path).data.T
    bz = Map(br_path).data.T
    
    # (Nx, Ny, 3)
    b = np.stack([bx, by, bz], axis=-1)
    return b


def get_hmi_congrid(bp_path, bt_path, br_path, reshape=(512, 256, 3)):
    b = get_hmi(bp_path, bt_path, br_path)
    b = congrid(b, reshape[0], reshape[1], reshape[2])

    # (Nx, Ny, 3)
    return b


def hmi_to_model_input(hmi_b, nx=512, ny=256, isplot=False):
    """
    hmi_b: [Nx, Ny, 3]
    
    return: [1, 1, Ny, Nx, 3]
    """

    res = {}

    ox, oy, _ = hmi_b.shape
    model_input = congrid(hmi_b, (nx, ny, 3))
    # [512, 256, 3] -> [bs, 1, 256, 512, 3]
    model_input = model_input.transpose(1, 0, 2)[None, None, ...]

    if isplot:
        plot_model_input(model_input)
    
    # l = 0.36 # Mm
    print("model_input", model_input.shape)
    print(f"HMI : {ox, oy}")
    print(f"NEW : {nx, ny}")
    l = 696 * 0.03 * np.pi / 180
    print(f"1 pixel = {l} Mm")

    dx = ((ox - 1)/(nx - 1))*l
    dy = ((oy - 1)/(ny - 1))*l
    # dx = (ox/nx) * l
    # dy = (oy/ny) * l
    dz = dy 

    print(f"dx = {dx} Mm")
    print(f"dy = {dy} Mm")
    print(f"dz = {dz} Mm")

    dx_cm, dy_cm, dz_cm = dx * 1e8, dy * 1e8, dz * 1e8  # cm
    dV = dx_cm * dy_cm * dz_cm # cm^3
    print(f"dV = {dV} cm^3")

    Lx = (ox - 1) * l 
    Ly = (oy - 1) * l
    # Lx = ox * l
    # Ly = oy * l
    Lz = Ly

    print(f"Lx = {Lx} Mm")
    print(f"Ly = {Ly} Mm")
    print(f"Lz = {Lz} Mm")

    x = np.linspace(0, (ox-1)*l, nx)
    y = np.linspace(0, (oy-1)*l, ny)
    z = y

    assert np.allclose(dx, np.diff(x)[0])
    assert np.allclose(dy, np.diff(y)[0])
    assert np.allclose(dz, np.diff(z)[0])

    x = x - x[256]
    y = y - y[128]

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)

    print("xyz", x.shape, y.shape, z.shape)

    res['ox'] = ox 
    res['oy'] = oy
    res['nx'] = nx
    res['ny'] = ny 
    res['dx'] = dx
    res['dy'] = dy
    res['dz'] = dz
    res['dx_cm'] = dx_cm
    res['dy_cm'] = dy_cm
    res['dz_cm'] = dz_cm
    res['dV'] = dV
    res['Lx'] = Lx
    res['Ly'] = Ly
    res['Lz'] = Lz
    res['x'] = x
    res['y'] = y
    res['z'] = z

    return model_input, res


def hmi_to_congrid(hmi_b, nx=512, ny=256, isprint=True):
    """
    hmi_b: [Nx, Ny, 3]
    
    return: [1, 1, Ny, Nx, 3]
    """

    res_o = {}
    res = {}

    # [onx, ony, 3]
    onx, ony, _ = hmi_b.shape
    # [nx, ny, 3]
    hmi_b_congrid = congrid(hmi_b, (nx, ny, 3))
    if isprint:
        print(f"HMI : {onx, ony}")
        print(f"NEW : {nx, ny}")

    res_o['nx'] = onx 
    res_o['ny'] = ony
    res['nx'] = nx
    res['ny'] = ny 

    #----------------------------------------------------
    # l = 0.36 Mm
    odx = 696 * 0.03 * np.pi / 180
    ody = odx
    odx = np.float32(odx)
    ody = np.float32(ody)
    if isprint:
        print(f"odx = {odx} Mm")
        print(f"ody = {ody} Mm")

    dx = ((onx - 1)/(nx - 1))*odx
    dy = ((ony - 1)/(ny - 1))*ody
    dx = np.float32(dx)
    dy = np.float32(dy)
    if isprint:
        print(f"dx = {dx} Mm")
        print(f"dy = {dy} Mm")

    res_o['dx'] = odx
    res_o['dy'] = ody
    res['dx'] = dx
    res['dy'] = dy
    
    #----------------------------------------------------
    Lx = (onx - 1) * odx
    Ly = (ony - 1) * ody
    Lx = np.float32(Lx)
    Ly = np.float32(Ly)
    if isprint:
        print(f"Lx = {Lx} Mm")
        print(f"Ly = {Ly} Mm")

    res_o['Lx'] = Lx
    res_o['Ly'] = Ly
    res['Lx'] = Lx
    res['Ly'] = Ly

    #----------------------------------------------------
    ox = np.linspace(0, (onx-1)*odx, onx)
    oy = np.linspace(0, (ony-1)*ody, ony)
    assert np.allclose(odx, np.diff(ox)[0])
    assert np.allclose(ody, np.diff(oy)[0])
    # ox = ox - ox[onx//2]
    # oy = oy - oy[ony//2]
    ox = ox.astype(np.float32)
    oy = oy.astype(np.float32)

    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dy, ny)
    assert np.allclose(dx, np.diff(x)[0])
    assert np.allclose(dy, np.diff(y)[0])
    # x = x - x[nx//2]
    # y = y - y[ny//2]
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    res_o['x'] = ox
    res_o['y'] = oy
    res['x'] = x
    res['y'] = y

    #----------------------------------------------------
    odx_cm, ody_cm = odx * 1e8, ody * 1e8  # cm
    odx_cm = np.float32(odx_cm)
    ody_cm = np.float32(ody_cm)

    dx_cm, dy_cm = dx * 1e8, dy * 1e8  # cm
    dx_cm = np.float32(dx_cm)
    dy_cm = np.float32(dy_cm)

    res_o['dx_cm'] = odx_cm
    res_o['dy_cm'] = ody_cm
    res['dx_cm'] = dx_cm
    res['dy_cm'] = dy_cm

    return hmi_b_congrid, res_o, res


# plot ----------------------------------------------------------------
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator

def draw_z_slices(b, B, lbl="$B_z$", comp=None, hs=[0, 5, 10], vmms=[2000, 500, 100], cmap="gray", figsize=(17,13), 
                  x=None, y=None, 
                  left_title=None, right_title=None, title_fontsize=None, title_y=None, label_fontsize=None, tick_fontsize=None,
                  xlocator=None, ylocator=None, major_ticksize=None, bar_size=None, bar_pad=None, bar_title_fontsize=None, 
                  bar_ticks_fontsize=None, **kwargs):
    """
    b : prediction
    B : label
    """
    fig, axs = plt.subplots(3, 2, figsize=figsize)

    #------------------------------------------------------------------------------
    h = hs[0]
    vmm = vmms[0]
    vmin = -vmm
    vmax = vmm
    if comp is None:
        img = B[:, :, h].T
        vmin = 0
    else:
        img = B[:, :, h, comp].T
    axs[0, 0].pcolormesh(x, y, img, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0, 0].set_title(left_title, fontsize=title_fontsize, y=title_y)
    axs[0, 0].set_ylabel("y (Mm)", fontsize=label_fontsize)
    axs[0, 0].tick_params(labelsize=tick_fontsize, which='both', top=True, right=True)
    axs[0, 0].set_aspect("equal")
    axs[0, 0].xaxis.set_ticklabels([])
    axs[0, 0].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[0, 0].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[0, 0].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[0, 0].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    axs[0, 0].tick_params(which='major', length=major_ticksize)
    axs[0, 0].tick_params(which='minor', length=major_ticksize/2)
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size=bar_size, pad=bar_pad)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.axis("off")
    
    if comp is None:
        img = b[:, :, h].T
        vmin = 0
    else:
        img = b[:, :, h, comp].T
    im = axs[0, 1].pcolormesh(x, y, img, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title(right_title, fontsize=title_fontsize, y=title_y)
    # axs[0, 1].sharey(axs[0, 0])
    axs[0, 1].tick_params(labelsize=tick_fontsize, which='both', top=True, right=True)
    axs[0, 1].set_aspect("equal")
    axs[0, 1].xaxis.set_ticklabels([])
    axs[0, 1].yaxis.set_ticklabels([])
    axs[0, 1].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[0, 1].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[0, 1].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[0, 1].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    axs[0, 1].tick_params(which='major', length=major_ticksize)
    axs[0, 1].tick_params(which='minor', length=major_ticksize/2)
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size=bar_size, pad=bar_pad)
    cb = fig.colorbar(im, cax=cax, ticks=[-vmm, -vmm/2, 0, vmm/2, vmm])
    cb.set_label(lbl+' $(z='+f'{h}'+')$\n(G)', size=bar_title_fontsize)
    cax.tick_params(labelsize=bar_ticks_fontsize)

    #------------------------------------------------------------------------------
    h = hs[1]
    vmm = vmms[1]
    vmin = -vmm
    vmax = vmm
    if comp is None:
        img = B[:, :, h].T
        vmin = 0
    else:
        img = B[:, :, h, comp].T
    axs[1, 0].pcolormesh(x, y, img, cmap=cmap, vmin=vmin, vmax=vmax)
    # axs[1, 0].sharex(axs[0, 0])
    axs[1, 0].set_ylabel("y (Mm)", fontsize=label_fontsize)
    axs[1, 0].tick_params(labelsize=tick_fontsize, which='both', top=True, right=True)
    axs[1, 0].set_aspect("equal")
    axs[1, 0].xaxis.set_ticklabels([])
    axs[1, 0].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[1, 0].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[1, 0].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[1, 0].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    axs[1, 0].tick_params(which='major', length=major_ticksize)
    axs[1, 0].tick_params(which='minor', length=major_ticksize/2)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size=bar_size, pad=bar_pad)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.axis("off")
    
    if comp is None:
        img = b[:, :, h].T
        vmin = 0
    else:
        img = b[:, :, h, comp].T
    im = axs[1, 1].pcolormesh(x, y, img, cmap=cmap, vmin=vmin, vmax=vmax)
    # axs[1, 1].sharex(axs[0, 1])
    # axs[1, 1].sharey(axs[1, 0])
    axs[1, 1].tick_params(labelsize=tick_fontsize, which='both', top=True, right=True)
    axs[1, 1].set_aspect("equal")
    axs[1, 1].xaxis.set_ticklabels([])
    axs[1, 1].yaxis.set_ticklabels([])
    axs[1, 1].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[1, 1].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[1, 1].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[1, 1].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    axs[1, 1].tick_params(which='major', length=major_ticksize)
    axs[1, 1].tick_params(which='minor', length=major_ticksize/2)
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size=bar_size, pad=bar_pad)
    cb = fig.colorbar(im, cax=cax, ticks=[-vmm, -vmm/2, 0, vmm/2, vmm])
    cb.set_label(lbl+' $(z='+f'{h}'+')$\n(G)', size=bar_title_fontsize)
    cax.tick_params(labelsize=bar_ticks_fontsize)

    #------------------------------------------------------------------------------
    h = hs[2]
    vmm = vmms[2]
    vmin = -vmm
    vmax = vmm
    if comp is None:
        img = B[:, :, h].T
        vmin = 0
    else:
        img = B[:, :, h, comp].T
    axs[2, 0].pcolormesh(x, y, img, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2, 0].set_xlabel("x (Mm)", fontsize=label_fontsize)
    # axs[2, 0].sharex(axs[0, 0])
    axs[2, 0].set_ylabel("y (Mm)", fontsize=label_fontsize)
    axs[2, 0].tick_params(labelsize=tick_fontsize, which='both', top=True, right=True)
    axs[2, 0].set_aspect("equal")
    axs[2, 0].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[2, 0].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[2, 0].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[2, 0].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    axs[2, 0].tick_params(which='major', length=major_ticksize)
    axs[2, 0].tick_params(which='minor', length=major_ticksize/2)
    divider = make_axes_locatable(axs[2, 0])
    cax = divider.append_axes("right", size=bar_size, pad=bar_pad)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.axis("off")
    
    if comp is None:
        img = b[:, :, h].T
        vmin = 0
    else:
        img = b[:, :, h, comp].T
    im = axs[2, 1].pcolormesh(x, y, img, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2, 1].set_xlabel("x (Mm)", fontsize=label_fontsize)
    # axs[2, 1].sharex(axs[0, 1])
    # axs[2, 1].sharey(axs[2, 0])
    axs[2, 1].tick_params(labelsize=tick_fontsize, which='both', top=True, right=True)
    axs[2, 1].set_aspect("equal")
    axs[2, 1].yaxis.set_ticklabels([])
    axs[2, 1].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[2, 1].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[2, 1].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[2, 1].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    axs[2, 1].tick_params(which='major', length=major_ticksize)
    axs[2, 1].tick_params(which='minor', length=major_ticksize/2)
    divider = make_axes_locatable(axs[2, 1])
    cax = divider.append_axes("right", size=bar_size, pad=bar_pad)
    cb = fig.colorbar(im, cax=cax, ticks=[-vmm, -vmm/2, 0, vmm/2, vmm])
    cb.set_label(lbl+' $(z='+f'{h}'+')$\n(G)', size=bar_title_fontsize)
    cax.tick_params(labelsize=bar_ticks_fontsize)

    plt.tight_layout()
    plt.show()


def draw_integrate_map(j_map, J_map, lbl='Integrated current density\n(mA/cm)', vmin=500, vmax=3000, cmap="jet", figsize=(15, 12),
                       x=None, y=None, left_title=None, right_title=None, title_fontsize=None, title_y=None, label_fontsize=None,
                       tick_fontsize=None, xlocator=None, ylocator=None, major_ticksize=None, bar_size=None, bar_pad=None,
                       bar_title_fontsize=None, bar_ticks_fontsize=None, **kwargs):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    #------------------------------------------------------------------------------
    axs[0].pcolormesh(x, y, J_map.T, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(left_title, fontsize=title_fontsize, y=title_y)
    axs[0].set_ylabel("y (Mm)", fontsize=label_fontsize)
    axs[0].tick_params(labelsize=tick_fontsize, which='both', top=True, right=True)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("x (Mm)", fontsize=label_fontsize)
    axs[0].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[0].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[0].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[0].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    axs[0].tick_params(which='major', length=major_ticksize)
    axs[0].tick_params(which='minor', length=major_ticksize/2)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size=bar_size, pad=bar_pad)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.axis("off")

    im = axs[1].pcolormesh(x, y, j_map.T, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title(right_title, fontsize=title_fontsize, y=title_y)
    axs[1].tick_params(labelsize=tick_fontsize, which='both', top=True, right=True)
    axs[1].set_aspect("equal")
    axs[1].yaxis.set_ticklabels([])
    axs[1].set_xlabel("x (Mm)", fontsize=label_fontsize)
    axs[1].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[1].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[1].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[1].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    axs[1].tick_params(which='major', length=major_ticksize)
    axs[1].tick_params(which='minor', length=major_ticksize/2)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size=bar_size, pad=bar_pad)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(lbl, size=bar_title_fontsize)
    cax.tick_params(labelsize=bar_ticks_fontsize)

    plt.tight_layout()
    plt.show()


def draw_fieldlines(b, B, vmm=2000, cmap="gray", tit=(0.02, 0.90), times=(0.67, 0.05), figsize=(17, 10), 
                    x=None, y=None, dx=None, dy=None, left_title=None, fl_title_fontsize=None, fl_title_y=None, 
                    fl_label_fontsize=None, fl_tick_fontsize=None, fl_major_ticksize=None,
                    xlocator=None, ylocator=None, hmi_title=None, hmi_timestamp=None, 
                    annotation_fontsize=None, tracer_xs_true=None, hmi_linecolor=None, hmi_linesize=None,
                    aia_171_resampled_map=None, aia_171_title=None, aia_171_timestamp=None,
                    aia_171_linecolor=None, aia_171_linesize=None, right_title=None, tracer_xs=None, **kwargs):
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # ISEE ---------------------------------------------------------------
    axs[0, 0].pcolormesh(x, y, B[:, :, 0, 2].T, cmap=cmap, vmin=-vmm, vmax=vmm)
    axs[0, 0].set_title(left_title, fontsize=fl_title_fontsize, y=fl_title_y)
    axs[0, 0].set_ylabel("y (Mm)", fontsize=fl_label_fontsize)
    axs[0, 0].tick_params(labelsize=fl_tick_fontsize, which='both', top=True, right=True)
    axs[0, 0].tick_params(which='major', length=fl_major_ticksize)
    axs[0, 0].tick_params(which='minor', length=fl_major_ticksize/2)
    axs[0, 0].set_aspect("equal")
    axs[0, 0].xaxis.set_ticklabels([])
    axs[0, 0].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[0, 0].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[0, 0].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[0, 0].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    axs[0, 0].annotate(f"{hmi_title}", xy=tit, xycoords='axes fraction', color='white', fontsize=annotation_fontsize)
    axs[0, 0].annotate(f"{hmi_timestamp}", xy=times, xycoords='axes fraction', color='white', fontsize=annotation_fontsize)
    for xl in tracer_xs_true:
        axs[0, 0].plot(xl[:,0]*dx + x[0], xl[:,1]*dy + y[0], color=hmi_linecolor, linewidth=hmi_linesize)
    
    axs[1, 0].pcolormesh(x, y, aia_171_resampled_map.data, 
                         cmap = aia_171_resampled_map.plot_settings['cmap'],
                         norm = aia_171_resampled_map.plot_settings['norm'])
    axs[1, 0].set_title("")
    axs[1, 0].set_xlabel("x (Mm)", fontsize=fl_label_fontsize)
    axs[1, 0].set_ylabel("y (Mm)", fontsize=fl_label_fontsize)
    axs[1, 0].tick_params(labelsize=fl_tick_fontsize, which='both', top=True, right=True)
    axs[1, 0].tick_params(which='major', length=fl_major_ticksize)
    axs[1, 0].tick_params(which='minor', length=fl_major_ticksize/2)
    axs[1, 0].set_aspect("equal")
    axs[1, 0].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[1, 0].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[1, 0].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[1, 0].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    axs[1, 0].annotate(f"{aia_171_title}", xy=tit, xycoords='axes fraction', color='white', fontsize=annotation_fontsize)
    axs[1, 0].annotate(f"{aia_171_timestamp}", xy=times, xycoords='axes fraction', color='white', fontsize=annotation_fontsize)
    for xl in tracer_xs_true:
        axs[1, 0].plot(xl[:,0]*dx + x[0], xl[:,1]*dy + y[0], color=aia_171_linecolor, linewidth=aia_171_linesize)
    
    
    # PINO ---------------------------------------------------------------
    axs[0, 1].pcolormesh(x, y, b[:, :, 0, 2].T, cmap=cmap, vmin=-vmm, vmax=vmm)
    axs[0, 1].set_title(right_title, fontsize=fl_title_fontsize, y=fl_title_y)
    axs[0, 1].tick_params(labelsize=fl_tick_fontsize, which='both', top=True, right=True)
    axs[0, 1].tick_params(which='major', length=fl_major_ticksize)
    axs[0, 1].tick_params(which='minor', length=fl_major_ticksize/2)
    axs[0, 1].set_aspect("equal")
    # axs[0, 1].sharey(axs[0, 0])
    axs[0, 1].xaxis.set_ticklabels([])
    axs[0, 1].yaxis.set_ticklabels([])
    axs[0, 1].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[0, 1].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[0, 1].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[0, 1].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    for xl in tracer_xs:
        axs[0, 1].plot(xl[:,0]*dx + x[0], xl[:,1]*dy + y[0], color=hmi_linecolor, linewidth=hmi_linesize)
    
    axs[1, 1].pcolormesh(x, y, aia_171_resampled_map.data, 
                         cmap = aia_171_resampled_map.plot_settings['cmap'],
                         norm = aia_171_resampled_map.plot_settings['norm'])
    axs[1, 1].set_title("")
    axs[1, 1].set_xlabel("x (Mm)", fontsize=fl_label_fontsize)
    axs[1, 1].tick_params(labelsize=fl_tick_fontsize, which='both', top=True, right=True)
    axs[1, 1].tick_params(which='major', length=fl_major_ticksize)
    axs[1, 1].tick_params(which='minor', length=fl_major_ticksize/2)
    axs[1, 1].set_aspect("equal")
    axs[1, 1].yaxis.set_ticklabels([])
    axs[1, 1].xaxis.set_major_locator(MultipleLocator(xlocator))
    axs[1, 1].xaxis.set_minor_locator(MultipleLocator(xlocator/2))
    axs[1, 1].yaxis.set_major_locator(MultipleLocator(ylocator))
    axs[1, 1].yaxis.set_minor_locator(MultipleLocator(ylocator/2))
    for xl in tracer_xs:
        axs[1, 1].plot(xl[:,0]*dx + x[0], xl[:,1]*dy + y[0], color=aia_171_linecolor, linewidth=aia_171_linesize)
    
    plt.tight_layout()
    plt.show()

def B_to_model_input(B):
    """
    B: [Nx, Ny, Nz, 3]
    """

    # bottom (Nx, Ny, 3)
    bottom = B[:, :, 0, :]

    # (Nx, Ny, 3) -> (1, 1, 256, 512, 3)
    model_input = bottom.transpose(1, 0, 2)[None, None, ...]

    print("model_input", model_input.shape)

    return model_input

def get_model_input(filename, ret=True):
    bottom = np.load(filename)

    # (Nx, Ny)
    bx = bottom['bx']
    by = bottom['by']
    bz = bottom['bz']

    # (Nx, Ny, 3)
    model_input = np.stack([bx, by, bz], axis=-1)
    # (Nx, Ny, 3) -> (1, 1, Ny, Nx, 3)
    model_input = model_input.transpose(1, 0, 2)[None, None, ...]

    if ret:
        print("model_input", model_input.shape)

    x = bottom['x']
    y = bottom['y']

    return model_input, x, y


# field line

def get_square_indices(center, size=3, step=2):
    row, col = center
    half_size = size // 2
    
    rows = np.arange(row - half_size, row + half_size + 1, step)
    cols = np.arange(col - half_size, col + half_size + 1, step)

    row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')

    indices = np.stack([row_grid, col_grid], axis=-1).reshape(-1, 2)

    indices = [tuple(idx) for idx in indices]
    
    indices = [(xx, yy, 0) for xx, yy in indices]
    return np.array(indices)

def return_seeds(xypoints, seed_size, seed_step,
                 x, y, dx, dy):
    seeds_xs = []

    for xypoint in xypoints:
        i = int((xypoint[0] - x[0])/dx)
        j = int((xypoint[1] - y[0])/dy)
        seeds = get_square_indices((i, j), size=seed_size, step=seed_step)
        seeds_xs.append(seeds)

    seeds_xs = np.vstack(seeds_xs)

    return seeds_xs

def return_fieldlines(tracer, grid, seeds_xs):
    tracer_xs = []
    tracer.trace(seeds_xs, grid)
    tracer_xs.append(tracer.xs)
    tracer_xs = [item for sublist in tracer_xs for item in sublist]

    return tracer_xs



def load_bin(filename, nx, ny, nz):
    b = np.fromfile(filename, dtype=np.float64)
    b = b.reshape((3, nx, ny, nz)).transpose(1, 2, 3, 0)
    return b

from streamtracer import StreamTracer, VectorGrid
from rtmag.process.paper.metric import vector_norm, current_density

def get_xs(file_path, seeds_xs, nsteps=10000, step_size=0.1):
    files = np.load(file_path)
    b = files['b'].astype(np.float64)
    grid_spacing = [1, 1, 1]
    grid = VectorGrid(b, grid_spacing)
    tracer = StreamTracer(nsteps, step_size)
    tracer_xs = []
    tracer.trace(seeds_xs, grid)
    tracer_xs.append(tracer.xs)
    tracer_xs = [item for sublist in tracer_xs for item in sublist]
    return tracer_xs

def get_xsxs(b, seeds_xs, nsteps=10000, step_size=0.1):
    tracer = StreamTracer(nsteps, step_size)
    grid_spacing = [1, 1, 1]
    grid = VectorGrid(b.astype(np.float64), grid_spacing)
    tracer_xs = []
    tracer.trace(seeds_xs, grid)
    tracer_xs.append(tracer.xs)
    tracer_xs = [item for sublist in tracer_xs for item in sublist]
    return tracer_xs


def get_bottom(file_path):
    files = np.load(file_path)
    b = files['b'].astype(np.float64)
    bottom = b[:, :, 0, 2].T
    return bottom

def get_seeds(file_path, xypoints, seed_size=4, seed_step=2):
    files = np.load(file_path)
    x = files['x']
    y = files['y']

    dx = x[1] - x[0]  # Mm
    dy = y[1] - y[0]  # Mm
  
    seeds_xs = return_seeds(xypoints, seed_size, seed_step, x, y, dx, dy)

    return x, y, dx, dy, seeds_xs


def get_j_map(file_path):
    files = np.load(file_path)
    b = files['b']
    x = files['x']
    y = files['y']

    dx = x[1] - x[0]  # Mm
    dy = y[1] - y[0]
    dz = dy 

    dx_cm = dx * 1e8
    dy_cm = dy * 1e8
    dz_cm = dz * 1e8
    j = current_density(b, dx_cm, dy_cm, dz_cm)  # [mA/m^2]
    j_map = vector_norm(j).sum(2)                # [mA/m^2]
    j_map *= 1e-7                                # [A/cm^2]
    j_map *= dz_cm   

    bottom = j[:, :, 0, 2].T    
    return j_map, bottom 


def get_j_map_Mm(b, dx, dy, dz):
    """
    b : [Nx, Ny, Nz, 3]
    dx, dy, dz : Mm
    """

    dx_cm = dx * 1e8
    dy_cm = dy * 1e8
    dz_cm = dz * 1e8
    j = current_density(b, dx_cm, dy_cm, dz_cm)  # [mA/m^2]
    j_map = vector_norm(j).sum(2)                # [mA/m^2]
    j_map *= 1e-7                                # [A/cm^2]
    j_map *= dz_cm                               # [A/cm]

    bottom = j[:, :, 0, 2].T    
    return j_map, bottom 


def get_dxdydz(nx, ny, hmi_br_map):
    ony, onx = hmi_br_map.data.shape
    odx = 696 * 0.03 * np.pi / 180
    ody = odx

    dx = ((onx - 1)/(nx - 1))*odx
    dy = ((ony - 1)/(ny - 1))*ody
    dz = dy
    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dy, ny)
    z = y

    return x, y, z, dx, dy, dz
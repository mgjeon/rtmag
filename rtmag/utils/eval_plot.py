import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from rtmag.utils.eval import energy_np2 as energy
from rtmag.utils.eval import vector_norm_np as vector_norm
from rtmag.utils.diff import curl_np2 as curl

#-----------------------------------------------------------------------------------------

def plot_overview(b, B, z=0, b_norm=2500, ret=False):
    fig, axs = plt.subplots(2, 3, figsize=(12, 4))

    ax = axs[0]
    ax[0].imshow(b[..., z, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[0].set_title('bx')
    ax[1].imshow(b[..., z, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[1].set_title('by')
    ax[2].imshow(b[..., z, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[2].set_title('bz')

    ax = axs[1]
    ax[0].imshow(B[..., z, 0].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[0].set_title('Bx')
    ax[1].imshow(B[..., z, 1].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[1].set_title('By')
    ax[2].imshow(B[..., z, 2].transpose(), vmin=-b_norm, vmax=b_norm, cmap='gray', origin='lower')
    ax[2].set_title('Bz')

    fig.suptitle(f'z={z}')
    
    plt.tight_layout()

    if ret:
        plt.close()
        return fig
    else:
        plt.show()


def plot_s(mag, title, n_samples, v_mm=2500, ret=False):
    fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12))
    heights = np.linspace(0, 1, n_samples) ** 2 * (mag.shape[2] - 1)  # more samples from lower heights
    heights = heights.astype(np.int32)
    for i in range(3):
        for j, h in enumerate(heights):
            v_min_max = int(v_mm / (h+1))
            axs[i, j].imshow(mag[:, :, h, i].transpose(), cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                            origin='lower')
            axs[i, j].set_axis_off()
    for j, h in enumerate(heights):
        axs[0, j].set_title('%.01f' % h, fontsize=20)
    fig.tight_layout()
    fig.suptitle(title, fontsize=25)
    plt.tight_layout()

    if ret:
        plt.close()
        return fig
    else:
        plt.show()


def plot_sample(b, B, n_samples=10, v_mm=2500, ret=False):
    if ret:
        fig1 = plot_s(b, 'b', n_samples, v_mm, ret)
        fig2 = plot_s(B, 'B', n_samples, v_mm, ret)
        return fig1, fig2
    else:
        plot_s(b, 'b', n_samples, v_mm)
        plot_s(B, 'B', n_samples, v_mm)

#-----------------------------------------------------------------------------------------

def plot_validation(b, b_mm = [-2500, 2500], j_mm = [1e1, 1e3], jxb_mm = [1e2, 1e6], e_mm = [0, 1e6]):
    b_z0 = b[:, :, 0, :]

    j = curl(b)
    j_map = vector_norm(j).sum(2)

    jxb = np.cross(j, b, axis=-1)
    jxb_map = vector_norm(jxb).sum(2)

    me = energy(b)
    energy_map = me.sum(2)

    plt.figure(figsize=(10,6))
    plt.subplot(221)
    plt.imshow(b_z0[..., 2].T, origin='lower', vmin=b_mm[0], vmax=b_mm[1], cmap='gray')
    plt.title('$B_z$')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(j_map.T, origin='lower', cmap='viridis', norm=LogNorm(vmin=j_mm[0], vmax=j_mm[1]))
    plt.title('$|J|$')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(jxb_map.T, origin='lower', cmap='inferno', norm=LogNorm(vmin=jxb_mm[0], vmax=jxb_mm[1]))
    plt.title('$|J x B|$')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(energy_map.T, origin='lower', vmin=e_mm[0], vmax=e_mm[1], cmap='jet')
    plt.title('Energy')
    plt.axis('off')
    plt.show()   
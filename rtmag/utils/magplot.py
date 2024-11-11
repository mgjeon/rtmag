import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

def create_coordinates(bounds):
    xbounds = (bounds[0], bounds[1])
    ybounds = (bounds[2], bounds[3])
    zbounds = (bounds[4], bounds[5])
    meshgrid = np.mgrid[xbounds[0]:xbounds[1]+1, ybounds[0]:ybounds[1]+1, zbounds[0]:zbounds[1]+1]
    return np.stack(meshgrid, axis=-1).astype(np.float32)


def create_mesh(bx, by, bz):
    bx, by, bz = map(np.array, (bx, by, bz))
    Nx, Ny, Nz = bx.shape
    co_bounds = (0, Nx-1, 0, Ny-1, 0, Nz-1)
    co_coords = create_coordinates(co_bounds).reshape(-1, 3)
    co_coord = co_coords.reshape(Nx, Ny, Nz, 3)
    x = co_coord[..., 0]
    y = co_coord[..., 1]
    z = co_coord[..., 2]
    mesh = pv.StructuredGrid(x, y, z)
    vectors = np.stack([bx, by, bz], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
    mesh['vector'] = vectors
    mesh.active_vectors_name = 'vector'
    magnitude = np.linalg.norm(vectors, axis=-1)
    mesh['magnitude'] = magnitude
    mesh.active_scalars_name = 'magnitude'
    return mesh


def create_mesh_xyz(x, y, z, bx, by, bz):
    x, y, z, bx, by, bz = map(np.array, (x, y, z, bx, by, bz))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    mesh = pv.StructuredGrid(X, Y, Z)
    vectors = np.stack([bx, by, bz], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
    mesh['vector'] = vectors
    mesh.active_vectors_name = 'vector'
    magnitude = np.linalg.norm(vectors, axis=-1)
    mesh['magnitude'] = magnitude
    mesh.active_scalars_name = 'magnitude'
    return mesh


class mag_plotter:
    def __init__(self, grd):
        # grid = copy.deepcopy(grd)
        grid = grd
        self.grid = grid
        x_ind_min, y_ind_min, z_ind_min = 0, 0, 0
        Nx, Ny, Nz = self.grid.dimensions
        x_ind_max, y_ind_max, z_ind_max = Nx-1, Ny-1, Nz-1

        self.x_ind_min, self.y_ind_min, self.z_ind_min = x_ind_min, y_ind_min, z_ind_min
        self.x_ind_max, self.y_ind_max, self.z_ind_max = x_ind_max, y_ind_max, z_ind_max
        
        bottom_subset = (x_ind_min, x_ind_max, y_ind_min, y_ind_max, 0, 0)
        bottom = self.grid.extract_subset(bottom_subset).extract_surface()
        bottom.active_vectors_name = 'vector'
        bottom.active_scalars_name = 'magnitude'

        self.bottom = bottom

        self.x_bottom = bottom.points[:, 0].reshape(Nx, Ny)
        self.y_bottom = bottom.points[:, 1].reshape(Nx, Ny)
        self.B_bottom = bottom['vector'].reshape(Nx, Ny, 3)

        B = self.grid['vector'].reshape(Nz, Ny, Nx, 3)
        self.B = B.transpose(2, 1, 0, 3)

    def plt_Bz(self):
        plt.close()
        fig, ax = plt.subplots(figsize=(6,6))
        CS = plt.contour(self.x_bottom, self.y_bottom, self.B_bottom[:, :, 2], 
                         origin='lower', colors='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.clabel(CS, fontsize=9, inline=True)
        ax.set_title(r"$B_z(z=0)$")
        plt.show()

    def pv_bottom(self):
        p = pv.Plotter()
        p.show_bounds()
        p.add_mesh(self.grid.outline())
        p.add_mesh(self.bottom)
        p.show()

    def pv_Bz_2D(self):
        p = pv.Plotter()
        p.show_bounds()
        p.add_mesh(self.grid.outline())
        sargs = dict(title="B_z")
        ctr = self.bottom.contour(scalars=self.bottom['vector'][:, 2])
        p.add_mesh(ctr, cmap='plasma', scalar_bar_args=sargs)
        p.show()

    def pv_Bz_3D(self):
        p = pv.Plotter()
        p.show_bounds()
        p.add_mesh(self.grid.outline())
        sargs = dict(title="B_z")
        ctr = self.grid.contour(scalars=self.grid['vector'][:, 2])
        p.add_mesh(ctr, cmap='plasma', scalar_bar_args=sargs, opacity=0.5)
        p.show()

    def pv_streamline(self):
        p = pv.Plotter()
        p.show_bounds()
        p.add_mesh(self.grid.outline())
        i_size = self.grid.bounds[1]-self.grid.bounds[0]
        j_size = self.grid.bounds[3]-self.grid.bounds[2]
        seed = pv.Plane(center=(self.grid.center[0], self.grid.center[1], 0), direction=(0,0,1), 
                i_size=i_size, j_size=j_size, 
                i_resolution=10, j_resolution=10)
        p.add_mesh(seed)
        strl = self.grid.streamlines_from_source(seed,
                                                 vectors='vector',
                                                 max_time=180,
                                                 initial_step_length=0.1,
                                                 integration_direction='both')
        
        p.add_mesh(strl.tube(radius=i_size/400), cmap='bwr', ambient=0.2)

        sargs = dict(title="B_z")
        ctr = self.bottom.contour(scalars=self.bottom['vector'][:, 2])
        p.add_mesh(ctr, cmap='plasma', scalar_bar_args=sargs)
        p.show()

    def pv_streamline_Bz(self, window_size=None, title=None, title_fontsize=20, camera_position=None, i_siz=None, j_siz=None, i_resolution=10, j_resolution=10, vmin=-2000, vmax=2000):
        p = pv.Plotter()
        # p.show_bounds()
        p.add_mesh(self.grid.outline())
        sargs_B = dict(
            title='Bz [G]',
            title_font_size=15,
            height=0.25,
            width=0.05,
            vertical=True,
            position_x = 0.05,
            position_y = 0.05,
        )
        dargs_B = dict(
            scalars='vector', 
            component=2, 
            clim=(vmin, vmax), 
            scalar_bar_args=sargs_B, 
            show_scalar_bar=True, 
            lighting=False
        )
        p.add_mesh(self.bottom, cmap='gray', **dargs_B)

        if (i_siz is not None) and (j_siz is not None):
            i_size = i_siz
            j_size = j_siz
        else:
            i_size = self.grid.bounds[1]-self.grid.bounds[0]
            j_size = self.grid.bounds[3]-self.grid.bounds[2]
        seed = pv.Plane(center=(self.grid.center[0], self.grid.center[1], 0), direction=(0,0,1), 
                i_size=i_size, j_size=j_size, 
                i_resolution=i_resolution, j_resolution=j_resolution)
        # p.add_mesh(seed)
        strl = self.grid.streamlines_from_source(seed,
                                                 vectors='vector',
                                                 max_time=180,
                                                 initial_step_length=0.1,
                                                 integration_direction='both')
        
        p.add_mesh(strl.tube(radius=i_size/400), 
                   lighting=False,
                   color='blue')
        if camera_position is not None:
             p.camera_position = camera_position
        if window_size is not None:
            p.window_size = window_size
        if title is not None:
            p.add_title(title, font_size=title_fontsize)
        p.show()
        return p
    
    def create_mesh(self, i_siz=None, j_siz=None, i_resolution=10, j_resolution=10, vmin=-2000, vmax=2000, max_time=10000):
        if (i_siz is not None) and (j_siz is not None):
            i_size = i_siz
            j_size = j_siz
        else:
            i_size = self.grid.bounds[1]-self.grid.bounds[0]
            j_size = self.grid.bounds[3]-self.grid.bounds[2]
        seed = pv.Plane(center=(self.grid.center[0], self.grid.center[1], 0), direction=(0,0,1), 
                i_size=i_size, j_size=j_size, 
                i_resolution=i_resolution, j_resolution=j_resolution)
        # p.add_mesh(seed)
        strl = self.grid.streamlines_from_source(seed,
                                                 vectors='vector',
                                                 max_time=max_time,
                                                 initial_step_length=0.1,
                                                 integration_direction='both')
        
        sargs_B = dict(
            title='Bz [G]',
            title_font_size=15,
            height=0.25,
            width=0.05,
            vertical=True,
            position_x = 0.05,
            position_y = 0.05,
        )
        dargs_B = dict(
            scalars='vector', 
            component=2, 
            clim=(vmin, vmax), 
            scalar_bar_args=sargs_B, 
            lighting=False
        )

        self.tube = strl.tube(radius=i_size/400)
        self.dargs_B = dargs_B        
        return strl.tube(radius=i_size/400), self.bottom, dargs_B

    def plt_Bz_imshow(self, z=0, vmin=None, vmax=None, title=None, fontsize=20):         
        plt.close()
        fig, ax = plt.subplots(figsize=(6,6))
        if (vmin is not None) and (vmax is not None):
            CS = plt.imshow(self.B[:, :, z, 2].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        else:
            CS = plt.imshow(self.B[:, :, z, 2].transpose(), origin='lower', cmap='gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # ax.set_title(f"B_z(z={z})")
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        fig.colorbar(CS, label=r'$B_z$'+f'(z={z})')
        plt.show()

        return self.B[:, :, z, 2].transpose()
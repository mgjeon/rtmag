import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def nc_list(nc_path):
    isee_datapath = Path(nc_path)
    filelist = sorted([x for x in isee_datapath.glob('*.nc')])
    return filelist


def load_fields(nc_file):
    nc=netCDF4.Dataset(nc_file, 'r')

    nc_x=nc.variables['x']
    x=nc_x[:]
    x = np.array(x).astype(np.float32)
    nc_y=nc.variables['y']
    y=nc_y[:]
    y = np.array(y).astype(np.float32)
    nc_z=nc.variables['z']
    z=nc_z[:]
    z = np.array(z).astype(np.float32)
    
    nc_bx=nc.variables['Bx']
    bx=nc_bx[:].transpose(2,1,0)
    nc_by=nc.variables['By']
    by=nc_by[:].transpose(2,1,0)
    nc_bz=nc.variables['Bz']
    bz=nc_bz[:].transpose(2,1,0)
    b = np.stack([bx, by, bz])
    b = np.array(b).astype(np.float32)

    nc_bpx=nc.variables['Bx_pot']
    bpx=nc_bpx[:].transpose(2,1,0)
    nc_bpy=nc.variables['By_pot']
    bpy=nc_bpy[:].transpose(2,1,0)
    nc_bpz=nc.variables['Bz_pot']
    bpz=nc_bpz[:].transpose(2,1,0)
    bp = np.stack([bpx, bpy, bpz])
    bp = np.array(bp).astype(np.float32)
    return x, y, z, b, bp


def load_xyz_b_bp(nc_file):
    nc=netCDF4.Dataset(nc_file, 'r')

    nc_x=nc.variables['x']
    x=nc_x[:]
    x = np.array(x).astype(np.float32)
    nc_y=nc.variables['y']
    y=nc_y[:]
    y = np.array(y).astype(np.float32)
    nc_z=nc.variables['z']
    z=nc_z[:]
    z = np.array(z).astype(np.float32)
    
    nc_bx=nc.variables['Bx']
    bx=nc_bx[:].transpose(2,1,0)
    nc_by=nc.variables['By']
    by=nc_by[:].transpose(2,1,0)
    nc_bz=nc.variables['Bz']
    bz=nc_bz[:].transpose(2,1,0)
    b = np.stack([bx, by, bz], -1)
    b = np.array(b).astype(np.float32)

    nc_bpx=nc.variables['Bx_pot']
    bpx=nc_bpx[:].transpose(2,1,0)
    nc_bpy=nc.variables['By_pot']
    bpy=nc_bpy[:].transpose(2,1,0)
    nc_bpz=nc.variables['Bz_pot']
    bpz=nc_bpz[:].transpose(2,1,0)
    bp = np.stack([bpx, bpy, bpz], -1)
    bp = np.array(bp).astype(np.float32)
    return x, y, z, b, bp


def load_nc_b(filename):
    nc = netCDF4.Dataset(filename, 'r')

    nc_bx = nc.variables['Bx']
    bx = np.array(nc_bx[:].transpose(2, 1, 0))
    nc_by = nc.variables['By']
    by = np.array(nc_by[:].transpose(2, 1, 0))
    nc_bz = nc.variables['Bz']
    bz = np.array(nc_bz[:].transpose(2, 1, 0))

    b = np.stack([bx, by, bz], -1)

    return b 


def load_nc_bpot(filename):
    nc = netCDF4.Dataset(filename, 'r')

    nc_bx = nc.variables['Bx_pot']
    bx = np.array(nc_bx[:].transpose(2, 1, 0))
    nc_by = nc.variables['By_pot']
    by = np.array(nc_by[:].transpose(2, 1, 0))
    nc_bz = nc.variables['Bz_pot']
    bz = np.array(nc_bz[:].transpose(2, 1, 0))

    b = np.stack([bx, by, bz], -1)

    return b 

#-----------------------------------------------------------------------------------------

class nlfff:

      def __init__(self,filename):
            self.filename=filename

            nc=netCDF4.Dataset(self.filename,'r')
            self.NOAA=nc.NOAA
            self.year_month_day_time=nc.year_month_day_time
            self.project=nc.project
            self.production_date=nc.production_date
            self.version=nc.version
            self.data_doi=nc.data_doi
            self.http_link=nc.http_link
            self.Distributor=nc.Distributor
            
            nc_x=nc.variables['x']
            self.x=np.array(nc_x[:])
            print(nc_x.long_name,' unit:',nc_x.units)
            nc_y=nc.variables['y']
            self.y=np.array(nc_y[:])
            print(nc_y.long_name,' unit:',nc_y.units)
            nc_z=nc.variables['z']
            self.z=np.array(nc_z[:])
            print(nc_z.long_name,' unit:',nc_z.units)
            
            nc_bx=nc.variables['Bx']
            self.bx=np.array(nc_bx[:].transpose(2,1,0))
            print(nc_bx.long_name,' unit:',nc_bx.units)
            nc_by=nc.variables['By']
            self.by=np.array(nc_by[:].transpose(2,1,0))
            print(nc_by.long_name,' unit:',nc_by.units)
            nc_bz=nc.variables['Bz']
            self.bz=np.array(nc_bz[:].transpose(2,1,0))
            print(nc_bz.long_name,' unit:',nc_bz.units)
            
            nc_bxp=nc.variables['Bx_pot']
            self.bx_pot=np.array(nc_bxp[:].transpose(2,1,0))
            print(nc_bxp.long_name,' unit:',nc_bxp.units)
            nc_byp=nc.variables['By_pot']
            self.by_pot=np.array(nc_byp[:].transpose(2,1,0))
            print(nc_byp.long_name,' unit:',nc_byp.units)
            nc_bzp=nc.variables['Bz_pot']
            self.bz_pot=np.array(nc_bzp[:].transpose(2,1,0))
            print(nc_bzp.long_name,' unit:',nc_bzp.units)
            
      def info(self):
            print(f"NOAA",self.NOAA)
            print(f'year_month_day_time',self.year_month_day_time)
            print(f"project",self.project)
            print(f"production_date",self.production_date)
            print(f"version",self.version)
            print(f"data_doi",self.data_doi)
            print(f"http_link",self.http_link)
            print(f"Distributor",self.Distributor)

      def plot(self):
            xs=12.0
            ys=4.0

            xmin=min(self.x)
            xmax=max(self.x)
            ymin=min(self.y)
            ymax=max(self.y)

            plt.close()
            fig=plt.figure(figsize=(xs,ys))
            ax1=fig.add_axes((0.08,0.35,0.25,0.25*xs/ys*(ymax-ymin)/(xmax-xmin)))
            ax2=fig.add_axes((0.4,0.35,0.25,0.25*xs/ys*(ymax-ymin)/(xmax-xmin)))
            ax3=fig.add_axes((0.72,0.35,0.25,0.25*xs/ys*(ymax-ymin)/(xmax-xmin)))
            cax1=fig.add_axes((0.08,0.15,0.25,0.05))
            cax2=fig.add_axes((0.4,0.15,0.25,0.05))
            cax3=fig.add_axes((0.72,0.15,0.25,0.05))
            
            vmin=-3000.0 
            vmax=3000.0
            
            im1=ax1.pcolormesh(self.x,self.y,self.bx[:,:,0].transpose(),vmin=vmin,vmax=vmax,cmap='gist_gray',shading='auto')
            im2=ax2.pcolormesh(self.x,self.y,self.by[:,:,0].transpose(),vmin=vmin,vmax=vmax,cmap='gist_gray',shading='auto')
            im3=ax3.pcolormesh(self.x,self.y,self.bz[:,:,0].transpose(),vmin=vmin,vmax=vmax,cmap='gist_gray',shading='auto')

            cbar1=plt.colorbar(im1,cax=cax1,orientation='horizontal')
            cbar2=plt.colorbar(im2,cax=cax2,orientation='horizontal')
            cbar3=plt.colorbar(im3,cax=cax3,orientation='horizontal')
            
            ax1.set_title('Bx [G]')
            ax1.set_xlabel('x [Mm]')
            ax1.set_ylabel('y [Mm]')
            
            ax2.set_title('By [G]')
            ax2.set_xlabel('x [Mm]')
            ax2.set_ylabel('y [Mm]')
            
            ax3.set_title('Bz [G]')
            ax3.set_xlabel('x [Mm]')
            ax3.set_ylabel('y [Mm]')
            
            plt.pause(0.1)
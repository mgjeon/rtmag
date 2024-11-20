"""
Python class for a 3d magnetic field snapshot on a staggered Cartesian grid.

anthony.yeates@durham.ac.uk
"""

import numpy as np
from scipy.io import netcdf, FortranFile
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.integrate import ode, trapezoid
import scipy.fftpack as fp
import scipy.sparse as sp
import time
import os
 
class BField:
    """
    Magnetic field snapshot.
    """
      
    def __init__(self, filename, clean=True):
        """
            Constructor - creates BField object from netcdf file with 3D B-field arrays located
            at grid points.
        """
       
        self.codepath = './'
        self.tmpath = './'
        self.filename = filename
       
        # Compile the fortran code:
        # os.system('gfortran -o fastfl fastfl.f90 -O3 -ffast-math -funroll-loops --param max-unroll-times=5 -fopenmp')
        # os.system('gfortran -o fastflh fastflh.f90 -O3 -ffast-math -funroll-loops --param max-unroll-times=5 -fopenmp -g -fcheck=all -fbacktrace')


        # os.system('gfortran -o fastflh fastflh.f90 -O0 -ffast-math -funroll-loops --param max-unroll-times=5 -g -fcheck=all -fbacktrace')     # for debugging

        
        # Open netcdf file:
        try:
            fh = netcdf.netcdf_file(filename, 'r', mmap=False)
        except OSError:
            raise OSError('Problem with input file'+filename)
        
        # Read in coordinates:
        try:
            self.x1 = fh.variables['x'][:]
            self.y1 = fh.variables['y'][:]
            self.z1 = fh.variables['z'][:]
        except KeyError:
            raise KeyError('Problem with reading coordinates from '+filename)

        # Grid spacing:
        self.dx = np.mean(self.x1[1:] - self.x1[:-1])
        self.dy = np.mean(self.y1[1:] - self.y1[:-1])
        self.dz = np.mean(self.z1[1:] - self.z1[:-1])
        if (np.std(self.x1[1:] - self.x1[:-1]) > 1e-5 \
            or np.std(self.y1[1:] - self.y1[:-1]) > 1e-5 \
            or np.std(self.z1[1:] - self.z1[:-1]) > 1e-5):
            raise Exception('Grid is not uniformly spaced in '+filename)

        # Grid sizes:
        self.nx = np.size(self.x1, 0)
        self.ny = np.size(self.y1, 0)
        self.nz = np.size(self.z1, 0)

        # Generate cell-centre coordinate arrays, including ghost cells:
        self.xc = np.linspace(self.x1[0]-0.5*self.dx, self.x1[-1]+0.5*self.dx, self.nx+1)
        self.yc = np.linspace(self.y1[0]-0.5*self.dy, self.y1[-1]+0.5*self.dy, self.ny+1)
        self.zc = np.linspace(self.z1[0]-0.5*self.dz, self.z1[-1]+0.5*self.dz, self.nz+1)

        # Read in magnetic field at grid points and generate interpolators:
        try:
            bx1 = np.swapaxes(fh.variables['bx'][:],0,2)
            self.bxs = rgi((self.x1,self.y1,self.z1), bx1, bounds_error=False, fill_value=0)
            del(bx1)
            by1 = np.swapaxes(fh.variables['by'][:],0,2)
            self.bys = rgi((self.x1,self.y1,self.z1), by1, bounds_error=False, fill_value=0)
            del(by1)            
            bz1 = np.swapaxes(fh.variables['bz'][:],0,2)
            self.bzs = rgi((self.x1,self.y1,self.z1), bz1, bounds_error=False, fill_value=0)
            del(bz1)              
        except KeyError:
            raise KeyError('Problem with reading B from '+filename)

        fh.close()

        # Compute vector potential:
        self.computeADeVore()
        
        # Re-compute B at cell centres so that it matches perfectly:
        if (clean):
            del(self.bxs, self.bys, self.bzs)
            self.computeBfromA()
  
        # Save to file (for fortran tracer):
        fid = FortranFile(self.tmpath+'bc.unf', 'w')
        fid.write_record(np.array([self.nx], dtype=np.int32))
        fid.write_record(np.array([self.ny], dtype=np.int32))
        fid.write_record(np.array([self.nz], dtype=np.int32))
        fid.write_record(self.x1.astype(np.float64))
        fid.write_record(self.y1.astype(np.float64))
        fid.write_record(self.z1.astype(np.float64))
        fid.write_record(self.xc.astype(np.float64))
        fid.write_record(self.yc.astype(np.float64))
        fid.write_record(self.zc.astype(np.float64))
        x, y, z = np.meshgrid(self.x1, self.yc, self.zc, indexing='ij')
        bx = self.bx(x, y, z)
        fid.write_record(np.swapaxes(bx.astype(np.float64), 0, 2))
        del(x, y, z, bx)
        x, y, z = np.meshgrid(self.xc, self.y1, self.zc, indexing='ij')
        by = self.by(x, y, z)
        fid.write_record(np.swapaxes(by.astype(np.float64), 0, 2))
        del(x, y, z, by)
        x, y, z = np.meshgrid(self.xc, self.yc, self.z1, indexing='ij')
        bz = self.bz(x, y, z)
        fid.write_record(np.swapaxes(bz.astype(np.float64), 0, 2))
        del(x, y, z, bz)        
        fid.close()
 
 
    def bx(self, x, y, z):
        """
            Evaluate Bx at given point(s).
            Purpose of this routine is to provide an interface to bxs for general shapes
            of coordinate arrays.
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.bxs(xx)
 
 
    def by(self, x, y, z):
        """
            Evaluate By at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.bys(xx)
    
    
    def bz(self, x, y, z):
        """
            Evaluate Bz at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.bzs(xx)        


    def ax(self, x, y, z):
        """
            Evaluate Ax at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.axs(xx)
 
 
    def ay(self, x, y, z):
        """
            Evaluate Ay at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.ays(xx)
    
    
    def az(self, x, y, z):
        """
            Evaluate Az at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.azs(xx)  
 
 
    def bpx(self, x, y, z):
        """
            Evaluate Bpx at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.bpxs(xx)
 
 
    def bpy(self, x, y, z):
        """
            Evaluate Bpy at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.bpys(xx)
    
    
    def bpz(self, x, y, z):
        """
            Evaluate Bpz at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.bpzs(xx) 


    def apx(self, x, y, z):
        """
            Evaluate Apx at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.apxs(xx)
 
 
    def apy(self, x, y, z):
        """
            Evaluate Apy at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.apys(xx)
    
    
    def apz(self, x, y, z):
        """
            Evaluate Apz at given point(s).
        """
        xx = np.stack((x, y, z), axis=len(np.shape(x)))
        return self.apzs(xx) 
    

    def computeA0Coulomb(self, potential=False):
        """
            Compute vector potential a0 on lower boundary **ribs** using Coulomb
            gauge where
                ax0 = -du/dy,    ay0 = du/dx
                d^2u/dx^2 + d^2u/dy^2 = bz
            Solve for the "non-monopole" modes with periodic fft, then add monopole
            component back separately.
            Here bz is the face-centre interpolated value.
        """

        # Evaluate bz on 2d grid at face centres (no ghost cells):
        x2, y2 = np.meshgrid(self.xc[1:-1], self.yc[1:-1], indexing='ij')
        if (potential):
            bz = self.bpz(x2, y2, x2*0 + self.z1[0])
        else:
            bz = self.bz(x2, y2, x2*0 + self.z1[0])
        
        # Find monopole component:
        b0 = np.mean(bz)
        
        # Solve Poisson equation for u using fft:
        # - compute 2d fft of rhs (assumes periodic boundary conditions):       
        u = fp.fft2(bz)
       
        # - solve in Fourier space (5-point stencil) - remove monopole term:
        nx, ny = self.nx, self.ny
        m, n = np.mgrid[0:nx-1, 0:ny-1]
        m, n = np.cos(2*m*np.pi/(nx-1)), np.cos(2*n*np.pi/(ny-1))
        u /= 2.0*((m - 1.0)/(self.dx)**2 + (n - 1.0)/(self.dy)**2) + 1e-16
        u[0,0] = 0

        # - invert fft:
        u = np.real(fp.ifft2(u))
        
        # - boundary values by periodicity:
        ub = np.zeros((nx+1,ny+1))
        ub[1:-1,1:-1] = u
        ub[0,:] = ub[-2,:]
        ub[-1,:] = ub[1,:]
        ub[:,0] = ub[:,-2]
        ub[:,-1] = ub[:,1]

        # Compute ax0 and ay0:
        a0x = -(ub[1:-1,1:] - ub[1:-1,:-1])/self.dy
        a0y = (ub[1:,1:-1] - ub[:-1,1:-1])/self.dx

        # Add monopole component separately:
        _, y2 = np.meshgrid(self.xc[1:-1], self.y1, indexing='ij')
        a0x -= 0.5*y2*b0
        x2, _ = np.meshgrid(self.x1, self.yc[1:-1], indexing='ij')
        a0y += 0.5*x2*b0

        return a0x, a0y, bz


    def computeADeVore(self, potential=False):
        """
            Compute vector potential on ribs, using DeVore gauge:
                ax = int_z0^z by dz + a0x
                ay = -int_z0^z bx dz + a0y
            If keyword "potential" is True, then do this for the potential field Bp.
        """

        if (potential):
            print('Computing Ap...')
            afile = 'ape.unf'
        else:
            print('Computing A...')
            afile = 'ae.unf'

        # Compute vector potential on lower boundary (ribs):
        a0x, a0y, _ = self.computeA0Coulomb(potential=potential)

        # Compute Ax:
        ax = np.zeros((self.nx+1, self.ny, self.nz))
        # - evaluate by on y cell-faces (interior):
        x3, y3, z3 = np.meshgrid(self.xc[1:-1], self.y1, self.zc[1:-1], indexing='ij')
        if (potential):
            by = self.bpy(x3, y3, z3)
        else:
            by = self.by(x3, y3, z3)
        # - compute ax on interior by summation:
        ax[1:-1,:,:] = np.repeat(np.expand_dims(a0x, axis=-1), self.nz, axis=2)
        ax[1:-1,:,1:] += np.cumsum(by, axis=2)*self.dz
        # - fill boundary values by zero-gradient:
        ax[0,:,:] = ax[1,:,:]
        ax[-1,:,:] = ax[-2,:,:]
        # - make interpolator:
        if (potential):
            self.apxs = rgi((self.xc, self.y1, self.z1), ax, bounds_error=False, fill_value=0)
        else:
            self.axs = rgi((self.xc, self.y1, self.z1), ax, bounds_error=False, fill_value=0)
        # - save to fortran file:
        fid = FortranFile(self.tmpath+afile, 'w')
        fid.write_record(np.swapaxes(ax.astype(np.float64), 0, 2))
        del(ax)

        # Compute Ay:
        ay = np.zeros((self.nx, self.ny+1, self.nz))        
        # - evaluate bx on x cell-faces (interior):
        x3, y3, z3 = np.meshgrid(self.x1, self.yc[1:-1], self.zc[1:-1], indexing='ij')
        if (potential):
            bx = self.bpx(x3, y3, z3)
        else:
            bx = self.bx(x3, y3, z3)
        # - compute ay on interior by summation:
        ay[:,1:-1,:] = np.repeat(np.expand_dims(a0y, axis=-1), self.nz, axis=2)
        ay[:,1:-1,1:] -= np.cumsum(bx, axis=2)*self.dz
        # - fill boundary values by zero-gradient:
        ay[:,0,:] = ay[:,1,:]
        ay[:,-1,:] = ay[:,-2,:]
        # - make interpolator:
        if (potential):
            self.apys = rgi((self.x1, self.yc, self.z1), ay, bounds_error=False, fill_value=0)
        else:
            self.ays = rgi((self.x1, self.yc, self.z1), ay, bounds_error=False, fill_value=0)
        # - save to fortran file:
        fid.write_record(np.swapaxes(ay.astype(np.float64), 0, 2))
        del(ay)

        # Compute Az (just zero):
        az = np.zeros((self.nx, self.ny, self.nz+1))
        # - make interpolator:
        if (potential):
            self.apzs = rgi((self.x1, self.y1, self.zc), az, bounds_error=False, fill_value=0)
        else:
            self.azs = rgi((self.x1, self.y1, self.zc), az, bounds_error=False, fill_value=0)
        # - save to fortran file:       
        fid.write_record(np.swapaxes(az.astype(np.float64), 0, 2))
        fid.close()
        del(az)       

    def computeBfromA(self):
        """
            Re-compute B at cell centres by curling A on ribs numerically, so that they
            match perfectly. (This is "divergence cleaning".)
        """
      
        # Get A on grid:
        x3, y3, z3 = np.meshgrid(self.xc[1:-1], self.y1, self.z1, indexing='ij')
        ax = self.ax(x3, y3, z3) 
        x3, y3, z3 = np.meshgrid(self.x1, self.yc[1:-1], self.z1, indexing='ij')
        ay = self.ay(x3, y3, z3)  
        x3, y3, z3 = np.meshgrid(self.x1, self.y1, self.zc[1:-1], indexing='ij')
        az = self.az(x3, y3, z3)
        del(x3, y3, z3)
        
        # Declare B arrays at face centres:
        bx = np.zeros((self.nx, self.ny+1, self.nz+1))
        by = np.zeros((self.nx+1, self.ny, self.nz+1))
        bz = np.zeros((self.nx+1, self.ny+1, self.nz))
     
        # Compute B on interior:
        bx[:,1:-1,1:-1] = (az[:,1:,:] - az[:,:-1,:])/self.dy - (ay[:,:,1:] - ay[:,:,:-1])/self.dz
        by[1:-1,:,1:-1] = (ax[:,:,1:] - ax[:,:,:-1])/self.dz - (az[1:,:,:] - az[:-1,:,:])/self.dx
        bz[1:-1,1:-1,:] = (ay[1:,:,:] - ay[:-1,:,:])/self.dx - (ax[:,1:,:] - ax[:,:-1,:])/self.dy
        del(ax, ay, az)
        
        # Add boundary points (jxn = 0): 
        bx, by, bz = self.ghostB(bx, by, bz)     
        
        # Create new interpolators:
        self.bxs = rgi((self.x1, self.yc, self.zc), bx, bounds_error=False, fill_value=0)
        self.bys = rgi((self.xc, self.y1, self.zc), by, bounds_error=False, fill_value=0)
        self.bzs = rgi((self.xc, self.yc, self.z1), bz, bounds_error=False, fill_value=0)
        del(bx, by, bz)
        
        
    def ghostB(self, bx, by, bz):
        """
            Compute ghost cell values of B on face centres, using condition that j x n = 0 on boundary.
            These ghost values are needed for interpolation, once B is on the staggered grid.
        """       
        
        # jx = 0:
        bz[:,0,:] = bz[:,1,:] - self.dy*(by[:,0,1:] - by[:,0,:-1])/self.dz
        bz[:,-1,:] = bz[:,-2,:] + self.dy*(by[:,-1,1:] - by[:,-1,:-1])/self.dz
        by[:,:,0] = by[:,:,1] - self.dz*(bz[:,1:,0] - bz[:,:-1,0])/self.dy
        by[:,:,-1] = by[:,:,-2] + self.dz*(bz[:,1:,-1] - bz[:,:-1,-1])/self.dy
        # jy = 0:
        bz[0,:,:] = bz[1,:,:] - self.dx*(bx[0,:,1:] - bx[0,:,:-1])/self.dz
        bz[-1,:,:] = bz[-2,:,:] + self.dx*(bx[-1,:,1:] - bx[-1,:,:-1])/self.dz
        bx[:,:,0] = bx[:,:,1] - self.dz*(bz[1:,:,0] - bz[:-1,:,0])/self.dx
        bx[:,:,-1] = bx[:,:,-2] + self.dz*(bz[1:,:,-1] - bz[:-1,:,-1])/self.dx
        # jz = 0:
        bx[:,0,:] = bx[:,1,:] - self.dy*(by[1:,0,:] - by[:-1,0,:])/self.dx
        bx[:,-1,:] = bx[:,-2,:] + self.dy*(by[1:,-1,:] - by[:-1,-1,:])/self.dx
        by[0,:,:] = by[1,:,:] - self.dx*(bx[0,1:,:] - bx[0,:-1,:])/self.dy
        by[-1,:,:] = by[-2,:,:] + self.dx*(bx[-1,1:,:] - bx[-1,:-1,:])/self.dy  

        return bx, by, bz

    
    def trace(self, x0, potential=False):
        """
            Trace the fieldline(s) starting from x0 by calling fortran fastfl.f90 code.
            Use this for plotting field lines.
        """
       
        if potential:
            bfile = 'bpc.unf'
        else:
            bfile = 'bc.unf'
    
        nl = np.size(x0, axis=0)
        print('Tracing %i field lines...' % nl)

        # Break x0 into manageable chunks and send to fortran separately:
        nchnk = 1000
        
        xl = []
        
        j = 0
        while (j < nl):
            if (j+nchnk > nl):
                nchnk = nl-j
            fid = FortranFile(self.tmpath+'x0.unf', 'w')
            fid.write_record(np.array([nchnk], dtype=np.int32))
            fid.write_record(x0[j:j+nchnk,:].astype(np.float64).T)
            fid.close()
            
            # Trace field lines using fortran:
            os.system(self.codepath+'fastfl '+self.tmpath+' '+bfile)

            # Read in results and return:
            fid = FortranFile(self.tmpath+'xl.unf', 'r')
            nmax = fid.read_ints(dtype=np.int32)[0]
            xl0 = fid.read_reals(dtype=np.float64).reshape((3,nmax,nchnk))
            xl0 = np.swapaxes(xl0, 0, 2)
            stat0 = fid.read_ints(dtype=np.int32)
            fid.close()
            
            if (np.sum(stat0!=1) > 0):
                print('Number of bad field lines: %i' % np.sum(stat0!=1))

            # Remove temporary files:
            os.system('rm -f '+self.tmpath+'x0.unf')
            os.system('rm -f '+self.tmpath+'xl.unf')
        
            # Extract meaningful points into list of field lines:
            for i in range(nchnk):
                xl0i = xl0[i,:,0]
                ind = (xl0[i,:,0] >= self.x1[0]) & (xl0[i,:,0] <= self.x1[-1]) \
                    & (xl0[i,:,1] >= self.y1[0]) & (xl0[i,:,1] <= self.y1[-1]) \
                    & (xl0[i,:,2] >= self.z1[0]) & (xl0[i,:,2] <= self.z1[-1])
                xl.append(xl0[i,ind,:])
        
            j += nchnk
        
        return(xl)
    
    
    def flHelicity(self, x0, hmax=1, maxerror=1e-1, potential=False, badvalue=0):
        """
            Compute field-line helicity from startpoints x0 by calling fortran fastflh.f90 code.
            Parameter hmax (real) is maximum step-size, in fraction of a cell.
        """
       
        if potential:
            bfile = 'bpc.unf'
            afile = 'ape.unf'
        else:
            bfile = 'bc.unf'
            afile = 'ae.unf'
    
        nl = np.size(x0, axis=0)
        print('Tracing %i field lines...' % nl)

        # Save startpoints to fortran file:
        fid = FortranFile(self.tmpath+'x0.unf', 'w')
        fid.write_record(np.array([nl], dtype=np.int32))
        fid.write_record(x0.astype(np.float64).T)
        fid.close()
        
        # Compute FLH using fortran:
        os.system(self.codepath+'fastflh '+self.tmpath+' '+bfile+' '+afile+(' %f %f' % (hmax, maxerror)))

        # Read in results:
        fid = FortranFile(self.tmpath+'flh.unf', 'r')
        flh = fid.read_reals(dtype=np.float64)
        stat = fid.read_ints(dtype=np.int32)
        fid.close()
        
        # Set field lines where tracing failed to "badvalue":
        flh[stat!=1] = badvalue
        if (np.sum(stat!=1) > 0):
            print('Number of bad field lines: %i' % np.sum(stat!=1))

        # Remove temporary files:
        os.system('rm -f '+self.tmpath+'x0.unf')
        os.system('rm -f '+self.tmpath+'flh.unf')
        
        return(flh)   
    
    
    def computePotentialField(self):
        """
            Compute potential field in the box that matches B.n at face centres on all six
            boundaries.
            The potential is located at the 3D cube centres.
            Creates new interpolators for Bp.
        """
        
        print('Computing potential field...')

        # Grid spacing (uniform):
        dx = self.dx; dy = self.dy; dz = self.dz

        # Number of grid points:
        nx = self.nx; ny = self.ny; nz = self.nz

        # Get Neumann boundary conditions:     
        y2, z2 = np.meshgrid(self.yc[1:-1], self.zc[1:-1], indexing='ij')
        bx0 = self.bx(y2*0 + self.x1[0], y2, z2)
        bx1 = self.bx(y2*0 + self.x1[-1], y2, z2)
        x2, z2 = np.meshgrid(self.xc[1:-1], self.zc[1:-1], indexing='ij')
        by0 = self.by(x2, x2*0 + self.y1[0], z2)
        by1 = self.by(x2, x2*0 + self.y1[-1], z2)
        x2, y2 = np.meshgrid(self.xc[1:-1], self.yc[1:-1], indexing='ij')
        bz0 = self.bz(x2, y2, x2*0 + self.z1[0])
        bz1 = self.bz(x2, y2, x2*0 + self.z1[-1])
        
        # Remove any flux imbalance from the boundary conditions:
        # (necessary to solve Neumann problem, but will change B on boundary)
        imbalance = np.sum(bx0)*dy*dz - np.sum(bx1)*dy*dz + np.sum(by0)*dx*dz
        imbalance = imbalance - np.sum(by1)*dx*dz + np.sum(bz0)*dx*dy - np.sum(bz1)*dx*dy
        surfarea = 2*(ny-1)*(nz-1)*dy*dz + 2*(nx-1)*(nz-1)*dx*dz + 2*(nx-1)*(ny-1)*dx*dy
        bx0 -= imbalance/surfarea
        bx1 += imbalance/surfarea
        by0 -= imbalance/surfarea
        by1 += imbalance/surfarea
        bz0 -= imbalance/surfarea
        bz1 += imbalance/surfarea
         
        u = np.zeros((nx-1, ny-1, nz-1))
        u[0,:,:] = bx0/dx
        u[-1,:,:] -= bx1/dx
        u[:,0,:] += by0/dy
        u[:,-1,:] -= by1/dy
        u[:,:,0] += bz0/dz
        u[:,:,-1] -= bz1/dz
        
        # Compute DCT in each direction [Neumann boundaries]:
        u = fp.dct(u, axis=0, type=2, norm='ortho')
        u = fp.dct(u, axis=1, type=2, norm='ortho')
        u = fp.dct(u, axis=2, type=2, norm='ortho')
        
        # Solve in Fourier space:
        m, n, p = np.mgrid[0:nx-1, 0:ny-1, 0:nz-1]
        m = (2*np.cos(m*np.pi/(nx-1)) - 2)/dx**2
        n = (2*np.cos(n*np.pi/(ny-1)) - 2)/dy**2
        p = (2*np.cos(p*np.pi/(nz-1)) - 2)/dz**2
        u = u/(m + n + p + 1e-16)
        
        # Invert DCT:
        u = fp.idct(u, axis=2, type=2, norm='ortho')
        u = fp.idct(u, axis=1, type=2, norm='ortho')
        u = fp.idct(u, axis=0, type=2, norm='ortho')

        # Add boundary conditions:
        ub = np.zeros((nx+1, ny+1, nz+1))
        ub[1:-1,1:-1,1:-1] = u
        ub[0,1:-1,1:-1] = ub[1,1:-1,1:-1] - bx0*dx
        ub[-1,1:-1,1:-1] += ub[-2,1:-1,1:-1] + bx1*dx
        ub[1:-1,0,1:-1] += ub[1:-1,1,1:-1] - by0*dy
        ub[1:-1,-1,1:-1] += ub[1:-1,-2,1:-1] + by1*dy
        ub[1:-1,1:-1,0] += ub[1:-1,1:-1,1] - bz0*dz
        ub[1:-1,1:-1,-1] += ub[1:-1,1:-1,-2] + bz1*dz
        del(u)
        
        # Declare Bp arrays at face centres:
        bpx = np.zeros((nx, ny+1, nz+1))
        bpy = np.zeros((nx+1, ny, nz+1))
        bpz = np.zeros((nx+1, ny+1, nz))
        
	# Compute Bp on cell faces by taking the gradient of ub:
        bpx[:,1:-1,1:-1] = (ub[1:,1:-1,1:-1] - ub[:-1,1:-1,1:-1])/dx
        bpy[1:-1,:,1:-1] = (ub[1:-1,1:,1:-1] - ub[1:-1,:-1,1:-1])/dy
        bpz[1:-1,1:-1,:] = (ub[1:-1,1:-1,1:] - ub[1:-1,1:-1,:-1])/dz
        del(ub)
       
        # Add boundary points (jxn = 0): 
        bpx, bpy, bpz = self.ghostB(bpx, bpy, bpz)
        
        # Create new interpolators:
        self.bpxs = rgi((self.x1, self.yc, self.zc), bpx, bounds_error=False, fill_value=0)
        self.bpys = rgi((self.xc, self.y1, self.zc), bpy, bounds_error=False, fill_value=0)
        self.bpzs = rgi((self.xc, self.yc, self.z1), bpz, bounds_error=False, fill_value=0)
        
        # Save to file (for fortran tracer):
        fid = FortranFile(self.tmpath+'bpc.unf', 'w')
        fid.write_record(np.array([self.nx], dtype=np.int32))
        fid.write_record(np.array([self.ny], dtype=np.int32))
        fid.write_record(np.array([self.nz], dtype=np.int32))
        fid.write_record(self.x1.astype(np.float64))
        fid.write_record(self.y1.astype(np.float64))
        fid.write_record(self.z1.astype(np.float64))
        fid.write_record(self.xc.astype(np.float64))
        fid.write_record(self.yc.astype(np.float64))
        fid.write_record(self.zc.astype(np.float64))
        fid.write_record(np.swapaxes(bpx.astype(np.float64), 0, 2))
        fid.write_record(np.swapaxes(bpy.astype(np.float64), 0, 2))
        fid.write_record(np.swapaxes(bpz.astype(np.float64), 0, 2))
        del(bpx, bpy, bpz)        
        fid.close()


    def poisson2dNeumann(self, rhs, ul, ur, ub, ut, dx, dy):
        """
            Solve Poisson equation on 2d grid with Neumann boundary conditions, using
            fast-Poisson solver (2nd order stencil).
            Boundary conditions are 1D arrays of n.grad(u) for ul[eft], ur[ight], ub[ottom], ut[op]
            Assumes that the variable u is located at grid points, including on the boundaries.
        """
        
        nx = np.size(rhs, axis=0)
        ny = np.size(rhs, axis=1)
    
        # Apply Neumann boundary conditions:
        rhs[0,:] -= 2*ul/dx
        rhs[-1,:] -= 2*ur/dx
        rhs[:,0] -= 2*ub/dy
        rhs[:,-1] -= 2*ut/dy
    
        # Fourier transform:
        u = fp.dct(rhs, axis=0, type=1)
        u = fp.dct(u, axis=1, type=1)
    
        # Solve in Fourier space:
        m, n = np.mgrid[0:nx, 0:ny]
        m = (2*np.cos(m*np.pi/(nx-1)) - 2)/dx**2
        n = (2*np.cos(n*np.pi/(ny-1)) - 2)/dy**2
        u /= m + n + 1e-16
        u[0,0] = 0

        # Invert DCT:
        u = fp.dct(u, axis=1, type=1)/2/(ny-1)
        u = fp.dct(u, axis=0, type=1)/2/(nx-1)
    
        return u


    def matchPotentialGauge(self):
        """
            Modify the vector potential so that Axn matches the potential field on all boundaries.
            Do this by changing A to A + grad(phi) where phi is located at grid points.
            First, find phi on each boundary face by solving
                laplace(phi) = div_2d(Ap - A)   with n.grad(phi) = n.(Ap - A)
            Then add constants to five of the faces so that phi is continuous at the edges.
            Finally, extend phi to the interior by solving the 3D Laplace equation (Dirichlet).
        """
        
        print('Matching Axn to Apxn on boundary...')
    
        dx = self.dx; dy = self.dy; dz = self.dz
        nx = self.nx; ny = self.ny; nz = self.nz
    
        # Solve Poisson equation for phi on each boundary (up to constant):
    
        # x = xmin
        y2, z2 = np.meshgrid(self.yc, self.z1, indexing='ij')
        day = self.apy(y2*0 + self.x1[0], y2, z2) - self.ay(y2*0 + self.x1[0], y2, z2)
        y21, z21 = np.meshgrid(self.y1, self.zc, indexing='ij')
        daz = self.apz(y21*0 + self.x1[0], y21, z21) - self.az(y21*0 + self.x1[0], y21, z21)
        rhs = (day[1:,:] - day[:-1,:])/dy + (daz[:,1:] - daz[:,:-1])/dz
        phi_xmin = self.poisson2dNeumann(rhs, -0.5*(day[0,:] + day[1,:]), 0.5*(day[-2,:] + day[-1,:]), \
                            -0.5*(daz[:,0] + daz[:,1]), 0.5*(daz[:,-2] + daz[:,-1]), dy, dz)
        # x = xmax
        day = self.apy(y2*0 + self.x1[-1], y2, z2) - self.ay(y2*0 + self.x1[-1], y2, z2)
        daz = self.apz(y21*0 + self.x1[-1], y21, z21) - self.az(y21*0 + self.x1[-1], y21, z21)
        rhs = (day[1:,:] - day[:-1,:])/dy + (daz[:,1:] - daz[:,:-1])/dz
        phi_xmax = self.poisson2dNeumann(rhs, -0.5*(day[0,:] + day[1,:]), 0.5*(day[-2,:] + day[-1,:]), \
                            -0.5*(daz[:,0] + daz[:,1]), 0.5*(daz[:,-2] + daz[:,-1]), dy, dz)
        # y = ymin
        x2, z2 = np.meshgrid(self.xc, self.z1, indexing='ij')
        dax = self.apx(x2, x2*0+self.y1[0], z2) - self.ax(x2, x2*0+self.y1[0], z2)
        x21, z21 = np.meshgrid(self.x1, self.zc, indexing='ij')
        daz = self.apz(x21, x21*0+self.y1[0], z21) - self.az(x21, x21*0+self.y1[0], z21)
        rhs = (dax[1:,:] - dax[:-1,:])/dx + (daz[:,1:] - daz[:,:-1])/dz
        phi_ymin = self.poisson2dNeumann(rhs, -0.5*(dax[0,:] + dax[1,:]), 0.5*(dax[-2,:] + dax[-1,:]), \
                            -0.5*(daz[:,0] + daz[:,1]), 0.5*(daz[:,-2] + daz[:,-1]), dx, dz)
        # y = ymax
        dax = self.apx(x2, x2*0+self.y1[-1], z2) - self.ax(x2, x2*0+self.y1[-1], z2)
        daz = self.apz(x21, x21*0+self.y1[-1], z21) - self.az(x21, x21*0+self.y1[-1], z21)
        rhs = (dax[1:,:] - dax[:-1,:])/dx + (daz[:,1:] - daz[:,:-1])/dz
        phi_ymax = self.poisson2dNeumann(rhs, -0.5*(dax[0,:] + dax[1,:]), 0.5*(dax[-2,:] + dax[-1,:]), \
                            -0.5*(daz[:,0] + daz[:,1]), 0.5*(daz[:,-2] + daz[:,-1]), dx, dz)
        # z = zmin
        x2, y2 = np.meshgrid(self.xc, self.y1, indexing='ij')
        dax = self.apx(x2, y2, x2*0+self.z1[0]) - self.ax(x2, y2, x2*0+self.z1[0])
        x21, y21 = np.meshgrid(self.x1, self.yc, indexing='ij')
        day = self.apy(x21, y21, x21*0+self.z1[0]) - self.ay(x21, y21, x21*0+self.z1[0])
        rhs = (dax[1:,:] - dax[:-1,:])/dx + (day[:,1:] - day[:,:-1])/dy
        phi_zmin = self.poisson2dNeumann(rhs, -0.5*(dax[0,:] + dax[1,:]), 0.5*(dax[-2,:] + dax[-1,:]), \
                            -0.5*(day[:,0] + day[:,1]), 0.5*(day[:,-2] + day[:,-1]), dx, dy)
        # z = zmax
        x2, y2 = np.meshgrid(self.xc, self.y1, indexing='ij')
        x21, y21 = np.meshgrid(self.x1, self.yc, indexing='ij')
        dax = self.apx(x2, y2, x2*0+self.z1[-1]) - self.ax(x2, y2, x2*0+self.z1[-1])
        day = self.apy(x21, y21, x21*0+self.z1[-1]) - self.ay(x21, y21, x21*0+self.z1[-1])
        rhs = (dax[1:,:] - dax[:-1,:])/dx + (day[:,1:] - day[:,:-1])/dy
        phi_zmax = self.poisson2dNeumann(rhs, -0.5*(dax[0,:] + dax[1,:]), 0.5*(dax[-2,:] + dax[-1,:]), \
                               -0.5*(day[:,0] + day[:,1]), 0.5*(day[:,-2] + day[:,-1]), dx, dy)

        # Add constants to phi on five of the faces so that it is continuous on all edges:
        phi_xmin += np.mean(phi_zmax[0,:]) - np.mean(phi_xmin[:,-1])
        phi_xmax += np.mean(phi_zmax[-1,:]) - np.mean(phi_xmax[:,-1])
        phi_ymin += np.mean(phi_zmax[:,0]) - np.mean(phi_ymin[:,-1])
        phi_ymax += np.mean(phi_zmax[:,-1]) - np.mean(phi_ymax[:,-1])
        phi_zmin += 0.25*(np.mean(phi_xmin[:,0]) + np.mean(phi_xmax[:,0]) \
                        + np.mean(phi_ymin[:,0]) + np.mean(phi_ymax[:,0])) \
                    - 0.25*(np.mean(phi_zmin[:,0]) + np.mean(phi_zmin[:,-1]) \
                        + np.mean(phi_zmin[0,:]) + np.mean(phi_zmin[-1,:]))
    
        # Extend phi to interior by 3D Dirichlet solution of Laplace equation:
        u = np.zeros((nx-2, ny-2, nz-2))
        u[0,:,:] -= phi_xmin[1:-1,1:-1]/dx**2
        u[-1,:,:] -= phi_xmax[1:-1,1:-1]/dx**2
        u[:,0,:] -= phi_ymin[1:-1,1:-1]/dy**2
        u[:,-1,:] -= phi_ymax[1:-1,1:-1]/dy**2
        u[:,:,0] -= phi_zmin[1:-1,1:-1]/dz**2
        u[:,:,-1] -= phi_zmax[1:-1,1:-1]/dz**2
    
        u = fp.dst(u, axis=0, type=1)
        u = fp.dst(u, axis=1, type=1)
        u = fp.dst(u, axis=2, type=1)
        
        m, p, q = np.mgrid[1:nx-1, 1:ny-1 ,1:nz-1]
        m = (2*np.cos(m*np.pi/nx) - 2)/dx**2
        p = (2*np.cos(p*np.pi/ny) - 2)/dy**2
        q = (2*np.cos(q*np.pi/nz) - 2)/dz**2
        u /= m + p + q
        
        u = fp.dst(u, axis=2, type=1)/2/(nz-1)
        u = fp.dst(u, axis=1, type=1)/2/(ny-1)
        u = fp.dst(u, axis=0, type=1)/2/(nx-1)
        
        phi = np.zeros((nx, ny, nz))
        phi[1:-1,1:-1,1:-1] = u
        phi[0,:,:] = phi_xmin
        phi[-1,:,:] = phi_xmax
        phi[:,0,:] = phi_ymin
        phi[:,-1,:] = phi_ymax
        phi[:,:,0] = phi_zmin
        phi[:,:,-1] = phi_zmax
        
        # Change gauge of vector potential:
        fid = FortranFile(self.tmpath+'ae.unf', 'w')       
        x3, y3, z3 = np.meshgrid(self.xc, self.y1, self.z1, indexing='ij')
        ax = self.ax(x3, y3, z3)
        ax[1:-1,:,:] += (phi[1:,:,:] - phi[:-1,:,:])/dx
        ax[0,:,:] = ax[1,:,:]
        ax[-1,:,:] = ax[-2,:,:]
        fid.write_record(np.swapaxes(ax.astype(np.float64), 0, 2))
        self.axs = rgi((self.xc, self.y1, self.z1), ax)
        del(ax)
        x3, y3, z3 = np.meshgrid(self.x1, self.yc, self.z1, indexing='ij')
        ay = self.ay(x3, y3, z3)
        ay[:,1:-1,:] += (phi[:,1:,:] - phi[:,:-1,:])/dy
        ay[:,0,:] = ay[:,1,:]
        ay[:,-1,:] = ay[:,-2,:]
        fid.write_record(np.swapaxes(ay.astype(np.float64), 0, 2))        
        self.ays = rgi((self.x1, self.yc, self.z1), ay)
        del(ay)
        x3, y3, z3 = np.meshgrid(self.x1, self.y1, self.zc, indexing='ij')
        az = self.az(x3, y3, z3)
        az[:,:,1:-1] += (phi[:,:,1:] - phi[:,:,:-1])/dz
        az[:,:,0] = az[:,:,1]
        az[:,:,-1] = az[:,:,-2]
        fid.write_record(np.swapaxes(az.astype(np.float64), 0, 2))                
        self.azs = rgi((self.x1, self.y1, self.zc), az)
        del(az)
        fid.close()

        return phi


    def lapMatrix(self):
        """
            Construct finite-difference matrix for surface laplacian.
            
            Uses sparse matrices: lil_matrix format for speed of construction, but then
            converted to csc_matrix format for output (for ease of future use).
        """
        # - determine size of matrix. Include y & z edges and all vertices in x faces.
        #   Include x edges in y faces.
        nx, ny, nz = self.nx, self.ny, self.nz
        n = 2*ny*nz + 2*(nx-2)*nz + 2*(nx-2)*(ny-2)
               
        dx, dy, dz = self.dx, self.dy, self.dz

        # Create index array for each face:
        ix0 = np.zeros((ny+2, nz+2), dtype=np.int32)
        ix0[1:-1,1:-1] = np.reshape(np.arange(ny*nz), (ny, nz))
        k = ny*nz
        ix1 = np.zeros((ny+2, nz+2), dtype=np.int32)
        ix1[1:-1,1:-1] = np.reshape(np.arange(k, k+ny*nz), (ny, nz))
        k += ny*nz
        iy0 = np.zeros((nx, nz+2), dtype=np.int32)
        iy0[1:-1,1:-1] = np.reshape(np.arange(k, k+(nx-2)*nz), (nx-2, nz))
        k += (nx-2)*nz
        iy1 = np.zeros((nx, nz+2), dtype=np.int32)
        iy1[1:-1,1:-1] = np.reshape(np.arange(k, k+(nx-2)*nz), (nx-2, nz))
        k += (nx-2)*nz
        iz0 = np.zeros((nx, ny), dtype=np.int32)
        iz0[1:-1,1:-1] = np.reshape(np.arange(k, k+(nx-2)*(ny-2)), (nx-2, ny-2))
        k += (nx-2)*(ny-2)
        iz1 = np.zeros((nx, ny), dtype=np.int32)
        iz1[1:-1,1:-1] = np.reshape(np.arange(k, k+(nx-2)*(ny-2)), (nx-2, ny-2))
               
        # Set ghost cells in index arrays to indices from neighbouring faces:
        ix0[0,1:-1], ix0[-1,1:-1] = iy0[1,1:-1], iy1[1,1:-1]
        ix0[2:-2,0], ix0[1,0], ix0[-2,0] = iz0[1,1:-1], iy0[1,1], iy1[1,1]
        ix0[2:-2,-1], ix0[1,-1], ix0[-2,-1] = iz1[1,1:-1], iy0[1,-2], iy1[1,-2]
        ix1[0,1:-1], ix1[-1,1:-1] = iy0[-2,1:-1], iy1[-2,1:-1]
        ix1[2:-2,0], ix1[1,0], ix1[-2,0] = iz0[-2,1:-1], iy0[-2,1], iy1[-2,1]
        ix1[2:-2,-1], ix1[1,-1], ix1[-2,-1] = iz1[-2,1:-1], iy0[-2,-2], iy1[-2,-2]        
        iy0[0,1:-1], iy0[-1,1:-1] = ix0[1,1:-1], ix1[1,1:-1]
        iy0[1:-1,0], iy0[1:-1,-1] = iz0[1:-1,1], iz1[1:-1,1]
        iy0[0,1:-1], iy0[-1,1:-1] = ix0[1,1:-1], ix1[1,1:-1]
        iy1[1:-1,0], iy1[1:-1,-1] = iz0[1:-1,-2], iz1[1:-1,-2]
        iy1[1:-1,0], iy1[1:-1,-1] = iz0[1:-1,-2], iz1[1:-1,-2]
        iy1[0,1:-1], iy1[-1,1:-1] = ix0[-2,1:-1], ix1[-2,1:-1]       
        iz0[0,1:-1], iz0[-1,1:-1] = ix0[2:-2,1], ix1[2:-2,1]
        iz0[1:-1,0], iz0[1:-1,-1] = iy0[1:-1,1], iy1[1:-1,1]
        iz1[0,1:-1], iz1[-1,1:-1] = ix0[2:-2,-2], ix1[2:-2,-2]
        iz1[1:-1,0], iz1[1:-1,-1] = iy0[1:-1,-2], iy1[1:-1,-2]
    
        # -- distances to neighbouring cells (to account for ghost cells on other faces):
        dyxf = ix0*0 + dy
        dyxf[0,:], dyxf[-1,:] = dx, dx
        dzxf = ix0*0 + dz
        dzxf[:,0], dzxf[:,-1] = dx, dx
        dxyf = iy0*0 + dx
        dzyf = iy0*0 + dz
        dzyf[:,0], dzyf[:,-1] = dy, dy
        dxzf = iz0*0 + dx
        dyzf = iz0*0 + dy
        
        self.L = sp.lil_matrix((n, n))   
        # - face x0 (xmin):
        for j in range(1, ny+1):
            for k in range(1, nz+1):
                i = ix0[j,k]
                self.L[i,i] -= 2/dyxf[j-1,k]/dyxf[j+1,k]
                self.L[i,ix0[j-1,k]] += 2/dyxf[j-1,k]/(dyxf[j-1,k] + dyxf[j+1,k])
                self.L[i,ix0[j+1,k]] += 2/dyxf[j+1,k]/(dyxf[j-1,k] + dyxf[j+1,k])                                                          
                self.L[i,i] -= 2/dzxf[j,k-1]/dzxf[j,k+1]
                self.L[i,ix0[j,k-1]] += 2/dzxf[j,k-1]/(dzxf[j,k-1] + dzxf[j,k+1])
                self.L[i,ix0[j,k+1]] += 2/dzxf[j,k+1]/(dzxf[j,k-1] + dzxf[j,k+1])
        # - face x1 (xmax):
        for j in range(1, ny+1):
            for k in range(1, nz+1):
                i = ix1[j,k]
                self.L[i,i] -= 2/dyxf[j-1,k]/dyxf[j+1,k]
                self.L[i,ix1[j-1,k]] += 2/dyxf[j-1,k]/(dyxf[j-1,k] + dyxf[j+1,k])
                self.L[i,ix1[j+1,k]] += 2/dyxf[j+1,k]/(dyxf[j-1,k] + dyxf[j+1,k])                                                          
                self.L[i,i] -= 2/dzxf[j,k-1]/dzxf[j,k+1]
                self.L[i,ix1[j,k-1]] += 2/dzxf[j,k-1]/(dzxf[j,k-1] + dzxf[j,k+1])
                self.L[i,ix1[j,k+1]] += 2/dzxf[j,k+1]/(dzxf[j,k-1] + dzxf[j,k+1])             
        # - face y0 (ymin):
        for j in range(1, nx-1):
            for k in range(1, nz+1):
                i = iy0[j,k]
                self.L[i,i] -= 2/dxyf[j-1,k]/dxyf[j+1,k]
                self.L[i,iy0[j-1,k]] += 2/dxyf[j-1,k]/(dxyf[j-1,k] + dxyf[j+1,k])
                self.L[i,iy0[j+1,k]] += 2/dxyf[j+1,k]/(dxyf[j-1,k] + dxyf[j+1,k])                                                          
                self.L[i,i] -= 2/dzyf[j,k-1]/dzyf[j,k+1]
                self.L[i,iy0[j,k-1]] += 2/dzyf[j,k-1]/(dzyf[j,k-1] + dzyf[j,k+1])
                self.L[i,iy0[j,k+1]] += 2/dzyf[j,k+1]/(dzyf[j,k-1] + dzyf[j,k+1])      
        # - face y1 (ymax):
        for j in range(1, nx-1):
            for k in range(1, nz+1):
                i = iy1[j,k]
                self.L[i,i] -= 2/dxyf[j-1,k]/dxyf[j+1,k]
                self.L[i,iy1[j-1,k]] += 2/dxyf[j-1,k]/(dxyf[j-1,k] + dxyf[j+1,k])
                self.L[i,iy1[j+1,k]] += 2/dxyf[j+1,k]/(dxyf[j-1,k] + dxyf[j+1,k])                                                          
                self.L[i,i] -= 2/dzyf[j,k-1]/dzyf[j,k+1]
                self.L[i,iy1[j,k-1]] += 2/dzyf[j,k-1]/(dzyf[j,k-1] + dzyf[j,k+1])
                self.L[i,iy1[j,k+1]] += 2/dzyf[j,k+1]/(dzyf[j,k-1] + dzyf[j,k+1])   
        # - face z0 (zmin):
        for j in range(1, nx-1):
            for k in range(1, ny-1):
                i = iz0[j,k]
                self.L[i,i] -= 2/dxzf[j-1,k]/dxzf[j+1,k]
                self.L[i,iz0[j-1,k]] += 2/dxzf[j-1,k]/(dxzf[j-1,k] + dxzf[j+1,k])
                self.L[i,iz0[j+1,k]] += 2/dxzf[j+1,k]/(dxzf[j-1,k] + dxzf[j+1,k])                                                          
                self.L[i,i] -= 2/dyzf[j,k-1]/dyzf[j,k+1]
                self.L[i,iz0[j,k-1]] += 2/dyzf[j,k-1]/(dyzf[j,k-1] + dyzf[j,k+1])
                self.L[i,iz0[j,k+1]] += 2/dyzf[j,k+1]/(dyzf[j,k-1] + dyzf[j,k+1])    
        # - face z1 (zmax):
        for j in range(1, nx-1):
            for k in range(1, ny-1):
                i = iz1[j,k]
                self.L[i,i] -= 2/dxzf[j-1,k]/dxzf[j+1,k]
                self.L[i,iz1[j-1,k]] += 2/dxzf[j-1,k]/(dxzf[j-1,k] + dxzf[j+1,k])
                self.L[i,iz1[j+1,k]] += 2/dxzf[j+1,k]/(dxzf[j-1,k] + dxzf[j+1,k])                                                          
                self.L[i,i] -= 2/dyzf[j,k-1]/dyzf[j,k+1]
                self.L[i,iz1[j,k-1]] += 2/dyzf[j,k-1]/(dyzf[j,k-1] + dyzf[j,k+1])
                self.L[i,iz1[j,k+1]] += 2/dyzf[j,k+1]/(dyzf[j,k-1] + dyzf[j,k+1]) 
                
        self.L = self.L.tocsc()
        
        

    def boundaryDiv(self, ax, ay, az):
        """
            Compute tangential divergence of A on the six boundaries, and return as a vector.
            Inputs ax, ay, az should be references to interpolation functions for cmpts.
        """
        
        dx = self.dx; dy = self.dy; dz = self.dz
        nx = self.nx; ny = self.ny; nz = self.nz
        
        # -- first construct arrays of ax, ay, az on each face (including ghost cells):
        y2, z2 = np.meshgrid(self.yc, self.z1, indexing='ij')
        ayx0 = ay(y2*0 + self.x1[0], y2, z2)
        ayx1 = ay(y2*0 + self.x1[-1], y2, z2)
        y2, z2 = np.meshgrid(self.y1, self.zc, indexing='ij')
        azx0 = az(y2*0 + self.x1[0], y2, z2)
        azx1 = az(y2*0 + self.x1[-1], y2, z2)
        x2, z2 = np.meshgrid(self.xc, self.z1, indexing='ij')
        axy0 = ax(x2, x2*0+self.y1[0], z2)
        axy1 = ax(x2, x2*0+self.y1[-1], z2)
        x2, z2 = np.meshgrid(self.x1, self.zc, indexing='ij')
        azy0 = az(x2, x2*0+self.y1[0], z2)
        azy1 = az(x2, x2*0+self.y1[-1], z2)
        x2, y2 = np.meshgrid(self.xc, self.y1, indexing='ij')
        axz0 = ax(x2, y2, x2*0+self.z1[0])
        axz1 = ax(x2, y2, x2*0+self.z1[-1])
        x2, y2 = np.meshgrid(self.x1, self.yc, indexing='ij')
        ayz0 = ay(x2, y2, x2*0+self.z1[0])
        ayz1 = ay(x2, y2, x2*0+self.z1[-1])
        # -- set ghost cells to wrap around onto neighbouring face(s):
        ayx0[0,:], ayx0[-1,:] = -axy0[1,:], axy1[1,:]
        azx0[1:-1,0], azx0[1:-1,-1] = -axz0[1,1:-1], axz1[1,1:-1]
        azx0[0,0], azx0[0,-1], azx0[-1,0], azx0[-1,-1] = -axy0[1,0], axy0[1,-1], -axy1[1,0], axy1[1,-1]
        ayx1[0,:], ayx1[-1,:] = axy0[-2,:], -axy1[-2,:]
        azx1[1:-1,0], azx1[1:-1,-1] = axz0[-2,1:-1], -axz1[-2,1:-1]
        azx1[0,0], azx1[0,-1], azx1[-1,0], azx1[-1,-1] = axy0[-2,0], -axy0[-2,-1], axy1[-2,0], -axy1[-2,-1]       
        axy0[0,:], axy0[-1,:] = -ayx0[1,:], ayx1[1,:]
        azy0[1:-1,0], azy0[1:-1,-1] = -ayz0[1:-1,1], ayz1[1:-1,1]
        azy0[0,0], azy0[-1,0], azy0[0,-1], azy0[-1,-1] = -ayx0[1,0], -ayx1[1,0], ayx0[1,-1], ayx1[1,-1]
        axy1[0,:], axy1[-1,:] = ayx0[-2,:], -ayx1[-2,:]
        azy1[1:-1,0], azy1[1:-1,-1] = ayz0[1:-1,-2], -ayz1[1:-1,-2]
        azy1[0,0], azy1[-1,0], azy1[0,-1], azy1[-1,-1] = ayx0[-2,0], ayx1[-2,0], -ayx0[-2,-1], -ayx1[-2,-1]
        axz0[0,:], axz0[-1,:] = -azx0[:,1], azx1[:,1]
        ayz0[1:-1,0], ayz0[1:-1,-1] = -azy0[1:-1,1], azy1[1:-1,1]
        ayz0[0,0], ayz0[-1,0], ayz0[0,-1], ayz0[-1,-1] = -azx0[0,1], -azx1[0,1], azx0[-1,1], azx1[-1,1]
        axz1[0,:], axz1[-1,:] = azx0[:,-2], -azx1[:,-2]
        ayz1[1:-1,0], ayz1[1:-1,-1] = azy0[1:-1,-2], -azy1[1:-1,-2]
        ayz1[0,0], ayz1[-1,0], ayz1[0,-1], ayz1[-1,-1] = -azx0[0,-2], -azx1[0,-2], azx0[-1,-2], azx1[-1,-2]
        # -- arrays of (half) edge lengths (to account for ghost cells):
        dyxf = ayx0*0 + 0.5*dy
        dyxf[0,:], dyxf[-1,:] = 0.5*dx, 0.5*dx
        dzxf = azx0*0 + 0.5*dz
        dzxf[:,0], dzxf[:,-1] = 0.5*dx, 0.5*dx
        dxyf = axy0*0 + 0.5*dx
        dxyf[0,:], dxyf[-1,:] = 0.5*dy, 0.5*dy
        dzyf = azy0*0 + 0.5*dz
        dzyf[:,0], dzyf[:,-1] = 0.5*dy, 0.5*dy
        dxzf = axz0*0 + 0.5*dx
        dxzf[0,:], dxzf[-1,:] = 0.5*dz, 0.5*dz
        dyzf = ayz0*0 + 0.5*dy
        dyzf[:,0], dyzf[:,-1] = 0.5*dz, 0.5*dz              
                
        # -- compute div_h(A) by centred differences on each face:
        # x = xmin
        rhs = (ayx0[1:,:] - ayx0[:-1,:])/(dyxf[1:,:] + dyxf[:-1,:]) \
            + (azx0[:,1:] - azx0[:,:-1])/(dzxf[:,1:] + dzxf[:,:-1])
        g = np.reshape(rhs, ny*nz)
        # x = xmax
        rhs = (ayx1[1:,:] - ayx1[:-1,:])/(dyxf[1:,:] + dyxf[:-1,:]) \
            + (azx1[:,1:] - azx1[:,:-1])/(dzxf[:,1:] + dzxf[:,:-1])
        g = np.append(g, np.reshape(rhs, ny*nz))
        # y = ymin
        rhs = (axy0[1:,:] - axy0[:-1,:])/(dxyf[1:,:] + dxyf[:-1,:]) \
            + (azy0[:,1:] - azy0[:,:-1])/(dzyf[:,1:] + dzyf[:,:-1])
        g = np.append(g, np.reshape(rhs[1:-1,:], (nx-2)*nz))
        # y = ymax
        rhs = (axy1[1:,:] - axy1[:-1,:])/(dxyf[1:,:] + dxyf[:-1,:]) \
            + (azy1[:,1:] - azy1[:,:-1])/(dzyf[:,1:] + dzyf[:,:-1])
        g = np.append(g, np.reshape(rhs[1:-1,:], (nx-2)*nz))
        # z = zmin
        rhs = (axz0[1:,:] - axz0[:-1,:])/(dxzf[1:,:] + dxzf[:-1,:]) \
            + (ayz0[:,1:] - ayz0[:,:-1])/(dyzf[:,1:] + dyzf[:,:-1])
        g = np.append(g, np.reshape(rhs[1:-1,1:-1], (nx-2)*(ny-2)))
        # z = zmax
        rhs = (axz1[1:,:] - axz1[:-1,:])/(dxzf[1:,:] + dxzf[:-1,:]) \
            + (ayz1[:,1:] - ayz1[:,:-1])/(dyzf[:,1:] + dyzf[:,:-1])
        g = np.append(g, np.reshape(rhs[1:-1,1:-1], (nx-2)*(ny-2)))
        
        return g
    

    def matchUniversalGauge(self, potential=False):
        """
            Modify the vector potential so that Axn satisfies the Hornig universal gauge
            condition on all boundaries (i.e., div_h(A) = 0).
            Do this by changing A to A + grad(phi) where phi is located at grid points.
            First, find phi on each boundary face by solving
                laplace(phi) = -div_h(A)   with phi = 0.
            Then, extend phi to the interior by solving the 3D Laplace equation (Dirichlet),
            with fast-Poisson solver.
            
            If potential=True, do this for the potential field.
        """
           
        if potential:
            ax = self.apx
            ay = self.apy
            az = self.apz
            afile = 'ape.unf'
            print('Matching Apxn to universal gauge on each boundary...')
        else:
            ax = self.ax
            ay = self.ay
            az = self.az
            afile = 'ae.unf'
            print('Matching Axn to universal gauge on each boundary...')

        dx = self.dx; dy = self.dy; dz = self.dz
        nx = self.nx; ny = self.ny; nz = self.nz
   
        # Solve Poisson equation for phi on the boundary (up to constant):
        # - compute laplacian matrix (if necessary):
        try:
            temp = self.L[0,0]
        except:
            self.lapMatrix()
        # - construct rhs = -div_h(A) at grid points:
        g = -self.boundaryDiv(ax, ay, az)
        # - solve Poisson equation to get phi on boundary:
        #   (fix overall constant by fixing first element to zero)
        phib = g*0   
        phib[1:] = sp.linalg.spsolve(self.L[1:,1:], g[1:])
        del(g)
        # - unpack separate arrays for each boundary:
        phi_xmin = np.reshape(phib[0:ny*nz], (ny,nz))
        phi_xmax = np.reshape(phib[ny*nz:2*ny*nz], (ny,nz))
        phi_ymin, phi_ymax = np.zeros((nx,nz)), np.zeros((nx,nz))
        phi_ymin[1:-1,:] = np.reshape(phib[2*ny*nz:2*ny*nz+(nx-2)*nz], (nx-2,nz))
        phi_ymin[0,:], phi_ymin[-1,:] = phi_xmin[0,:], phi_xmax[0,:]
        phi_ymax[1:-1,:] = np.reshape(phib[2*ny*nz+(nx-2)*nz:2*ny*nz+2*(nx-2)*nz], (nx-2,nz))
        phi_ymax[0,:], phi_ymax[-1,:] = phi_xmin[-1,:], phi_xmax[-1,:]
        phi_zmin, phi_zmax = np.zeros((nx,ny)), np.zeros((nx,ny))
        phi_zmin[1:-1,1:-1] = np.reshape(phib[2*ny*nz+2*(nx-2)*nz:2*ny*nz+2*(nx-2)*nz+(nx-2)*(ny-2)], (nx-2,ny-2))
        phi_zmin[0,:], phi_zmin[-1,:] = phi_xmin[:,0], phi_xmax[:,0]
        phi_zmin[1:-1,0], phi_zmin[1:-1,-1] = phi_ymin[1:-1,0], phi_ymax[1:-1,0]
        phi_zmax[1:-1,1:-1] = np.reshape(phib[2*ny*nz+2*(nx-2)*nz+(nx-2)*(ny-2):2*ny*nz+2*(nx-2)*nz+2*(nx-2)*(ny-2)], (nx-2,ny-2))
        phi_zmax[0,:], phi_zmax[-1,:] = phi_xmin[:,-1], phi_xmax[:,-1]
        phi_zmax[1:-1,0], phi_zmax[1:-1,-1] = phi_ymin[1:-1,-1], phi_ymax[1:-1,-1]        
        del(phib)

        # Extend phi to interior by 3D Dirichlet solution of Laplace equation, using fast-Poisson:
        u = np.zeros((nx-2, ny-2, nz-2))        
        u[0,:,:] -= phi_xmin[1:-1,1:-1]/dx**2
        u[-1,:,:] -= phi_xmax[1:-1,1:-1]/dx**2
        u[:,0,:] -= phi_ymin[1:-1,1:-1]/dy**2
        u[:,-1,:] -= phi_ymax[1:-1,1:-1]/dy**2
        u[:,:,0] -= phi_zmin[1:-1,1:-1]/dz**2
        u[:,:,-1] -= phi_zmax[1:-1,1:-1]/dz**2
    
        u = fp.dst(u, axis=0, type=1)
        u = fp.dst(u, axis=1, type=1)
        u = fp.dst(u, axis=2, type=1)
        
        m, p, q = np.mgrid[1:nx-1, 1:ny-1 ,1:nz-1]
        m = (2*np.cos(m*np.pi/nx) - 2)/dx**2
        p = (2*np.cos(p*np.pi/ny) - 2)/dy**2
        q = (2*np.cos(q*np.pi/nz) - 2)/dz**2
        u /= m + p + q
        
        u = fp.dst(u, axis=2, type=1)/2/(nz-1)
        u = fp.dst(u, axis=1, type=1)/2/(ny-1)
        u = fp.dst(u, axis=0, type=1)/2/(nx-1)
        
        phi = np.zeros((nx, ny, nz))
        phi[1:-1,1:-1,1:-1] = u
        phi[0,:,:] = phi_xmin
        phi[-1,:,:] = phi_xmax
        phi[:,0,:] = phi_ymin
        phi[:,-1,:] = phi_ymax
        phi[:,:,0] = phi_zmin
        phi[:,:,-1] = phi_zmax
        
        # Change gauge of vector potential:
        fid = FortranFile(self.tmpath+afile, 'w')       
        x3, y3, z3 = np.meshgrid(self.xc, self.y1, self.z1, indexing='ij')
        ax1 = ax(x3, y3, z3)
        ax1[1:-1,:,:] += (phi[1:,:,:] - phi[:-1,:,:])/dx
        ax1[0,:,:] = ax1[1,:,:]
        ax1[-1,:,:] = ax1[-2,:,:]
        fid.write_record(np.swapaxes(ax1.astype(np.float64), 0, 2))
        if potential:
            self.apxs = rgi((self.xc, self.y1, self.z1), ax1)           
        else:
            self.axs = rgi((self.xc, self.y1, self.z1), ax1)
        del(ax1)
        x3, y3, z3 = np.meshgrid(self.x1, self.yc, self.z1, indexing='ij')
        ay1 = ay(x3, y3, z3)
        ay1[:,1:-1,:] += (phi[:,1:,:] - phi[:,:-1,:])/dy
        ay1[:,0,:] = ay1[:,1,:]
        ay1[:,-1,:] = ay1[:,-2,:]
        fid.write_record(np.swapaxes(ay1.astype(np.float64), 0, 2))
        if potential:
            self.apys = rgi((self.x1, self.yc, self.z1), ay1)      
        else:
            self.ays = rgi((self.x1, self.yc, self.z1), ay1)
        del(ay1)
        x3, y3, z3 = np.meshgrid(self.x1, self.y1, self.zc, indexing='ij')
        az1 = az(x3, y3, z3)
        az1[:,:,1:-1] += (phi[:,:,1:] - phi[:,:,:-1])/dz
        az1[:,:,0] = az1[:,:,1]
        az1[:,:,-1] = az1[:,:,-2]
        fid.write_record(np.swapaxes(az1.astype(np.float64), 0, 2))       
        if potential:
            self.apzs = rgi((self.x1, self.y1, self.zc), az1)            
        else:
            self.azs = rgi((self.x1, self.y1, self.zc), az1)
        del(az1)
        fid.close()

        return phi
    
    
    def relativeHelicity(self):
        """
            Compute the total relative helicity by integrating (A + Ap).(B - Bp).
            Values are interpolated to the original grid points.
        """
        x3, y3, z3 = np.meshgrid(self.x1, self.y1, self.z1, indexing='ij')
        ax, ay, az = self.ax(x3, y3, z3), self.ay(x3, y3, z3), self.az(x3, y3, z3)
        apx, apy, apz = self.apx(x3, y3, z3), self.apy(x3, y3, z3), self.apz(x3, y3, z3)
        bx, by, bz = self.bx(x3, y3, z3), self.by(x3, y3, z3), self.bz(x3, y3, z3)
        bpx, bpy, bpz = self.bpx(x3, y3, z3), self.bpy(x3, y3, z3), self.bpz(x3, y3, z3)
                
        h = (ax + apx)*(bx - bpx) + (ay + apy)*(by - bpy) + (az + apz)*(bz - bpz)
        rh = trapz(trapz(trapz(h, self.x1, axis=0), self.y1, axis=0), self.z1, axis=0)
        return rh
    
    
    def totalHelicity(self, potential=False):
        """
            Compute the total helicity by integrating A.B.
            Values are interpolated to the original grid points.
            If potential=True then integrate Ap.Bp.
        """
        x3, y3, z3 = np.meshgrid(self.x1, self.y1, self.z1, indexing='ij')
        if (potential):
            ax, ay, az = self.apx(x3, y3, z3), self.apy(x3, y3, z3), self.apz(x3, y3, z3)
            bx, by, bz = self.bpx(x3, y3, z3), self.bpy(x3, y3, z3), self.bpz(x3, y3, z3)
        else:
            ax, ay, az = self.ax(x3, y3, z3), self.ay(x3, y3, z3), self.az(x3, y3, z3)
            bx, by, bz = self.bx(x3, y3, z3), self.by(x3, y3, z3), self.bz(x3, y3, z3)
         
        h = ax*bx + ay*by + az*bz
        th = trapz(trapz(trapz(h, self.x1, axis=0), self.y1, axis=0), self.z1, axis=0)
        return th   
    
    
    def crossHelicity(self):
        """
            Compute the cross term in the relative helicity by volume integration
            of Ap.B - A.Bp.
            Values are interpolated to the original grid points.
        """
        x3, y3, z3 = np.meshgrid(self.x1, self.y1, self.z1, indexing='ij')
        ax, ay, az = self.ax(x3, y3, z3), self.ay(x3, y3, z3), self.az(x3, y3, z3)
        apx, apy, apz = self.apx(x3, y3, z3), self.apy(x3, y3, z3), self.apz(x3, y3, z3)
        bx, by, bz = self.bx(x3, y3, z3), self.by(x3, y3, z3), self.bz(x3, y3, z3)
        bpx, bpy, bpz = self.bpx(x3, y3, z3), self.bpy(x3, y3, z3), self.bpz(x3, y3, z3)
         
        h = apx*bx - ax*bpx + apy*by - ay*bpy + apz*bz - az*bpz
        th = trapz(trapz(trapz(h, self.x1, axis=0), self.y1, axis=0), self.z1, axis=0)
        return th   

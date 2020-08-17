"""Source grid fitting classes and functions.

To Do:
   * Implement lmfit
   * Add more columns to the DistortedGrid object
"""
from __future__ import print_function
from __future__ import absolute_import
import os
import scipy
import numpy as np
from astropy.io import fits
from lmfit import Parameters
from scipy import optimize
from scipy.spatial import distance
from itertools import product

class BaseGrid:

    def __init__(self, ystep, xstep, theta, y0, x0, ncols, nrows, 
                 optic_shifts=None):

        self.nrows = nrows
        self.ncols = ncols
        self.ystep = ystep
        self.xstep = xstep
        self.theta = theta
        self.y0 = y0
        self.x0 = x0

        if optic_shifts is not None:
            self.add_optic_shifts(optic_shifts)
        else:
            self.dy = np.zeros(nrows*ncols)
            self.dx = np.zeros(nrows*ncols)

    def add_optic_shifts(self, optic_shifts):

        dy, dx = optic_shifts

        nsources = self.nrows*self.ncols
        if (dy.shape[0]==nsources)*(dx.shape[0]==sources):
            self.dy = dy
            self.dx = dx
        else:
            raise ValueError

    def make_source_grid(self):

        ## Create a standard nrows x ncols grid of points
        y_array = np.asarray([n*self.ystep - (self.nrows-1)*self.ystep/2. \
                                  for n in range(self.nrows)])
        x_array = np.asarray([n*self.xstep - (self.ncols-1)*self.xstep/2. \
                                  for n in range(self.ncols)])
        Y, X = np.meshgrid(y_array, x_array)
        
        ## Rotate grid using rotation matrix
        Xr = np.cos(self.theta)*X - np.sin(self.theta)*Y
        Yr = np.sin(self.theta)*X + np.cos(self.theta)*Y
        
        ## Move center of grid to desired x/y center coordinates
        Xr += self.x0
        Yr += self.y0

        ## Add optic shifts
        Xr += self.dx
        Yr += self.dy
        
        ## Return the flattened arrays
        return Yr.flatten(), Xr.flatten() 
    
class DetectedGrid(BaseGrid):
    """Grid of detected sources.
    
    Grid is defined by grid spacing in two orthogonal directions and 
    a rotation angle of the grid lines with respect to the x-axis.
       
    Attributes:
        ystep (float): Grid spacing in the vertical direction.
        xstep (float): Grid spacing in the horizontal direction.
        theta (float): Angle [rads] between the grid and global horizontal directions
        y0 (float): Grid center y-position.
        x0 (float) Grid center x-position.:
    """

    colnames = ['X', 'Y', 'DX', 'DY', 
                'FLUX', 'RFLUX',
                'XX', 'YY', 'DXX', 'DYY']
    
    def __init__(self, ystep, xstep, theta, y0, x0, ncols, nrows, data,
                 optic_shifts=None):
        """Initializes SourceGrid class with grid parameters."""

        super().__init__(ystep, xstep, theta, y0, x0, ncols, nrows, optic_shifts)

        self._data = {}
        for col in self.colnames:
            if data[col].shape[0] != ncols*nrows:
                raise ValueError('{0} array size must be {1}'.format(col, ncols*nrows))
            else:
                self._data[col] = data[col]

    def __getitem__(self, column):

        return self._data[column]

    def __setitem__(self, column, array):

        if array.shape[0] != self.ncols*self.nrows:
            raise ValueError('{0} array size must be {1}'.format(column, 
                                                                 self.ncols*self.nrows))
        else:
            self._data[column] = array

    @classmethod
    def from_fits(cls, infile):

        with fits.open(infile) as hdulist:

            x0 = hdulist[0].header['X0']
            y0 = hdulist[0].header['Y0']
            theta = hdulist[0].header['THETA']
            xstep = hdulist[0].header['XSTEP']
            ystep = hdulist[0].header['YSTEP']
            ncols = hdulist[0].header['NCOLS']
            nrows = hdulist[0].header['NROWS']

            dy = hdulist[1].data['OPTIC_DY']
            dx = hdulist[1].data['OPTIC_DX']
            optic_shifts = (dy, dx)

            data = {}
            for col in cls.colnames:
                data[col] = hdulist[2].data[col]

        return cls(ystep, xstep, theta, y0, x0, nrows, ncols, data, 
                   optic_shifts=optic_shifts)

    def write_fits(self, outfile, **kwargs):

        ## Primary HDU with projected grid information
        hdr = fits.Header()
        hdr['X0'] = self.x0
        hdr['Y0'] = self.y0
        hdr['XSTEP'] = self.xstep
        hdr['YSTEP'] = self.ystep
        hdr['THETA'] = self.theta
        hdr['NCOLS'] = self.ncols
        hdr['NROWS'] = self.nrows
        prihdu = fits.PrimaryHDU(header=hdr)

        ## Optic shifts HDU
        optic_cols = [fits.Column('OPTIC_DY', array=self.dy, format='D'),
                      fits.Column('OPTIC_DX', array=self.dx, format='D')]
        optic_tablehdu = fits.BinTableHDU.from_columns(optic_cols)

        ## Source properties HDU
        source_cols = [fits.Column(col, array=self[col], format='D') \
                           for col in self.colnames]

        source_tablehdu = fits.BinTableHDU.from_columns(source_cols)
        hdulist = fits.HDUList([prihdu, optic_tablehdu, source_tablehdu])
        hdulist.writeto(outfile, **kwargs)

def coordinate_distances(y0, x0, y1, x1, metric='euclidean'):
    
    coords0 = np.stack([y0, x0], axis=1)
    coords1 = np.stack([y1, x1], axis=1)
    
    ## Calculate distances to all points
    distance_array = distance.cdist(coords0, coords1, metric=metric)
    indices = np.argsort(distance_array, axis=1)
    distances = np.sort(distance_array, axis=1)
    
    return indices, distances

def fit_error(params, srcY, srcX, nrows, ncols, optic_shifts=None, pixel_extent=None):
    """Calculate sum of positional errors of true source grid and model grid.
    
    For every true source, the distance to the nearest neighbor source 
    from a model source grid is calculated.  The mean for every true source
    is taken as the fit error between true source grid and model grid.
       
    Args:
        params (list): List of grid parameters.
        srcY (numpy.ndarray): Array of source y-positions.
        srcX (numpy.ndarray): Array of source x-positions.
        nrows (int): Number of grid rows.
        ncols (int): Number of grid columns.
        
    Returns:
        Float representing sum of nearest neighbor distances.
    """
    
    ## Fit parameters
    parvals = params.valuesdict()
    ystep = parvals['ystep']
    xstep = parvals['xstep']
    theta = parvals['theta']
    y0 = parvals['y0']
    x0 = parvals['x0']
    
    ## Create grid model and construct x/y coordinate arrays
    grid = BaseGrid(ystep, xstep, theta, y0, x0, ncols, nrows,
                    optic_shifts=optic_shifts)    
    gY, gX = grid.make_source_grid()

    ## Add function to filter BaseGrid position vectors on CCD pixel extent

    ## Calculate distances from each grid model source to nearest true source    
    indices, distances = coordinate_distances(srcY, srcX, gY, gX)
    
    return distances[:, 0]

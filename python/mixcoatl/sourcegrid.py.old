"""Source grid fitting classes and functions.

To Do:
   * Change class references to grid distortions.
   * Switch to masked arrays and masked columns
"""
from __future__ import print_function
from __future__ import absolute_import
import os
import scipy
import numpy as np
from astropy.io import fits
from scipy import optimize
from scipy.spatial import distance
from itertools import product

class BaseGrid:

    def __init__(self, ystep, xstep, theta, y0, x0, ncols, nrows):
        """Initializes SourceGrid class with grid parameters."""

        self.nrows = nrows
        self.ncols = ncols
        self.ystep = ystep
        self.xstep = xstep
        self.theta = theta
        self.y0 = y0
        self.x0 = x0

    def make_ideal_grid(self):

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
        
        ## Return the flattened arrays
        return Yr.flatten(), Xr.flatten()        
    
class DistortedGrid(BaseGrid):
    """Infinite undistorted grid of points.
    
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
                'FLUX', 'DFLUX',
                'XX', 'YY', 'DXX', 'DYY']
    
    def __init__(self, ystep, xstep, theta, y0, x0, ncols, nrows, data):
        """Initializes SourceGrid class with grid parameters."""

        super().__init__(ystep, xstep, theta, y0, x0, ncols, nrows)

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

            data = {}
            for col in cls.colnames:
                data[col] = hdulist[1].data[col]

        return cls(ystep, xstep, theta, y0, x0, nrows, ncols, data)

    def get_source_coordinates(self, centered=False, distorted=True):
        """Create x/y coordinate arrays for (nrows x ncols) grid of sources."""

        y = self['Y']
        x = self['X']
        
        ## Optionally add distortions
        if distorted:
            y += self['DY']
            x += self['DX']

        ## Optionally center
        if centered:
            y -= self.y0
            x -= self.x0

        return y, x

    def write_fits(self, outfile, **kwargs):

        hdr = fits.Header()
        hdr['X0'] = self.x0
        hdr['Y0'] = self.y0
        hdr['XSTEP'] = self.xstep
        hdr['YSTEP'] = self.ystep
        hdr['THETA'] = self.theta
        hdr['NCOLS'] = self.ncols
        hdr['NROWS'] = self.nrows

        cols = [fits.Column(col, array=self[col], format='D') \
                    for col in self.colnames]

        prihdu = fits.PrimaryHDU(header=hdr)
        tablehdu = fits.BinTableHDU.from_columns(cols)
        hdulist = fits.HDUList([prihdu, tablehdu])
        hdulist.writeto(outfile, **kwargs)

def coordinate_distances(y0, x0, y1, x1, metric='euclidean'):
    
    coords0 = np.stack([y0, x0], axis=1)
    coords1 = np.stack([y1, x1], axis=1)
    
    ## Calculate distances to all points
    distance_array = distance.cdist(coords0, coords1, metric=metric)
    indices = np.argsort(distance_array, axis=1)
    distances = np.sort(distance_array, axis=1)
    
    return indices, distances

def fit_error(params, srcY, srcX, nrows, ncols, distortions=None):
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
    dy, dx, theta, y0, x0 = params
    
    ## Create grid model and construct x/y coordinate arrays
    grid = BaseGrid(dy, dx, theta, y0, x0, ncols, nrows)    
    gY, gX = grid.make_ideal_grid()
    if distortions is not None:
        dY, dX = distortions
        dY[np.isnan(dY)] = 0.
        dX[np.isnan(dX)] = 0.
        gY += dY
        gX += dX

    srcY, srcX = remove_nan_coords(srcY, srcX)   

    ## Calculate distances from each grid model source to nearest true source    
    indices, distances = coordinate_distances(srcY, srcX, gY, gX)
    nn_distances = distances[:, 0]
    
    return np.mean(nn_distances)
    
def grid_fit(srcY, srcX, ncols, nrows, y0_guess, x0_guess, distortions=None):

    y, x = remove_nan_coords(srcY, srcX)
    nsources = y.shape[0]

    ## Calculate median distance to 4 nearest neighbors
    indices, distances = coordinate_distances(y, x, y, x)
    nn_indices = indices[:, 1:5]
    nn_distances = distances[: , 1:5]
    med_dist = np.median(nn_distances)

    ## Arrays to hold results (default is NaN)
    dist1_array = np.full(nsources, np.nan)
    dist2_array = np.full(nsources, np.nan)
    theta_array = np.full(nsources, np.nan)

    ## Calculate grid rotation
    for i in range(nsources):

        ## For each source, get 4 nearest neighbors
        yc = y[i]
        xc = x[i]

        for j in range(4):

            nn_dist = nn_distances[i, j] 
            if np.abs(nn_dist  - med_dist) > 10.: continue                
            y_nn = y[nn_indices[i, j]]
            x_nn = x[nn_indices[i, j]]

            ## Use the nearest neighbor in Quadrant 1 (with respect to source)
            if x_nn > xc:
                if y_nn > yc:
                    dist1_array[i] = nn_dist
                    theta_array[i] = np.arctan((y_nn-yc)/(x_nn-xc))
                else:
                    dist2_array[i] = nn_dist

    ## Take mean over all sources (ignoring NaN entries)
    theta = np.nanmedian(theta_array)
    if theta >= np.pi/4.:
        theta = theta - (np.pi/2.)
        xstep = np.nanmedian(dist2_array)
        ystep = np.nanmedian(dist1_array)
    else:
        xstep = np.nanmedian(dist1_array)
        ystep = np.nanmedian(dist2_array)

    ## Informed guess
    guess = informed_guess(y, x, ystep, xstep, theta, y0_guess, x0_guess,
                           ncols=ncols, nrows=nrows, distortions=distortions)

    ## Perform constrained fit to determine y0, x0
    bounds = [(ystep, ystep), (xstep, xstep), (theta, theta),
              (None, None), (None, None)]

    r = scipy.optimize.minimize(fit_error, guess, 
                                args=(y, x, ncols, nrows, distortions), 
                                method='SLSQP', bounds=bounds)
    y0 = r.x[3]
    x0 = r.x[4]

    ## Match catalog sources to grid sources
    grid = BaseGrid(ystep, xstep, theta, y0, x0, 
                    nrows, ncols)

    return grid

def remove_nan_coords(y, x):
    """Remove NaN entries for y/x position arrays.
    
    Args:
        y: Array of source y-positions.
        x: Array of source x-positions.
        
    Returns:
        Tuple of y/x positional arrays, removing any NaN 
        from the array.
    """

    y_good = y[~np.isnan(y) & ~np.isnan(x)]
    x_good = x[~np.isnan(y) & ~np.isnan(x)]
    
    return y_good, x_good

def informed_guess(srcY, srcX, ystep, xstep, theta, y0_guess, x0_guess,
                   ncols=49, nrows=49, distortions=None):
    """Initial optimization of grid x0/y0."""

    ## 3x3 grid around initial y0/x0 guess
    y0s = [y0_guess - ystep/2., y0_guess, y0_guess + ystep/2.]
    x0s = [x0_guess - xstep/2., x0_guess, x0_guess + xstep/2.]
    err = np.inf

    ## Update guess with best result
    for y0, x0 in product(y0s, x0s):
        params = [ystep, xstep, theta, y0, x0]
        new_err = fit_error(params, srcY, srcX, ncols, nrows, distortions)
        if new_err < err:
            err = new_err
            y0_guess = y0
            x0_guess = x0

    return [ystep, xstep, theta, y0_guess, x0_guess]

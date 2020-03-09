"""Source grid fitting classes and functions.

To Do:
   * Change class references to grid distortions.
   * Maybe add a dFlux term?
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

class GridDistortions:
    """Handler class for grid displacement results."""
    colnames = ['DX', 'DY', 'DXX', 'DYY', 'X', 'Y', 'XX', 'YY', 'FLUX']

    def __init__(self, source_grid, nrows, ncols, data):

        self.source_grid = source_grid
        self.nrows = nrows
        self.ncols = ncols
        self.data = data

    def __getitem__(self, column):

        return self.data[column]

    @classmethod
    def from_fits(self, infile):

        with fits.open(infile) as hdulist:

            x0 = hdulist[0].header['X0']
            y0 = hdulist[0].header['Y0']
            theta = hdulist[0].header['THETA']
            xstep = hdulist[0].header['XSTEP']
            ystep = hdulist[0].header['YSTEP']
            ncols = hdulist[0].header['NCOLS']
            nrows = hdulist[0].header['NROWS']
            source_grid = SourceGrid(ystep, xstep, theta, y0, x0)

            data = {}
            for col in self.colnames:
                data[col] = hdulist[1].data[col]

        return cls(source_grid, nrows, ncols, data)

    def mask_entries(self, l):
        
        ## Consider making this using astropy and masked arrays
        bad_indices = np.hypot(self.data['DX'], self.data['DY']) >= l
        self.data['DX'][bad_indices] = np.nan
        self.data['DY'][bad_indices] = np.nan
        self.data['DXX'][bad_indices] = np.nan
        self.data['DYY'][bad_indices] = np.nan
        self.data['FLUX'][bad_indices] = np.nan

    def write_fits(self, outfile, **kwargs):

        hdr = fits.Header()
        hdr['X0'] = self.source_grid.x0
        hdr['Y0'] = self.source_grid.y0
        hdr['XSTEP'] = self.source_grid.xstep
        hdr['YSTEP'] = self.source_grid.ystep
        hdr['THETA'] = self.source_grid.theta
        hdr['NCOLS'] = self.ncols
        hdr['NROWS'] = self.nrows

        columns = [fits.Column(col, array=self.data[col], format='D') for col in self.colnames]
        
        prihdu = fits.PrimaryHDU(header=hdr)
        tablehdu = fits.BinTableHDU.from_columns(columns)
        hdulist = fits.PrimaryHDU([prihdu, tablehdu])
        hdulist.writeto(outfile, **kwargs)

class SourceGrid:
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
    
    def __init__(self, ystep, xstep, theta, y0, x0):
        """Initializes SourceGrid class with grid parameters."""
        
        self.ystep = ystep
        self.xstep = xstep
        self.theta = theta
        self.y0 = y0
        self.x0 = x0
    
    @classmethod
    def from_source_catalog(cls, source_catalog, y0_guess=0.0, x0_guess=0.0, 
                            y_kwd='base_SdssCentroid_y', 
                            x_kwd='base_SdssCentroid_x', 
                            ncols=49, nrows=49, 
                            **kwargs):
        """Generate a source grid model from a given source catalog.
    
        Args:
            source_catalog: Source catalog table.
            y0_guess (float): Initial grid center y0 guess.
            x0_guess (float): Initial grid center x0 guess.
            y_kwd (string): Y-position keyword for source catalog.
            x_kwd (string): X-position keyword for source catalog.
        
        Returns:
            SourceGrid object with grid parameters calculated from the input
            source catalog.
        """
    
        ## Get real-valued x/y coordinate arrays
        srcY_raw = source_catalog[y_kwd]
        srcX_raw = source_catalog[x_kwd]
        srcY, srcX = remove_nan_coords(srcY_raw, srcX_raw)
        nsources = srcY.shape[0] 
        
        ## Calculate median distance to 4 nearest neighbors
        indices, distances = coordinate_distances(srcY, srcX, srcY, srcX)
        nn_indices = indices[:, 1:5]
        nn_distances = distances[: , 1:5]
        med_dist = np.median(nn_distances)
    
        ## Result arrays (default is NaN)
        dist1_array = np.full(nsources, np.nan)
        dist2_array = np.full(nsources, np.nan)
        theta_array = np.full(nsources, np.nan)
    
        ## Calculate grid rotation using 4 nearest neighbors
        for i in range(nsources):

            y = srcY[i]
            x = srcX[i]
        
            for j in range(4):
            
                nn_dist = nn_distances[i, j] 
                if np.abs(nn_dist  - med_dist) > 10.: continue                
                y_nn = srcY[nn_indices[i, j]]
                x_nn = srcX[nn_indices[i, j]]
    
                ## Use the nearest neighbor in Quadrant 1 (with respect to source)
                if x_nn > x:
                    if y_nn > y:
                        dist1_array[i] = nn_dist
                        theta_array[i] = np.arctan((y_nn-y)/(x_nn-x))
                    else:
                        dist2_array[i] = nn_dist
                    
        ## Mean over all sources (is nanmean needed?)
        theta = np.nanmedian(theta_array)
        if theta >= np.pi/4.:
            theta = theta - (np.pi/2.)
            xstep = np.nanmedian(dist2_array)
            ystep = np.nanmedian(dist1_array)
        else:
            xstep = np.nanmedian(dist1_array)
            ystep = np.nanmedian(dist2_array)
                
        ## Constrained fit to determine grid center x0/y0
        guess = cls.informed_guess(srcY, srcX, ystep, xstep, 
                                   theta, y0_guess, x0_guess,
                                   ncols=ncols, nrows=nrows)
        bounds = [(ystep, ystep), 
                  (xstep, xstep), 
                  (theta, theta), 
                  (None, None), 
                  (None, None)]
        
        fit_results = scipy.optimize.minimize(grid_fit_error, guess, 
                                              args=(srcY, srcX, ncols, nrows), 
                                              method='SLSQP', bounds=bounds)    

        return cls(*fit_results.x)
            
    def make_grid(self, nrows=49, ncols=49):
        """Create x/y coordinate arrays for (nrows x ncols) grid of sources."""
        
        ## Create a standard nrows x ncols grid of points
        y_array = np.asarray([n*self.ystep - (nrows-1)*self.ystep/2. for n in range(nrows)])
        x_array = np.asarray([n*self.xstep - (ncols-1)*self.xstep/2. for n in range(ncols)])
        Y, X = np.meshgrid(y_array, x_array)
        
        ## Rotate grid using rotation matrix
        Xr = np.cos(self.theta)*X - np.sin(self.theta)*Y
        Yr = np.sin(self.theta)*X + np.cos(self.theta)*Y
        
        ## Move center of grid to desired x/y center coordinates
        Xr += self.x0
        Yr += self.y0
        
        ## Return the flattened arrays
        return Yr.flatten(), Xr.flatten()

    @staticmethod
    def informed_guess(srcY, srcX, ystep, xstep, theta, y0_guess, x0_guess,
                       ncols=49, nrows=49):
        """Initial optimization of grid x0/y0."""

        ## 3x3 grid around initial y0/x0 guess
        y0s = [y0_guess - ystep/2., y0_guess, y0_guess + ystep/2.]
        x0s = [x0_guess - xstep/2., x0_guess, x0_guess + xstep/2.]
        err = np.inf
        
        ## Update guess with best result
        for y0, x0 in product(y0s, x0s):
            params = [ystep, xstep, theta, y0, x0]
            new_err = grid_fit_error(params, srcY, srcX, ncols, nrows)
            if new_err < err:
                err = new_err
                y0_guess = y0
                x0_guess = x0

        return [ystep, xstep, theta, y0_guess, x0_guess]
    
def grid_fit_error(params, srcY, srcX, nrows, ncols):
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
    grid = SourceGrid(dy, dx, theta, y0, x0)    
    gY, gX = grid.make_grid(nrows=nrows, ncols=ncols)    
    srcY, srcX = remove_nan_coords(srcY, srcX)   

    ## Calculate distances from each grid model source to nearest true source    
    indices, distances = coordinate_distances(srcY, srcX, gY, gX)
    nn_distances = distances[:, 0]
    
    return np.sqrt(np.mean(np.square(nn_distances)))

def remove_nan_coords(y, x):
    """Remove NaN entries for y/x position arrays.
    
    Args:
        y: Array of source y-positions.
        x: Array of source x-positions.
        
    Returns:
        Tuple of y/x positional arrays, removing any NaN 
        from the array.
    """
    
    y_new = y[~np.isnan(y) & ~np.isnan(x)]
    x_new = x[~np.isnan(y) & ~np.isnan(x)]
    
    return y_new, x_new
    
def coordinate_distances(y0, x0, y1, x1, metric='euclidean'):
    
    coords0 = np.stack([y0, x0], axis=1)
    coords1 = np.stack([y1, x1], axis=1)
    
    ## Calculate distances to all points
    distance_array = distance.cdist(coords0, coords1, metric=metric)
    indices = np.argsort(distance_array, axis=1)
    distances = np.sort(distance_array, axis=1)
    
    return indices, distances
    
def mask_by_ccd_geom(y, x, ccd_geometry):
    """Mask out source positions outside of CCD boundaries.
    
    Args:
        y (numpy.ndarray): Array of source y-positions.
        x (numpy.ndarray): Array of source x-positions.
        ccd_geometry (dict): Dictionary describing min/max CCD extent in x/y-directions.
        
    Returns:
        Tuple of y/x positional arrays, removing entries outside
        of CCD boundaries.
    """
    
    min_y = ccd_geometry['min_y']
    max_y = ccd_geometry['max_y']
    min_x = ccd_geometry['min_x']
    max_x = ccd_geometry['max_x']
    
    y_new = y[(min_y < y) & (y <= max_y) & (min_x < x) & (x <= max_x)]
    x_new = x[(min_y < y) & (y <= max_y) & (min_x < x) & (x <= max_x)]
    
    return y_new, x_new

"""Source grid fitting classes and functions.

To Do:
   * Implement new MatchedCatalog functionality
   * Maybe add getter/setter functionality for grid parameters
     that automatically updates centroids if changed.
"""
from __future__ import print_function
from __future__ import absolute_import
import os
import scipy
import numpy as np
from astropy.io import fits
from lmfit import Minimizer, Parameters
from scipy import optimize
from scipy.spatial import distance
from itertools import product

class DistortedGrid:

    def __init__(self, ystep, xstep, theta, y0, x0, ncols, nrows, 
                 normalized_shifts=None):

        ## Ideal grid parameters
        self.nrows = nrows
        self.ncols = ncols
        self.ystep = ystep
        self.xstep = xstep
        self.theta = theta
        self.y0 = y0
        self.x0 = x0

        self.make_source_grid()

        ## Add centroid shifts
        if normalized_shifts is None:
            self._norm_dy = np.zeros(nrows*ncols)
            self._norm_dx = np.zeros(nrows*ncols)
        else:
            self.add_normalized_shifts(normalized_shifts)

    @classmethod
    def from_fits(cls, infile):
        """Initialize DistortedGrid instance from a FITS file."""

        with fits.open(infile) as hdulist:

            x0 = hdulist['GRID_INFO'].header['X0']
            y0 = hdulist['GRID_INFO'].header['Y0']
            theta = hdulist['GRID_INFO'].header['THETA']
            xstep = hdulist['GRID_INFO'].header['XSTEP']
            ystep = hdulist['GRID_INFO'].header['YSTEP']
            ncols = hdulist['GRID_INFO'].header['NCOLS']
            nrows = hdulist['GRID_INFO'].header['NROWS']

            norm_dy = hdulist['GRID_INFO'].data['NORMALIZED_DY']
            norm_dx = hdulist['GRID_INFO'].data['NORMALIZED_DX']

        return cls(ystep, xstep, theta, y0, x0, nrows, ncols, 
                   normalized_shifts=(norm_dy, norm_dx))
    
    @property
    def norm_dx(self):
        return self._norm_dx

    @property
    def norm_dy(self):
        return self._norm_dy

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def add_centroid_shifts(self, centroid_shifts):
        """Calculate and add the normalized source centroid shifts."""

        dy, dx = centroid_shifts

        ## Rotate and normalize centroid_shifts
        norm_dx = (np.cos(-self.theta)*dx - np.sin(-self.theta)*dy)/self.xstep
        norm_dy = (np.sin(-self.theta)*dx + np.cos(-self.theta)*dy)/self.ystep

        self.add_normalized_shifts((norm_dy, norm_dx))

    def add_normalized_shifts(self, normalized_shifts):
        """Add the normalized source centroid shifts."""

        norm_dy, norm_dx = normalized_shifts
        nsources = self.nrows*self.ncols

        if (norm_dy.shape[0]==nsources)*(norm_dx.shape[0]==nsources):
            self._norm_dy = norm_dy
            self._norm_dx = norm_dx

    def make_grid_hdu(self):
        """Create an HDU with grid information."""

        hdr = fits.Header()
        hdr['X0'] = self.x0
        hdr['Y0'] = self.y0
        hdr['XSTEP'] = self.xstep
        hdr['YSTEP'] = self.ystep
        hdr['THETA'] = self.theta
        hdr['NCOLS'] = self.ncols
        hdr['NROWS'] = self.nrows

        ## Optic shifts HDU
        cols = [fits.Column('NORMALIZED_DY', array=self.norm_dy, format='D'),
                fits.Column('NORMALIZED_DX', array=self.norm_dx, format='D')]
        tablehdu = fits.BinTableHDU.from_columns(cols, header=hdr,
                                                 name='GRID_INFO')

        return tablehdu

    def make_source_grid(self):
        """Make rectilinear grid of sources."""

        ## Create a standard nrows x ncols grid of points
        y_array = np.asarray([n*self.ystep - (self.nrows-1)*self.ystep/2. \
                                  for n in range(self.nrows)])
        x_array = np.asarray([n*self.xstep - (self.ncols-1)*self.xstep/2. \
                                  for n in range(self.ncols)])
        y, x = np.meshgrid(y_array, x_array)

        self._y = y.flatten()
        self._x = x.flatten()

    def get_centroid_shifts(self):
        """Return the centroid shifts given the grid geometry."""

        dx = (np.cos(self.theta)*self.norm_dx - np.sin(self.theta)*self.norm_dy)*self.xstep
        dy = (np.sin(self.theta)*self.norm_dx + np.cos(self.theta)*self.norm_dy)*self.ystep

        return dy, dx

    def get_source_centroids(self, distorted=True):
        """Return source centroids given the grid geometry."""

        ## Add scaled centroid shifts
        if distorted:
            y = self.y + self.norm_dy*self.ystep
            x = self.x + self.norm_dx*self.xstep
        else:
            y = self.y
            x = self.x
        
        ## Rotate grid using rotation matrix
        xr = (np.cos(self.theta)*x - np.sin(self.theta)*y).flatten()
        yr = (np.sin(self.theta)*x + np.cos(self.theta)*y).flatten()
        
        ## Move center of grid to desired x/y center coordinates
        xr += self.x0
        yr += self.y0
        
        ## Return the flattened arrays
        return yr, xr

    def write_fits(self, outfile, **kwargs):
        """Write DistortedGrid instance to a FITS file."""

        ## Write grid HDU with minimal PrimaryHDU
        hdr = fits.Header()
        prihdu = fits.PrimaryHDU(header=hdr)
        tablehdu = self.make_grid_hdu()
        hdulist = fits.HDUList([prihdu, tablehdu])

        hdulist.writeto(outfile, **kwargs)

class MatchedCatalog:
    """WIP"""
    
    def __init__(self):
        pass

def coordinate_distances(y0, x0, y1, x1, metric='euclidean'):
    """Calculate the distances between two sets of points."""
    
    coords0 = np.stack([y0, x0], axis=1)
    coords1 = np.stack([y1, x1], axis=1)
    
    ## Calculate distances to all points
    distance_array = distance.cdist(coords0, coords1, metric=metric)
    indices = np.argsort(distance_array, axis=1)
    distances = np.sort(distance_array, axis=1)
    
    return indices, distances

def fit_error(params, srcY, srcX, nrows, ncols, normalized_shifts=None, 
              ccd_geom=None):
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
    grid = DistortedGrid(ystep, xstep, theta, y0, x0, ncols, nrows,
                         normalized_shifts=normalized_shifts)    
    gY, gX = grid.get_source_centroids()

    ## Filter source grid positions according to CCD geometry
    if ccd_geom is not None:
        ymin = 0
        ymax = ccd_geom.ny*2
        xmin = 0 
        xmax = ccd_geom.nx*8

        mask = (gY < ymax)*(gY > ymin)*(gX < xmax)*(gX > xmin)
        gY = gY[mask]
        gX = gX[mask]

    ## Calculate residuals   
    indices, distances = coordinate_distances(srcY, srcX, gY, gX)

    ## Scale if grid is offset by row/column
    if srcY.shape[0] < gY.shape[0]:
        return distances[:, 0] + (xstep+ystep)/2.
    else:
        return distances[:, 0]

def grid_fit(srcY, srcX, y0_guess, x0_guess, ncols, nrows,
             brute_search=False, vary_theta=False, 
             normalized_shifts=None, ccd_geom=None):

    ## Calculate mean xstep/ystep
    nsources = srcY.shape[0]
    indices, distances = coordinate_distances(srcY, srcX, srcY, srcX)
    nn_indices = indices[:, 1:5]
    nn_distances = distances[:, 1:5]
    med_dist = np.median(nn_distances)

    dist1_array = np.full(nsources, np.nan)
    dist2_array = np.full(nsources, np.nan)
    theta_array = np.full(nsources, np.nan)

    for i in range(nsources):

        yc = srcY[i]
        xc = srcX[i]

        for j in range(4):

            nn_dist = nn_distances[i, j]
            if np.abs(nn_dist - med_dist) > 10.: continue
            y_nn = srcY[nn_indices[i, j]]
            x_nn = srcX[nn_indices[i, j]]

            if x_nn > xc:
                if y_nn > yc:
                    dist1_array[i] = nn_dist
                    theta_array[i] = np.arctan((y_nn-yc)/(x_nn-xc))
                else:
                    dist2_array[i] = nn_dist

    ## Use theta to determine x/y step direction
    theta = np.nanmedian(theta_array)
    if theta >= np.pi/4.:
        theta = theta - (np.pi/2.)
        xstep = np.nanmedian(dist2_array)
        ystep = np.nanmedian(dist1_array)
    else:
        xstep = np.nanmedian(dist1_array)
        ystep = np.nanmedian(dist2_array)

    ## Define fit parameters
    params = Parameters()
    params.add('ystep', value=ystep, vary=False)
    params.add('xstep', value=xstep, vary=False)
    params.add('theta', value=theta, vary=False)

    ## Optionally perform initial brute search
    if brute_search:
        params.add('y0', value=y0_guess, min=y0_guess-ystep, max=y0_guess+ystep, 
                   vary=True, brute_step=ystep/4.)
        params.add('x0', value=x0_guess, min=x0_guess-xstep, max=x0_guess+xstep, 
                   vary=True, brute_step=xstep/4.)
        minner = Minimizer(fit_error, params, fcn_args=(srcY, srcX, ncols, nrows),
                           fcn_kws={'normalized_shifts' : normalized_shifts,
                                    'ccd_geom' : ccd_geom},
                           nan_policy='omit')
        result = minner.minimize(method='brute', params=params)
        params = result.params
        params['y0'].set(min=y0_guess-ystep/3., max=y0_guess+ystep/3.)
        params['x0'].set(min=x0_guess-xstep/3., max=x0_guess+xstep/3.)

    ## Else perform more constrained search
    else:
        params.add('y0', value=y0_guess, min=y0_guess-ystep/3., max=y0_guess+ystep/3., 
                   vary=True)
        params.add('x0', value=x0_guess, min=x0_guess-xstep/3., max=x0_guess+xstep/3., 
                   vary=True)

    ## Optionally enable parameter fit to theta
    if vary_theta:
        params['theta'].set(vary=True, min=theta-5*np.pi/180., max=theta+5*np.pi/180.)

    ## LM Fit
    minner = Minimizer(fit_error, params, fcn_args=(srcY, srcX, ncols, nrows),
                       fcn_kws={'normalized_shifts' : normalized_shifts,
                                'ccd_geom' : ccd_geom},
                       nan_policy='omit')
    result = minner.minimize(params=params)

    return result

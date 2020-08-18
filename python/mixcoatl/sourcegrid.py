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
from lmfit import Minimizer, Parameters
from scipy import optimize
from scipy.spatial import distance
from itertools import product

class DistortedGrid:

    def __init__(self, ystep, xstep, theta, y0, x0, ncols, nrows, 
                 centroids=None, centroid_shifts=None):

        ## Ideal grid parameters
        self.nrows = nrows
        self.ncols = ncols
        self.ystep = ystep
        self.xstep = xstep
        self.theta = theta
        self.y0 = y0
        self.x0 = x0

        ## Add centroid shifts
        if centroid_shifts is None:
            self._dy = np.zeros(nrows*ncols)
            self._dx = np.zeros(nrows*ncols)
        else:
            self.add_centroid_shifts(centroid_shifts)

        ## Add centroids
        if centroids is None:
            y, x = self.make_source_grid(distorted=False)
            self._y = gY
            self._x = gX
        else:
            y, x = centroids
            self._y = y
            self._x = x

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

            y = hdulist[1].data['SOURCE_Y']
            x = hdulist[1].data['SOURCE_X']
            dy = hdulist[1].data['SOURCE_DY']
            dx = hdulist[1].data['SOURCE_DX']

        return cls(ystep, xstep, theta, y0, x0, nrows, ncols, 
                   centroids=(y, x), centroid_shifts=(dy, dx))
    
    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def add_centroid_shifts(self, centroid_shifts):

        dy, dx = centroid_shifts
        nsources = self.nrows*self.ncols

        if (dy.shape[0]==nsources)*(dx.shape[0]==nsources):
            self._dy = dy
            self._dx = dx

    def make_source_grid(self, distorted=True):

        ## Create a standard nrows x ncols grid of points
        y_array = np.asarray([n*self.ystep - (self.nrows-1)*self.ystep/2. \
                                  for n in range(self.nrows)])
        x_array = np.asarray([n*self.xstep - (self.ncols-1)*self.xstep/2. \
                                  for n in range(self.ncols)])
        Y, X = np.meshgrid(y_array, x_array)
        
        ## Rotate grid using rotation matrix
        Xr = (np.cos(self.theta)*X - np.sin(self.theta)*Y).flatten()
        Yr = (np.sin(self.theta)*X + np.cos(self.theta)*Y).flatten()
        
        ## Move center of grid to desired x/y center coordinates
        Xr += self.x0
        Yr += self.y0

        ## Add centroid shifts
        if distorted:
            Xr += self.dx
            Yr += self.dy
        
        ## Return the flattened arrays
        return Yr, Xr

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
        optic_cols = [fits.Column('SOURCE_Y', array=self.y, format='D'),
                      fits.Column('SOURCE_X', array=self.x, format='D'),
                      fits.Column('SOURCE_DY', array=self.dy, format='D'),
                      fits.Column('SOURCE_DX', array=self.dx, format='D')]
        optic_tablehdu = fits.BinTableHDU.from_columns(optic_cols)

        hdulist = fits.HDUList([prihdu, optic_tablehdu])
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

def fit_error(params, srcY, srcX, nrows, ncols, centroid_shifts=None, ccd_geom=None):
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
                         centroid_shifts=centroid_shifts)    
    gY, gX = grid.make_source_grid()

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
        return distances[:, 0] + (gY.shape[0]-srcY.shape[0])*xstep
    else:
        return distances[:, 0]

def grid_fit(srcY, srcX, y0_guess, x0_guess, ncols, nrows,
             brute_search=False, vary_theta=False, centroid_shifts=None):

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
                           fcn_kws={'centroid_shifts' : centroid_shifts,
                                    'ccd_geom' : ccd_geom})
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
        params['theta'].set(vary=True)

    ## LM Fit
    minner = Minimizer(fit_error, params, fcn_args=(srcY, srcX, ncols, nrows),
                       fcn_kws={'centroid_shifts' : centroid_shifts,
                                'ccd_geom' : ccd_geom},
                       nan_policy='omit')
    result = minner.minimize(params=params)

    return result

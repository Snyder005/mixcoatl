"""Source grid fitting classes and functions.

To Do:
   * Rework DistortedGrid object in context of DM as a calibration object.
"""
from __future__ import print_function
from __future__ import absolute_import
import os
import scipy
import numpy as np
from astropy.io import fits
from lmfit import Minimizer, Parameters, fit_report
from scipy import optimize
from scipy.spatial import ConvexHull, convex_hull_plot_2d, distance
from itertools import product
from astropy.table import Table, Column

import lsst.afw.fits as afwFits

class DistortedGrid:

    def __init__(self, ystep, xstep, theta, y0, x0, ncols=49, nrows=49, normalized_shifts=None):

        ## Ideal grid parameters
        self.nrows = nrows
        self.ncols = ncols
        self.ystep = ystep
        self.xstep = xstep
        self.theta = theta
        self.y0 = y0
        self.x0 = x0

        self._norm_dy = np.zeros(nrows*ncols)
        self._norm_dx = np.zeros(nrows*ncols)

        ## Add centroid shifts
        if normalized_shifts is not None:
            norm_dy, norm_dx = normalized_shifts
            self.set_normalized_shifts(norm_dy, norm_dx)

    @classmethod
    def read_ecvs(cls, filename):
        """Read a ECSV ASCII table from a regular file.

        Parameters
        ----------
        filename : `str`
            Name of the file to read.

        Returns
        -------
        grid : `mixcoatl.sourcegrid.DistortedGrid`
            Resulting distorted grid of sources.
        """
        table = Table.read(filename, format='ascii.ecsv')

        return cls.from_astropy(table)

    @classmethod
    def read_fits(cls, filename, hdu=afwFits.DEFAULT_HDU):
        """Read a FITS binary table from a regular file.

        Parameters
        ----------
        filename : `str`
            Name of the file to read.
        hdu : `int`
            Number of the "header-data unit" to read (where 0 is the Primary HDU).
        
        Returns
        -------
        grid : `mixcoatl.sourcegrid.DistortedGrid`
            Resulting distorted grid of sources.
        """
        table = Table.read(infile, hdu=hdu, format='fits')

        return cls.from_astropy(table)

    @classmethod
    def from_astropy(cls, table):
        """Initialize from an Astropy Table.

        Parameters
        ----------
        table : `astropy.table.Table`

        Returns
        -------
        grid : `mixcoatl.sourcegrid.DistortedGrid`
            Resulting distorted grid of sources.
        """
        meta = table.meta

        y0 = meta['GRID_Y0']
        x0 = meta['GRID_X0']
        theta = meta['GRID_THETA']
        ystep = meta['GRID_YSTEP']
        xstep = meta['GRID_XSTEP']
        nrows = meta['GRID_NROWS']
        ncols = meta['GRID_NCOLS']

        norm_dy = np.zeros(nrows*ncols)
        norm_dx = np.zeros(nrows*ncols)

        all_grid_index = table['spotgrid_index']
        all_norm_dy = table['spotgrid_normalized_dy']
        all_norm_dx = table['spotgrid_normalized_dx']

        select = all_grid_index >= 0

        norm_dy[all_grid_index[select]] = all_norm_dy[select]
        norm_dx[all_grid_index[select]] = all_norm_dx[select]

        return cls(ystep, xstep, theta, y0, x0, nrows, ncols, 
                   normalized_shifts=(norm_dy, norm_dx))

    @property
    def norm_dx(self):
        return self._norm_dx

    @property
    def norm_dy(self):
        return self._norm_dy

    def set_normalized_shifts(self, norm_dy, norm_dx):
        """Replace existing normalized shifts.
        
        Parameters
        ----------
        norm_dy : `numpy.ndarray`
            An array of Y-axis normalized shifts.
        norm_dx : `numpy.ndarray`
            An array of X-axis normalized shifts.

        Raises
        ------
        ValueError
            Raised if normalized shift array lengths do not match total number of 
            sources.
        """
        nsources = self.nrows*self.ncols

        if (norm_dy.shape[0]==nsources)*(norm_dx.shape[0]==nsources):
            self._norm_dy = norm_dy
            self._norm_dx = norm_dx
        else:
            raise ValueError('Array lengths do not match: ({0}, {1}), ({2}, {2})'\
                .format(norm_dy.shape[0], norm_dx.shape[0], nrows*ncols))

    def as_astropy(self):
        """Create a table with distorted grid parameters.
        
        Returns
        -------
        table : `astropy.table.Table`
            Table containing distorted grid parameters.
        """
        meta = {'GRID_Y0' : self.y0,
                'GRID_X0' : self.x0,
                'GRID_YSTEP' : self.ystep,
                'GRID_XSTEP' : self.xstep,
                'GRID_THETA' : self.theta,
                'GRID_NROWS' : self.nrows,
                'GRID_NCOLS' : self.ncols}

        cols = [Column(range(nrows*ncols), name='spotgrid_index', dtype='>i4'),
                Column(self._norm_dy, name='spotgrid_normalized_dy', dtype='>f8'),
                Column(self._norm_dx, name='spotgrid_normalized_dx', dtype='>f8')]

        table = Table(cols, meta=meta)

        return table

    def get_centroid_shifts(self):
        """Return the centroid shifts given the distorted grid parameters.
        
        Returns
        -------
        dy : `numpy.ndarray`
            An array of Y-axis centroid shifts.
        dx : `numpy.ndarray`
            An array of X-axis centroid shifts.
        """
        dx = (np.cos(self.theta)*self.norm_dx - np.sin(self.theta)*self.norm_dy)*self.xstep
        dy = (np.sin(self.theta)*self.norm_dx + np.cos(self.theta)*self.norm_dy)*self.ystep

        return dy, dx

    def get_centroids(self, include_centroid_shifts=False):
        """Get source centroids given the distorted grid parameters.
        
        Parameters
        ----------
        include_coordinate_shifts : `bool`
            `True` if coordinate shifts are applied to centroids.

        Returns
        -------
        y : `numpy.ndarray`
            An array of Y-axis centroid coordinates.
        x : `numpy.ndarray`
            An array of X-axis centroid coordinates.
        """
        ## Make rectilinear grid
        row_spacings = np.asarray([n*self.ystep - (self.nrows-1)*self.ystep/2. \
                                  for n in range(self.nrows)])
        column_spacings = np.asarray([n*self.xstep - (self.ncols-1)*self.xstep/2. \
                                  for n in range(self.ncols)])
        rect_y, rect_x = np.meshgrid(row_spacings, column_spacings)

        rect_y = rect_y.flatten()
        rect_x = rect_x.flatten()

        ## Optionally add scaled centroid shifts
        if include_centroid_shifts:
            rect_y += self.norm_dy*self.ystep
            rect_x += self.norm_dx*self.xstep
        
        ## Rotate coordinates
        y = np.sin(self.theta)*rect_x + np.cos(self.theta)*rect_y
        x = np.cos(self.theta)*rect_x - np.sin(self.theta)*rect_y
        
        ## Translate coordinates
        y += self.y0
        x += self.x0
        
        return y, x

    def write_ecvs(self, filename):
        """Write a `DistortedGrid` to a ASCII ECSV file.

        Parameters
        ----------
        filename : `str`
            Name of the file to write.
        """
        table = self.as_astropy()
        table.write(filename, format='ascii.ecsv')

    def write_fits(self, outfile, overwrite=False):
        """Write a `DistortedGrid` to a regular multi-extension FITS file.

        Parameters
        ----------
        filename : `str`
            Name of the file to write.
        overwrite : `bool`, optional
            If `True` overwrite existing file. The default is `True`.
        """
        table = self.as_astropy()
        table.write(filename, format='fits', overwrite=overwrite)

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
              bbox=None):
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
    gY, gX = grid.get_centroids(include_centroid_shifts=True)

    ## Filter source grid positions according to CCD geometry
    if bbox is not None:
        ymin = 0
        ymax = bbox.getHeight()
        xmin = 0
        xmax = bbox.getWidth()

        mask = (gY < ymax)*(gY > ymin)*(gX < xmax)*(gX > xmin)
        gY = gY[mask]
        gX = gX[mask]

    ## Calculate residuals   
    indices, distances = coordinate_distances(srcY, srcX, gY, gX)

    return distances[:, 0]

def grid_fit(srcY, srcX, ncols, nrows, vary_theta=False,
             method='least_squares', normalized_shifts=None, bbox=None):

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

    ## Find initial guess for grid center based on orientation
    grid_center_guess = find_midpoint_guess(srcY, srcX, xstep, ystep, theta)
    y0_guess, x0_guess = grid_center_guess[1], grid_center_guess[0]    
    
    ## Define fit parameters
    params = Parameters()
    params.add('ystep', value=ystep, vary=False)
    params.add('xstep', value=xstep, vary=False)
    params.add('y0', value=y0_guess, min=y0_guess-3., max=y0_guess+3., vary=True)
    params.add('x0', value=x0_guess, min=x0_guess-3., max=x0_guess+3., vary=True)
    params.add('theta', value=theta, min=theta-0.5*np.pi/180., max=theta+0.5*np.pi/180., vary=False)
    
    minner = Minimizer(fit_error, params, fcn_args=(srcY, srcX, ncols, nrows),
                       fcn_kws={'normalized_shifts' : normalized_shifts,
                                'bbox' : bbox}, nan_policy='omit')
    result = minner.minimize(params=params, method=method, max_nfev=None)
    x0result = result.params['x0']
    y0result = result.params['y0']

    if vary_theta:
        result_params = result.params
        result_values = result_params.valuesdict()
        params['y0'].set(value=result_values['y0'], vary=False)
        params['x0'].set(value=result_values['x0'], vary=False)
        params['theta'].set(vary=True)
        theta_minner = Minimizer(fit_error, params, fcn_args=(srcY, srcX, ncols, nrows),
                       fcn_kws={'normalized_shifts' : normalized_shifts,
                                'bbox' : bbox}, nan_policy='omit')
        theta_result = theta_minner.minimize(params=params, method=method, max_nfev=None)
        result.params['theta'] = theta_result.params['theta']
        
    parvals = result.params.valuesdict()
    grid = DistortedGrid(parvals['ystep'], parvals['xstep'], 
                         parvals['theta'], parvals['y0'], 
                         parvals['x0'], ncols, nrows, 
                         normalized_shifts=normalized_shifts)
    
    return grid, result


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns the corners of the bounding box with the smallest area.

    r is a rotation matrix for the rectangle
    rval is a 4x2 matrix of coordinates for the corners of the box.
    
    Used to calculate initial guess for grid center.
    """

    # Get the convex hull for the points
    simplicies = points[ConvexHull(points).vertices]

    # Calculate edge angles
    edge = np.zeros((len(simplicies)-1, 2))
    edge = simplicies[1:] - simplicies[:-1]

    angles = np.zeros((len(edge)))
    angles = np.arctan2(edge[:, 1], edge[:, 0])

    angles = np.abs(np.mod(angles, np.pi/2.))
    angles = np.unique(angles)

    # Find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-np.pi/2.),
        np.cos(angles+np.pi/2.),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # Apply rotations to the hull
    rot_points = np.dot(rotations, simplicies.T)

    # Find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # Find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    idxs = np.argmin(areas)

    # Return the best box
    x1 = max_x[idxs]
    x2 = min_x[idxs]
    y1 = max_y[idxs]
    y2 = min_y[idxs]
    r = rotations[idxs]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval, r


def find_midpoint_guess(Y, X, xstep, ystep, theta):
    """Calculate an initial midpoint guess. Works for side, corner, 
       and full grid exposure"""
    
    # Get the star positions
    points = [[x,y] for x,y in zip(X,Y)]
    points_centroid = np.mean(points, axis=0)
    
    # Get the minimum area bounding rectangle corners
    # and rotation matrix
    rval, r = minimum_bounding_rectangle(np.array(points))
    rval_centroid = rval.mean(axis=0)

    # Rotate the bounding rectangle back to calculate which corner to use 
    # as a fiducial point
    rect = np.zeros((4, 2))
    rt=[[row[i] for row in r] for i in range(2)]
    rect[0] = np.dot(rval[0]-rval_centroid, rt)
    rect[1] = np.dot(rval[1]-rval_centroid, rt)
    rect[2] = np.dot(rval[2]-rval_centroid, rt)
    rect[3] = np.dot(rval[3]-rval_centroid, rt)

    # Calculate median grid spacing
    d_x = (49 - 1)*xstep
    d_y = (49 - 1)*ystep

    # Calculate guess
    if points_centroid[0] >= 2000 and points_centroid[1] >= 2000: # Top Right
        corner = np.dot([min(rect[:,0]), min(rect[:,1])], r) + rval_centroid
        guess = corner + np.dot([d_x/2., d_y/2.], r)
    elif points_centroid[0] <= 2000 and points_centroid[1] >= 2000: # Top Left
        corner = np.dot([max(rect[:,0]), min(rect[:,1])], r) + rval_centroid
        guess = corner + np.dot([-d_x/2., d_y/2.], r)
    elif points_centroid[0] >= 2000 and points_centroid[1] <= 2000: # Bottom Right
        corner = np.dot([min(rect[:,0]), max(rect[:,1])], r) + rval_centroid
        guess = corner + np.dot([d_x/2., -d_y/2.], r)
    elif points_centroid[0] <= 2000 and points_centroid[1] <= 2000: # Bottom Left
        corner = np.dot([max(rect[:,0]), max(rect[:,1])], r) + rval_centroid
        guess = corner + np.dot([-d_x/2., -d_y/2.], r)
    else:
        print("Unable to locate grid position.")
        guess = None

    return guess

def fit_check(srcX, srcY, gX, gY):
    """Returns the X & Y residuals of the fit, number of identified stars, 
       number of missing stars (ideal grid points without corresponding stars),
       and the X/Y coordinates of outliers (<5th and >95th percentile)."""

    residualsX = []
    residualsY = []
    identified_points = [[x,y] for x,y in zip(srcX, srcY)]
    sourcegrid_points = [[x,y] for x,y in zip(gX, gY)]
    
    # Total number of identified stars
    nsources = len(identified_points)
    
    # Calculate residuals
    countx = 0
    county = 0
    for ipt in identified_points:
        closest = sorted(sourcegrid_points, key=lambda pt : distance.euclidean(pt, ipt))[0]
        residualsX.append(closest[0]-ipt[0])
        residualsY.append(closest[1]-ipt[1])

    # Calculate the outlier points
    outliersX = np.array(identified_points)
    outliersY = np.array(identified_points)
    rX = np.array(residualsX)
    rY = np.array(residualsY)
    outliersX = outliersX[(rX < np.quantile(rX, 0.05)) | (rX > np.quantile(rX, 0.95))].tolist()
    outliersY = outliersY[(rY < np.quantile(rY, 0.05)) | (rY > np.quantile(rY, 0.95))].tolist()
    
    # Calculate number of identified stars have a corresponding ideal grid point within 40px
    count = 0
    for sgpt in sourcegrid_points:
        closest = sorted(identified_points, key=lambda pt : distance.euclidean(pt, sgpt))[0]
        if distance.euclidean(closest, sgpt) < 40: count += 1

    return residualsX, residualsY, nsources, count, outliersX, outliersY

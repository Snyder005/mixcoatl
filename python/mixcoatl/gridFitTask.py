import os
import numpy as np
from os.path import join
from astropy.io import fits
from scipy.spatial import ConvexHull, convex_hull_plot_2d, distance
from scipy.ndimage.interpolation import rotate

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from .sourcegrid import DistortedGrid, grid_fit, coordinate_distances
from .utils import ITL_AMP_GEOM, E2V_AMP_GEOM

class GridFitConfig(pexConfig.Config):
    """Configuration for GridFitTask."""

    nrows = pexConfig.Field("Number of grid rows.", int, default=49)
    ncols = pexConfig.Field("Number of grid columns.", int, default=49)
    y_kwd = pexConfig.Field("Source catalog y-position keyword", str, 
                            default='base_SdssCentroid_Y')
    x_kwd = pexConfig.Field("Source catalog y-position keyword", str, 
                            default='base_SdssCentroid_X')
    vary_theta = pexConfig.Field("Vary theta parameter during fit", bool,
                                 default=False)
    fit_method = pexConfig.Field("Method for fit", str,
                                 default='least_squares')
    outfile = pexConfig.Field("Output filename", str, default="test.cat")

class GridFitTask(pipeBase.Task):

    ConfigClass = GridFitConfig
    _DefaultName = "GridFitTask" 

    srcY = []
    srcX = []
    
    @pipeBase.timeMethod
    def run(self, infile, ccd_type=None, optics_grid_file=None):

        ## Keywords for catalog
        x_kwd = self.config.x_kwd
        y_kwd = self.config.y_kwd
        xx_kwd = 'base_SdssShape_XX'
        yy_kwd = 'base_SdssShape_YY'

        ## Get CCD geometry
        if ccd_type == 'ITL':
            ccd_geom = ITL_AMP_GEOM
        elif ccd_type == 'E2V':
            ccd_geom = E2V_AMP_GEOM
        else:
            ccd_geom = None

        ## Get source positions for fit
        with fits.open(infile) as src:

            all_srcY = src[1].data[y_kwd]
            all_srcX = src[1].data[x_kwd]
            points = np.asarray([[y,x] for y,x in zip(all_srcY,all_srcX)])
            
            # Mask the bad grid points
            quality_mask = (src[1].data['base_SdssShape_XX'] > 4.5)*(src[1].data['base_SdssShape_XX'] < 7.) \
                * (src[1].data['base_SdssShape_YY'] > 4.5)*(src[1].data['base_SdssShape_YY'] < 7.)

            # Mask points without at least two neighbors
            outlier_mask = []
            for pt in points:
                distances = np.sort([distance.euclidean(pt,p) for p in points])[1:]
                count = 0
                for d in distances:
                    if(d > 40) & (d < 100.) & (count < 2): 
                        count = count + 1
                    else:
                        break
                outlier_mask.append(count >= 2)

            full_mask = [m1 & m2 for m1,m2 in zip(quality_mask, outlier_mask)]

            self.srcY = all_srcY[full_mask]
            self.srcX = all_srcX[full_mask]

            ## Find initial guess for grid center based on orientation
            grid_center_guess = find_midpoint_guess(self.srcY, self.srcX, ccd_type)
            y0_guess, x0_guess = grid_center_guess[1], grid_center_guess[0]
            
            ## Optionally get existing normalized centroid shifts
            if optics_grid_file is not None:
                optics_grid = DistortedGrid.from_fits(optics_grid_file)
                normalized_shifts = (optics_grid.norm_dy, optics_grid.norm_dx)
            else:
                normalized_shifts = None
                
            ## Perform grid fit
            ncols = self.config.ncols
            nrows = self.config.nrows
            result = grid_fit(self.srcY, self.srcX, y0_guess, x0_guess, ncols, nrows,
                              vary_theta=self.config.vary_theta,
                              normalized_shifts=normalized_shifts,
                              method=self.config.fit_method,
                              ccd_geom=ccd_geom)

            ## Make best fit source grid
            parvals = result.params.valuesdict()
            grid = DistortedGrid(parvals['ystep'], parvals['xstep'], 
                                 parvals['theta'], parvals['y0'], 
                                 parvals['x0'], ncols, nrows, 
                                 normalized_shifts=normalized_shifts)

            ## Match grid to catalog
            gY, gX = grid.get_source_centroids()
            indices, dist = coordinate_distances(gY, gX, all_srcY, all_srcX)
            nn_indices = indices[:, 0]

            ## Populate grid information
            grid_index = np.full(all_srcX.shape[0], np.nan)
            grid_y = np.full(all_srcX.shape[0], np.nan)
            grid_x = np.full(all_srcX.shape[0], np.nan)
            grid_y[nn_indices] = gY
            grid_x[nn_indices] = gX
            grid_index[nn_indices] = np.arange(49*49)

            ## Merge tables
            new_cols = fits.ColDefs([fits.Column(name='spotgrid_index', 
                                                 format='D', array=grid_index),
                                     fits.Column(name='spotgrid_x', 
                                                 format='D', array=grid_x),
                                     fits.Column(name='spotgrid_y', 
                                                 format='D', array=grid_y)])
            cols = src[1].columns
            new_hdu = fits.BinTableHDU.from_columns(cols+new_cols)
            src[1] = new_hdu

            ## Append grid HDU
            grid_hdu = grid.make_grid_hdu()
            src.append(grid_hdu)
            src.writeto(self.config.outfile, overwrite=True)

        return grid, result


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    r is a rotation matrix for the rectangle
    rval is an nx2 matrix of coordinates
    
    Used to calculate initial guess for grid center.
    """

    # Get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # Calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

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
    rot_points = np.dot(rotations, hull_points.T)

    # Find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # Find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # Return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval, r


def find_gridspacing(srcY, srcX):
    """Calculate the median grid spacing and grid orientation."""

    nsources = len(srcX)
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

    return xstep, ystep, theta

def find_midpoint_guess(Y, X, ccd_type):
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
    xstep,ystep,theta = find_gridspacing(X,Y)
    d_x = (49 - 1)*xstep
    d_y = (49 - 1)*ystep

    # Offset for corresponding CCD coordinates
    if ccd_type == 'ITL':
        geom_offset = -2000.
    else:
        geom_offset = 0.

    # Calculate guess
    if points_centroid[0] >= (2000 + geom_offset) and points_centroid[1] >= 2000: # Top Right
        corner = np.dot([min(rect[:,0]), min(rect[:,1])], r) + rval_centroid
        guess = corner + np.dot([d_x/2., d_y/2.], r)
    elif points_centroid[0] <= (2000 + geom_offset) and points_centroid[1] >= 2000: # Top Left
        corner = np.dot([max(rect[:,0]), min(rect[:,1])], r) + rval_centroid
        guess = corner + np.dot([-d_x/2., d_y/2.], r)
    elif points_centroid[0] >= (2000 + geom_offset) and points_centroid[1] <= 2000: # Bottom Right
        corner = np.dot([min(rect[:,0]), max(rect[:,1])], r) + rval_centroid
        guess = corner + np.dot([d_x/2., -d_y/2.], r)
    elif points_centroid[0] <= (2000 + geom_offset) and points_centroid[1] <= 2000: # Bottom Left
        corner = np.dot([max(rect[:,0]), max(rect[:,1])], r) + rval_centroid
        guess = corner + np.dot([-d_x/2., -d_y/2.], r)
    else:
        print("Unable to locate grid position.")
        guess = None

    return guess

def fit_check(srcX, srcY, gX, gY, ccd_type):
    """Returns the X & Y residuals of the fit, number of identified 
       stars, and number of missing stars (ideal grid points
       without corresponding stars)"""
    arr = []
    residualsX = []
    residualsY = []
    identified_points = [[x,y] for x,y in zip(srcX, srcY)]
    
    # Mask the points to the CCD bounds (not exact for now)
    if ccd_type == 'ITL':
        mask = (gX < 2000.)*(gY < 4000.)*(gX > -2000.)*(gY > 0)
    else:
        mask = (gX < 4000.)*(gY < 4000.)*(gX > 0.)*(gY > 0)

    gX = gX[mask]
    gY = gY[mask]
    
    sourcegrid_points = [[x,y] for x,y in zip(gX, gY)]
    
    # Calculate residuals
    countx = 0
    county = 0
    for ipt in identified_points:
        closest = sorted(sourcegrid_points, key=lambda pt : distance.euclidean(pt, ipt))[0]
        residualsX.append(closest[0]-ipt[0])
        residualsY.append(closest[1]-ipt[1])
        if (closest[0]-ipt[0]) < -100: arr.append(ipt)
    
    # Calculate number of identified stars have a corresponding grid point within 20px
    count = 0
    for sgpt in sourcegrid_points:
        closest = sorted(identified_points, key=lambda pt : distance.euclidean(pt, sgpt))[0]
        if distance.euclidean(closest, sgpt) < 20: count = count + 1
    
    # Total number of identified stars
    nsources = len(residualsX)

    return residualsX, residualsY, nsources, len(sourcegrid_points) - count
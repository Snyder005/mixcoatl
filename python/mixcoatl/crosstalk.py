"""Crosstalk functions and classes.

This module contains a number of function and class definitions that are used
for performing the measurement of electronic crosstalk in multi-segmented CCD
images.

To Do:
    * Modify find_bright_columns to take DM objects as input parameters.
"""
import copy
import numpy as np
from astropy.io import fits

import lsst.afw.image as afwImage
from lsst.afw.detection import FootprintSet, Threshold

def calculate_covariance(exposure, amp1, amp2):
    """Calculate read noise covariance between amplifiers.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Exposure from which to measure read noise covariance.
    amp1 : `lsst.afw.cameraGeom.Amplifier`
        First amplifier to use in covariance calculation.
    amp2 : `lsst.afw.cameraGeom.Amplifier`
        Second amplifier to use in covariance calculation.

    Returns
    -------
    cov : `numpy.ndarray`, (2, 2)
        A 2-d array representing the covariance matrix.
    """
    ccd = exposure.getDetector()

    oscanImage1 = exposure.maskedImage[amp1.getRawHorizontalOverscanBBox()]
    oscanImage2 = exposure.maskedImage[amp2.getRawHorizontalOverscanBBox()]

    arr1 = oscanImage1.getImage().getArray().flatten()
    arr2 = oscanImage2.getImage().getArray().flatten()

    cov = np.cov(np.vstack([arr1, arr2]))

    return cov

def find_bright_columns(imarr, threshold):
    """Find bright columns in an image array.
    
    Parameters
    ----------
    imarr : `numpy.ndarrawy`, (Nx, Ny)
        An array representing an image to analyze.
    threshold : `float`
        Pixel value threshold defining a bright column.

    Returns
    -------
    bright_cols : `list`
        List of column indices corresponding to bright columns.
    """
    image = afwImage.ImageF(imarr)
    
    fp_set = FootprintSet(image, Threshold(threshold))    
    columns = dict([(x, []) for x in range(0, image.getWidth())])
    for footprint in fp_set.getFootprints():
        for span in footprint.getSpans():
            y = span.getY()
            for x in range(span.getX0(), span.getX1()+1):
                columns[x].append(y)
                
    bright_cols = []
    x0 = image.getX0()
    y0 = image.getY0()
    for x in columns:
        if bad_column(columns[x], 20):
            bright_cols.append(x - x0)
    #
    # Sort the output.
    #
    bright_cols.sort()
    
    return bright_cols

def bad_column(column_indices, threshold):
    """Identify bad columns by number of masked pixels.
    
    Parameters
    ----------
    column_indices : `list`
        List of column indices.
    threshold : `int`
        Number of bad pixels required to mark the column as bad.

    Returns
    -------
    is_bad_column : `bool`
        `True` if column is bad, `False` if not.
    """
    if len(column_indices) < threshold:
        # There are not enough masked pixels to mark this as a bad
        # column.
        return False
    # Fill an array with zeros, then fill with ones at mask locations.
    column = np.zeros(max(column_indices) + 1)
    column[(column_indices,)] = 1
    # Count pixels in contiguous masked sequences.
    masked_pixel_count = []
    last = 0
    for value in column:
        if value != 0 and last == 0:
            masked_pixel_count.append(1)
        elif value != 0 and last != 0:
            masked_pixel_count[-1] += 1
        last = value
    if len(masked_pixel_count) > 0 and max(masked_pixel_count) >= threshold:
        return True
    return False

def rectangular_mask(imarr, y_center, x_center, lx, ly):
    """Make a rectangular pixel mask.

    Parameters
    ----------
    imarr : `numpy.ndarray`, (Ny, Nx)
        2-D image pixel array.
    y_center : `int`
        Y-axis position of rectangle center.
    x_center : `int`
        X-axis position of rectangle center.
    lx : `int`
        Length of rectangle along X-axis.
    ly : `int`
        Length of rectangle along Y-axis.

    Returns
    -------
    mask : `numpy.ndarray`, (Ny, Nx)
        2-D mask boolean array.
    """
    Ny, Nx = imarr.shape
    Y, X = np.ogrid[:Ny, :Nx]
    select = (np.abs(Y - y_center) < ly/2.) & (np.abs(X - x_center) < lx/2.)

    return select

def satellite_mask(imarr, angle, distance, width):
    """Make a pixel mask along a target line.

    Parameters
    ----------
    imarr : `numpy.ndarray`, (Ny, Nx)
        2-D image pixel array.
    angle : `float`
        Angle (radians) between the X-axis and the line connecting the origin
        to the closest point on the target line.
    distance : `float`
        Distance from the origin to the closest point on the target line.
    width : `float`
        Width of the mask extending from either side of the target line.

    Returns
    -------
    mask : `numpy.ndarray`, (Ny, Nx)
        2-D mask boolean array.
    """
    Ny, Nx = imarr.shape
    Y, X = np.ogrid[:Ny, :Nx]
    select = np.abs((X*np.cos(angle) + Y*np.sin(angle)) - distance) < width

    return select

def circular_mask(imarr, y_center, x_center, radius):
    """Make a circular pixel mask.

    Parameters
    ----------
    imarr : `numpy.ndarray`, (Ny, Nx)
        2-D image pixel array.
    y_center : `int`
        Y-axis position of circle center.
    x_center : `int`
        X-axis position of circle center.
    radius : `float`
        Radius of the circle.

    Returns
    -------
    mask : `numpy.ndarray`, (Ny, Nx)
        2-D mask boolean array.
    """
    Ny, Nx = imarr.shape
    Y, X = np.ogrid[:Ny, :Nx]
    select = np.sqrt(np.square(Y - y_center) + np.square(X - x_center)) < radius

    return select

def annular_mask(imarr, y_center, x_center, inner_radius, outer_radius):
    """Make an annular pixel mask.

    Parameters
    ----------
    imarr : `numpy.ndarray`, (Ny, Nx)
        2-D image pixel array.
    y_center : `int`
        Y-axis position of annulus center.
    x_center : `int`
        X-axis position of annulus center.
    inner_radius : `float`
        Inner radius of the annulus.
    outer_radius : `float`
        Outer radius of the annulus.

    Returns
    -------
    mask : `numpy.ndarray`, (Ny, Nx)
        2-X mask boolean array.
    """
    if outer_radius <= inner_radius:
        raise ValueError('outer_radius {0.1f} must be greater then inner_radius {1:.1f').format(inner_radius,
                                                                                                outer_radius)
    Ny, Nx = imarr.shape
    Y, X = np.ogrid[:Ny, :Nx]
    R = np.sqrt(np.square(X-x_center) + np.square(Y-y_center))
    select = (R >= inner_radius) & (R < outer_radius)

    return select

def crosstalk_model(params, aggressor_imarr):
    """Create crosstalk victim model.

    Parameters
    ----------
    params : array-like, (4,)
        Input victim model parameters:
        - crosstalk coefficient.
        - Y-axis tilt.
        - X-axis tilt.
        - Constant offset.
    aggressor_imarr : `numpy.ndarray`, (Ny, Nx)
        2-D aggressor image pixel array.

    Returns
    -------
    model : `numpy.ndarray`, (Ny, Nx)
        2-D victim model pixel array.
    """
    ## Model parameters
    crosstalk_coeff = params[0]
    offset_z = params[1]
    tilt_y = params[2]
    tilt_x = params[3]

    ## Construct model
    Ny, Nx = aggressor_imarr.shape
    Y, X = np.mgrid[:Ny, :Nx]
    model = crosstalk_coeff*aggressor_imarr + tilt_y*Y + tilt_x*X + offset_z
    
    return model

def crosstalk_fit(aggressor_array, victim_array, select, covariance,
                  correct_covariance=False, seed=None):
    """Perform crosstalk victim model least-squares minimization.

    Parameters
    ----------
    aggressor_stamp: `numpy.ndarray`, (Ny, Nx)
        2-D aggressor postage stamp pixel array.
    victim_stamp: `numpy.ndarray`, (Ny, Nx)
        2-D victim postage stamp pixel array.
    mask: `numpy.ndarray`, (Ny, Nx)
        2-D mask boolean array.
    covariance : `numpy.ndarray`, (2, 2)
        Covariance between read noise of amplifiers.
    correct_covariance : 'bool'
        Correct covariance between read noise of amplifiers.
    seed : `int`
        Seed to initialize random generator.

    Returns
    -------
    results : `numpy.ndarray`, (10,)
        Results of least-squares minimization:
        - crosstalk coefficient.
        - Y-axis tilt.
        - X-axis tilt.
        - Constant offset.
        - Error estimate for crosstalk coefficient.
        - Error estimate for Y-axis tilt.
        - Error estimate for X-axis tilt.
        - Error estimate for constant offset.
        - Sum of residuals.
        - Reduced degrees of freedom.
    """    
    noise = np.sqrt(np.trace(covariance))
    aggressor_imarr = copy.deepcopy(aggressor_array)
    victim_imarr = copy.deepcopy(victim_array)

    ## Reduce correlated noise
    if correct_covariance:

        diag = np.diag(covariance)
        reverse_covariance = -1*covariance
        np.fill_diagonal(reverse_covariance, diag)

        rng = np.random.default_rng(seed)
        correction = rng.multivariate_normal([0.0, 0.0], reverse_covariance, size=aggressor_imarr.shape)

        aggressor_imarr += correction[:, :, 0]
        victim_imarr += correction[:, :, 1]
        noise *= np.sqrt(2)

    victim_stamp = victim_imarr[select]

    ## Construct masked, compressed basis arrays
    ay, ax = aggressor_imarr.shape
    Z = np.ones((ay, ax))[select]
    Y, X = np.mgrid[:ay, :ax]
    Y = Y[select]
    X = X[select]
    aggressor_stamp = aggressor_imarr[select]

    ## Perform least squares parameter estimation
    b = victim_stamp/noise
    A = np.vstack([aggressor_stamp, Z, Y, X]).T/noise
    params, res, rank, s = np.linalg.lstsq(A, b, rcond=-1)
    covar = np.linalg.inv(np.dot(A.T, A))
    dof = b.shape[0] - 4

    results = np.concatenate((params, np.sqrt(covar.diagonal()), res, [dof]))
    
    return results

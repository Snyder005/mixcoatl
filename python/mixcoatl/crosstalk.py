"""Crosstalk functions and classes.

This module contains a number of function and class definitions that are used
for performing the measurement of electronic crosstalk in multi-segmented CCD
images.
"""
import copy
import numpy as np
from astropy.io import fits

from lsst.eotest.fitsTools import fitsWriteto

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
    raise NotImplementedError

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

class CrosstalkMatrix():

    keys = ['XTALK', 'OFFSET_Z', 'TILT_Y', 'TILT_X',
            'SIGMA_XTALK', 'SIGMA_Z', 'SIGMA_Y', 'SIGMA_X',
            'RESIDUAL', 'DOF']

    def __init__(self, aggressor_id, signal=100000., matrix=None, victim_id=None, namps=16):

        ## Set sensor IDs
        self.aggressor_id = aggressor_id
        if victim_id is not None:
            self.victim_id = victim_id
        else:
            self.victim_id = aggressor_id
        self.namps = namps
        self.signal = signal

        ## Set crosstalk results
        self._matrix = np.full((10, self.namps, self.namps), np.nan)
        if matrix is not None:
            self._matrix = matrix

    @classmethod
    def from_fits(cls, infile):
        """Initialize CrosstalkMatrix from a FITS file."""

        with fits.open(infile) as hdulist:

            aggressor_id = hdulist[0].header['AGGRESSOR']
            victim_id = hdulist[0].header['VICTIM']
            namps = hdulist[0].header['NAMPS']
            signal = hdulist[0].header['SIGNAL']

            matrix = np.full((10, namps, namps), np.nan)
            for i, key in enumerate(cls.keys):
                matrix[i, :, :] = hdulist[key].data

        return cls(aggressor_id, signal=signal, matrix=matrix, victim_id=victim_id, namps=16)

    @property
    def matrix(self):
        return self._matrix

    def set_row(self, aggressor_amp, row_results):
        """Set matrix row from results dictionary."""

        for victim_amp in row_results.keys():
            self._matrix[:, aggressor_amp-1, victim_amp-1] = row_results[victim_amp]

    def set_diagonal(self, value):
        """Set diagonal of matrices to value (e.g. 0.0 or NaN)."""
        
        for i in range(10):
            np.fill_diagonal(self._matrix[i, :, :], value)

    def write_fits(self, outfile, *kwargs):
        """Write crosstalk results to FITS file."""

        ## Make primary HDU
        hdr = fits.Header()
        hdr['AGGRESSOR'] = self.aggressor_id
        hdr['VICTIM'] = self.victim_id
        hdr['NAMPS'] = self.namps
        hdr['SIGNAL'] = self.signal
        prihdu = fits.PrimaryHDU(header=hdr)

        xtalk_hdu = fits.ImageHDU(self._matrix[0,:,:], name='XTALK')
        offsetz_hdu = fits.ImageHDU(self._matrix[1,:,:], name='OFFSET_Z')
        tilty_hdu = fits.ImageHDU(self._matrix[2,:,:], name='TILT_Y')
        tiltx_hdu = fits.ImageHDU(self._matrix[3,:,:], name='TILT_X')
        xtalkerr_hdu = fits.ImageHDU(self._matrix[4,:,:], name='SIGMA_XTALK')
        zerr_hdu = fits.ImageHDU(self._matrix[5,:,:], name='SIGMA_Z')
        yerr_hdu = fits.ImageHDU(self._matrix[6,:,:], name='SIGMA_Y')
        xerr_hdu = fits.ImageHDU(self._matrix[7,:,:], name='SIGMA_X')
        chisq_hdu = fits.ImageHDU(self._matrix[8,:,:], name='RESIDUAL')
        dof_hdu = fits.ImageHDU(self._matrix[9,:,:], name='DOF')
        
        hdulist = fits.HDUList([prihdu, xtalk_hdu, offsetz_hdu, tilty_hdu, 
                                tiltx_hdu, xtalkerr_hdu, zerr_hdu, yerr_hdu, 
                                xerr_hdu, chisq_hdu, dof_hdu])

        hdulist.writeto(outfile, **kwargs)

    def write_yaml(self, outfile):
        """Write crosstalk coefficients to a YAML file."""

        ampNames = [str(i) for i in range(self.matrix.shape[1])]
        assert self.matrix.shape == (10, len(ampNames), len(ampNames))

        dIndent = indent
        indent = 0
        with open(outfile, "w") as fd:
            print(indent*" " + "crosstalk :", file=fd)
            indent += dIndent
            print(indent*" " + "%s :" % crosstalkName, file=fd)
            indent += dIndent

            for i, ampNameI in enumerate(ampNames):
                print(indent*" " + "%s : {" % ampNameI, file=fd)
                indent += dIndent
                print(indent*" ", file=fd, end='')

                for j, ampNameJ in enumerate(ampNames):
                    print("%s : %11.4e, " % (ampNameJ, coeff[i, j]), file=fd,
                          end='\n' + indent*" " if j%4 == 3 else '')
                print("}", file=fd)

                indent -= dIndent

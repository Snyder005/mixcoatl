"""
Analysis tasks for crosstalk.

To Do:
   * Make sure YAML file writing is working properly.
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import lsst.eotest.image_utils as imutils
from copy import deepcopy
from astropy.io import fits

import lsst.afw.math as afwMath
from lsst.eotest.sensor.MaskedCCD import MaskedCCD

from lsst.eotest.fitsTools import fitsWriteto

def make_stamp(imarr, y, x, l=200):
    """Get postage stamp for crosstalk calculations."""

    maxy, maxx = imarr.shape

    y0 = max(0, y-l//2)
    y1 = min(maxy, y+l//2)

    x0 = max(0, x-l//2)
    x1 = min(maxx, x+l//2)

    return deepcopy(imarr[y0:y1, x0:x1])

def calibrated_stack(ccds, amp, gain=1.0):

    ims = [ccd.unbiased_and_trimmed_image(amp) for ccd in ccds]
    imarr = afwMath.statisticsStack(ims, afwMath.MEDIAN).getImage().getArray()*gain

    return imarr

def calculate_noise(bias_frame):

    noise_dict = {}
    bias = MaskedCCD(bias_frame)
    for amp in range(1, 17):
        imarr = bias.unbiased_and_trimmed_image(amp).getImage().getArray()
        noise = imarr.std()
        noise_dict[amp] = noise

    return noise_dict

def crosstalk_model(coefficients, aggressor_array):
    """Create a crosstalk victim postage stamp from model."""

    ny, nx = aggressor_array.shape
    crosstalk_signal = coefficients[0]
    bias = coefficients[1]
    tilty = coefficients[2]
    tiltx = coefficients[3]

    Y, X = np.mgrid[:ny, :nx]
    model = crosstalk_signal*aggressor_array + tilty*Y + tiltx*X + bias
    return model

def crosstalk_model_fit(aggressor_stamp, victim_stamp, num_iter=3, nsig=5.0, noise=7.0):
    """Perform a crosstalk model fit for given  aggressor and victim stamps."""

    coefficients = np.asarray([[0,0,0,0]])
    victim_array = np.ma.masked_invalid(victim_stamp)
    mask = np.ma.getmask(victim_array)

    for i in range(num_iter):
        #
        # Mask outliers using residual
        #
        model = np.ma.masked_where(mask, crosstalk_model(coefficients[0],
                                                         aggressor_stamp))
        residual = victim_array - model
        res_mean = residual.mean()
        res_std = residual.std()
        victim_array = np.ma.masked_where(np.abs(residual-res_mean) \
                                              > nsig*res_std, victim_stamp)
        mask = np.ma.getmask(victim_array)
        #
        # Construct masked, compressed basis arrays
        #
        ay, ax = aggressor_stamp.shape
        bias = np.ma.masked_where(mask, np.ones((ay, ax))).compressed()
        Y, X = np.mgrid[:ay, :ax]
        Y = np.ma.masked_where(mask, Y).compressed()
        X = np.ma.masked_where(mask, X).compressed()
        aggressor_array = np.ma.masked_where(mask, aggressor_stamp).compressed()
        #
        # Perform least-squares minimization
        #
        b = victim_array.compressed()/noise
        A = np.vstack([aggressor_array, bias, Y, X]).T/noise
        coeffs, res, rank, s = np.linalg.lstsq(A, b, rcond=-1)
        covar = np.linalg.inv(np.dot(A.T, A))
        dof = b.shape[0] - 4
        
    return np.concatenate((coeffs, np.sqrt(covar.diagonal()), res, [dof]))

class CrosstalkMatrix():

    def __init__(self, aggressor_id, victim_id=None, filename=None, namps=16):
        self.header = fits.Header()
        self.header.set('AGGRESSOR', aggressor_id)
        if victim_id is not None:
            self.header.set('VICTIM', victim_id)
        else:
            self.header.set('VICTIM', aggressor_id)
        self.filename = filename
        self.namps = namps
        self._set_matrix()
        if self.filename is not None:
            self._read_matrix()

    def set_row(self, aggressor_amp, row):
        """Set matrix row from results dictionary"""
        for victim_amp in row.keys():
            self.matrix[:, aggressor_amp-1, victim_amp-1] = row[victim_amp]

    def _set_matrix(self):
        """Initialize crosstalk matrix as NaNs."""
        self.matrix = np.zeros((10, self.namps, self.namps), dtype=np.float)
        self.matrix[:] = np.nan

    def _read_matrix(self):
        """Read crosstalk matrix from file."""
        if self.filename[-5:] == '.fits':
            self._read_fits_matrix()
        elif self.filename[-5:] == '.yaml':
            self._read_yaml_matrix()
        else:
            raise ValueError('Crosstalk matrix file must be FITS or YAML filetype')

    def _read_fits_matrix(self):
        """Read crosstalk results from FITS file."""
        with fits.open(self.filename) as hdulist:
            for i in range(10):
                self.matrix[i,:,:] = hdulist[i].data

    def _read_yaml_matrix(self):
        """Read crosstalk results from a YAML file."""
        raise NotImplementedError

    def write_fits(self, outfile=None, overwrite=True):
        """Write crosstalk results to FITS file."""
        if outfile is None:
            outfile = self.filename
        else:
            self.filename = outfile
        #
        # Save matrix results into separate HDUs
        #
        xtalk_hdu = fits.PrimaryHDU(self.matrix[0,:,:], header=self.header)
        bias_hdu = fits.ImageHDU(self.matrix[1,:,:], name='BIAS')
        tilty_hdu = fits.ImageHDU(self.matrix[2,:,:], name='TILT_Y')
        tiltx_hdu = fits.ImageHDU(self.matrix[3,:,:], name='TILT_X')
        xtalkerr_hdu = fits.ImageHDU(self.matrix[4,:,:], name='SIGMA_XTALK')
        biaserr_hdu = fits.ImageHDU(self.matrix[5,:,:], name='SIGMA_BIAS')
        tiltyerr_hdu = fits.ImageHDU(self.matrix[6,:,:], name='SIGMA_TILT_Y')
        tiltxerr_hdu = fits.ImageHDU(self.matrix[7,:,:], name='SIGMA_TILT_X')
        chisq_hdu = fits.ImageHDU(self.matrix[8,:,:], name='CHI_SQUARE')
        dof_hdu = fits.ImageHDU(self.matrix[9,:,:], name='DOF')
        
        output = fits.HDUList([xtalk_hdu, bias_hdu, tilty_hdu, tiltx_hdu, 
                               xtalkerr_hdu, biaserr_hdu, tiltyerr_hdu, tiltxerr_hdu,
                               chisq_hdu, dof_hdu])
        fitsWriteto(output, outfile, overwrite=overwrite)

    def write_yaml(self, outfile, indent=2, crosstalkName='Unknown'):
        """Write crosstalk results to a YAML file."""

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

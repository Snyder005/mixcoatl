import numpy as np
from astropy.io import fits

from lsst.eotest.fitsTools import fitsWriteto

def make_stamp(imarr, y, x, l=200):
    """Make an image postage stamp."""

    maxy, maxx = imarr.shape
    
    y0 = max(0, y-l//2)
    y1 = min(maxy, y+l//2)

    x0 = max(0, x-l//2)
    x1 = min(maxx, x+l//2)

    return imarr[y0:y1, x0:x1]

def crosstalk_model(params, aggressor_imarr):
    """Create crosstalk victim model."""

    ## Model parameters
    crosstalk_coeff = params[0]
    offset_z = params[1]
    tilt_y = params[2]
    tilt_x = params[3]

    ## Construct model
    ny, nx = aggressor_imarr.shape
    Y, X = np.mgrid[:ny, :nx]
    model = crosstalk_coeff*aggressor_imarr + tilt_y*Y + tilt_x*X + offset_z
    
    return model

def crosstalk_fit(aggressor_stamp, victim_stamp, num_iter=3, nsig=5.0, noise=7.0):
    """Perform crosstalk model fit."""

    params = np.asarray([0, 0, 0, 0])
    victim_imarr = np.ma.masked_invalid(victim_stamp)
    mask = np.ma.getmask(victim_imarr)

    for i in range(num_iter):

        ## Mask outliers using residual
        model = np.ma.masked_where(mask, crosstalk_model(params, aggressor_stamp))
        
        residual = victim_imarr - model
        res_mean = residual.mean()
        res_std = residual.std()
        victim_imarr = np.ma.masked_where(np.abs(residual - res_mean) \
                                              > nsig*res_std, victim_stamp)
        mask = np.ma.getmask(victim_imarr)

        ## Construct masked, compressed basis arrays
        ay, ax = aggressor_stamp.shape
        Z = np.ma.masked_where(mask, np.ones((ay, ax))).compressed()
        Y, X = np.mgrid[:ay, :ax]
        Y = np.ma.masked_where(mask, Y).compressed()
        X = np.ma.masked_where(mask, X).compressed()
        aggressor_imarr = np.ma.masked_where(mask, aggressor_stamp).compressed()

        ## Perform least squares parameter estimation
        b = victim_imarr.compressed()/noise
        A = np.vstack([aggressor_imarr, Z, Y, X]).T/noise
        params, res, rank, s = np.linalg.lstsq(A, b, rcond=-1)
        covar = np.linalg.inv(np.dot(A.T, A))
        dof = b.shape[0] - 4

    return np.concatenate((params, np.sqrt(covar.diagonal()), res, [dof]))

class CrosstalkMatrix():

    keys = ['XTALK', 'OFFSET_Z', 'TILT_Y', 'TILT_X',
            'SIGMA_XTALK', 'SIGMA_Z', 'SIGMA_Y', 'SIGMA_X',
            'RESIDUAL', 'DOF']

    def __init__(self, aggressor_id, matrix=None, victim_id=None, namps=16):

        ## Set sensor IDs
        self.aggressor_id = aggressor_id
        if victim_id is not None:
            self.victim_id = victim_id
        else:
            self.victim_id = aggressor_id
        self.namps = namps
        
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

            matrix = np.full((10, namps, namps), np.nan)
            for i, key in enumerate(cls.keys):
                matrix[i, :, :] = hdulist[key].data

        return cls(aggressor_id, matrix=matrix, victim_id=victim_id, namps=16)

    @property
    def matrix(self):
        return self._matrix

    def set_row(self, aggressor_amp, row_results):
        """Set matrix row from results dictionary."""

        for victim_amp in row_results.keys():
            self._matrix[:, aggressor_amp-1, victim_amp-1] = row_results[victim_amp]

    def set_diagonal(self, value):
        """Set diagonal of matrices to value (e.g. 0.0 or NaN)""".
        
        for i in range(10):
            np.fill_diagonal(self._matrix[i, :, :], value)

    def write_fits(self, outfile, **kwargs):
        """Write crosstalk results to FITS file."""

        ## Make primary HDU
        hdr = fits.Header()
        hdr['AGGRESSOR'] = self.aggressor_id
        hdr['VICTIM'] = self.victim_id
        hdr['NAMPS'] = self.namps
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

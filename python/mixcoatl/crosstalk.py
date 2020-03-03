"""
Analysis tasks for crosstalk.

To Do:
   * Make sure YAML file writing is working properly.
"""
import os
import glob
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import lsst.eotest.image_utils as imutils
from copy import deepcopy
from astropy.io import fits
import time
import pickle

import siteUtils
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.math as afwMath
from lsst.eotest.sensor.MaskedCCD import MaskedCCD

from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms, channelToAmp, getAmpImage

from lsst.eotest.fitsTools import fitsWriteto

import multiprocessing as mp
import argparse

def make_stamp(imarr, y, x, l=200):
    """Get postage stamp for crosstalk calculations."""

    maxy, maxx = imarr.shape

    y0 = max(0, y-l//2)
    y1 = min(maxy, y+l//2)

    x0 = max(0, x-l//2)
    x1 = min(maxx, x+l//2)

    return deepcopy(imarr[y0:y1, x0:x1])

def calibrated_stack(ccds, amp, dark_ccd=None):

    ims = [ccd.unbiased_and_trimmed_image(amp) for ccd in ccds]
    imarr = afwMath.statisticsStack(ims, afwMath.MEDIAN).getImage().getArray()
    
    if dark_ccd is not None:
        exptime = ccds[0].md.get('EXPTIME')
        dark_exptime = dark_ccd.md.get('EXPTIME')
        dark_imarr = dark_ccd.unbiased_and_trimmed_image(amp).getImage().getArray()
        imarr -= dark_imarr*exptime/dark_exptime

    return imarr

def calculate_noise(bias_frame):

    noise_dict = {}
    bias = MaskedCCD(bias_frame)
    for amp in range(1, 17):
        imarr = bias.unbiased_and_trimmed_image(amp).getImage().getArray()
        noise = imarr.std()
        noise_dict[amp] = noise

    return noise_dict

def calculate_crosstalk(r, amp, ccds, gains, noise, aggressor_stamp, y, x, dark_ccd=None):

    imarr = calibrated_stack(ccds, amp, dark_ccd=dark_ccd)
    victim_stamp = make_stamp(imarr, y, x)*gains[amp]
    
    r[amp] = crosstalk_model_fit(aggressor_stamp, victim_stamp, noise=noise[amp])

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

def crosstalk_model_fit(aggressor_stamp, victim_stamp, num_iter=10, nsig=3.0, noise=7.0):
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

def main(main_dir, calib_dir, aggressor_id, victim_id=None, output_dir='./', 
         use_multiprocessing=False):

    ## Task configurables
    nsig = 2.0
    num_iter = 10
    threshold = 40000.0

    camera = camMapper._makeCamera()
    lct = LsstCameraTransforms(camera)

    ## Make bias dictionary
    raft_list  = ['R01', 'R02', 
                  'R10', 'R11', 'R12', 
                  'R20', 'R21', 'R22',
                  'R30']
    sensor_list = ['S00', 'S01', 'S02', 'S10', 'S11', 'S12', 'S20', 'S21', 'S22']
    bias_dict = {}
    dark_dict = {}
    gains_dict = {}
    gain_results = pickle.load(open(os.path.join(calib_dir, 'et_results.pkl'), 'rb'))
    for raft in raft_list:
        for sensor in sensor_list:
            sensor_id = '{0}_{1}'.format(raft, sensor)
            try:
                bias_frame = glob.glob(os.path.join(calib_dir, '{0}_superbias.fits'.format(sensor_id)))[0]
            except IndexError:
                bias_frame = None
            try:
                dark_frame = glob.glob(os.path.join(calib_dir, '{0}_superdark.fits'.format(sensor_id)))[0]
            except IndexError:
                dark_frame = None
            gains = gain_results.get_amp_gains(sensor_id)
            gains_dict[sensor_id] = gains
            bias_dict[sensor_id] = bias_frame
            dark_dict[sensor_id] = dark_frame

    ## Eventual command line args
    aggressor_gains = gains_dict[aggressor_id]
    aggressor_bias = bias_dict[aggressor_id]
    aggressor_dark = MaskedCCD(dark_dict[aggressor_id], bias_frame=aggressor_bias)
       
    if victim_id is not None:
        victim_bias = bias_dict[victim_id]
        victim_dark = MaskedCCD(dark_dict[victim_id], bias_frame=victim_bias)
        victim_gains = gains_dict[victim_id]
    else:
        victim_id = aggressor_id
        victim_bias = aggressor_bias
        victim_gains = aggressor_gains
        victim_dark = aggressor_dark

    victim_noise = calculate_noise(victim_bias)

    ## Sort directories by central CCD
    directory_list = [x.path for x in os.scandir(main_dir) if os.path.isdir(x.path)]
    xtalk_dict = {}
    for acquisition_dir in directory_list:
        basename = os.path.basename(acquisition_dir)
        if "xtalk" not in basename:
            continue
        xpos, ypos = basename.split('_')[-4:-2]    
        central_sensor, ccdX, ccdY = lct.focalMmToCcdPixel(float(ypos), float(xpos))
        if central_sensor in xtalk_dict:
            xtalk_dict[central_sensor].add((xpos, ypos))
        else:
            xtalk_dict[central_sensor] = {(xpos, ypos)}
    
    outfile = os.path.join(output_dir, '{0}_{1}_crosstalk_results.fits'.format(aggressor_id, victim_id))

    ## For given aggressor get infiles per position
    for i, pos in enumerate(xtalk_dict[aggressor_id]):
        xpos, ypos = pos

        ## Aggressor ccd files
        aggressor_infiles = glob.glob(os.path.join(main_dir, '*{0}_{1}*'.format(xpos, ypos), 
                                                   '*{0}.fits'.format(aggressor_id)))
        aggressor_ccds = [MaskedCCD(infile, bias_frame=aggressor_bias) for infile in aggressor_infiles]

        ## Victim ccd files
        victim_infiles = glob.glob(os.path.join(main_dir, '*{0}_{1}*'.format(xpos, ypos), 
                            '*{0}.fits'.format(victim_id)))
        victim_ccds = [MaskedCCD(infile, bias_frame=victim_bias) for infile in victim_infiles]

        if i == 0:
            crosstalk_matrix = CrosstalkMatrix(aggressor_id, victim_id=victim_id)
        else:
            crosstalk_matrix = CrosstalkMatrix(aggressor_id, victim_id=victim_id, filename=outfile)

        num_aggressors = 0
        for aggressor_amp in range(1, 17):

            aggressor_imarr = calibrated_stack(aggressor_ccds, aggressor_amp, dark_ccd=aggressor_dark)

            ## smooth and find largest peak
            gf_sigma = 20
            smoothed = gaussian_filter(aggressor_imarr, gf_sigma)
            y, x = np.unravel_index(smoothed.argmax(), smoothed.shape)

            ## check that circle centered on peak above threshold
            r = 20
            Y, X = np.ogrid[-y:smoothed.shape[0]-y, -x:smoothed.shape[1]-x]
            mask = X*X + Y*Y >= r*r
            test = np.ma.MaskedArray(aggressor_imarr, mask)
            if np.mean(test) > threshold:
                num_aggressors += 1

                aggressor_stamp = make_stamp(aggressor_imarr, y, x)*aggressor_gains[aggressor_amp]

                ## Optionally use multiprocessing
                if use_multiprocessing:
                    manager = mp.Manager()
                    row = manager.dict()
                    job = [mp.Process(target=calculate_crosstalk, args=(row, i, victim_ccds, 
                                                                        victim_gains, victim_noise, 
                                                                        aggressor_stamp, y, x, 
                                                                        victim_dark)) for i in range(1, 17)]
                    _ = [p.start() for p in job]
                    _ = [p.join() for p in job]
                else:
                    row = {}
                    for i in range(1, 17):
                        calculate_crosstalk(row, i, victim_ccds, victim_gains, victim_noise, 
                                            aggressor_stamp, y, x, victim_dark)
                crosstalk_matrix.set_row(aggressor_amp, row)

                if num_aggressors == 4: break

        crosstalk_matrix.write_fits(outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('aggressor_id', type=str)
    parser.add_argument('main_dir', type=str)
    parser.add_argument('calib_dir', type=str)
    parser.add_argument('-v', '--victim_id', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='./')
    args = parser.parse_args()

    aggressor_id = args.aggressor_id
    main_dir = args.main_dir
    calib_dir = args.calib_dir
    victim_id = args.victim_id
    output_dir = args.output_dir

    main(main_dir, calib_dir, aggressor_id,
         victim_id=victim_id, output_dir=output_dir)

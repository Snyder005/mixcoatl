"""MixCOATL Tasks for crosstalk analysis.

TODO:
    * Change logging implementation to match DM pipe task logging.
    * Add docstrings and confirm compliance with LSP coding style guide.
    * Update CrosstalkMatrix as needed.
"""
import numpy as np
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter
from sqlalchemy.orm.exc import NoResultFound
import logging
from datetime import datetime
import os
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.eotest.image_utils as imutils
from lsst.eotest.sensor.MaskedCCD import MaskedCCD
from lsst.eotest.sensor.BrightPixels import BrightPixels

from mixcoatl.crosstalk import CrosstalkMatrix, rectangular_mask, satellite_mask, crosstalk_fit
from mixcoatl.utils import AMP2SEG, calculate_covariance
from mixcoatl.database import Sensor, Segment, Result, db_session

class InterCCDCrosstalkConfig(pexConfig.Config):
    
    length = pexConfig.Field("Length of postage stamps in y and x direction", int, default=200)
    nsig = pexConfig.Field("Outlier rejection sigma threshold", float, default=5.0)
    num_iter = pexConfig.Field("Number of least square iterations", int, default=3)
    threshold = pexConfig.Field("Aggressor spot mean signal threshold", float, default=40000.)
    verbose = pexConfig.Field("Turn verbosity on", bool, default=True)

class InterCCDCrosstalkTask(pipeBase.Task):

    ConfigClass = InterCCDCrosstalkConfig
    _DefaultName = "InterCCDCrosstalkTask"

    def run(self, sensor_id1, infiles1, gains1, bias_frame1=None, 
            dark_frame1=None, crosstalk_matrix_file=None, **kwargs):

        ## Parse kwargs for separate victim CCD inputs
        try:
            sensor_id2 = kwargs['sensor_id2']
        except KeyError:
            sensor_id2 = sensor_id1
            infiles2 = infiles1
            gains2 = gains1
            bias_frame2 = bias_frame1
            dark_frame2 = dark_frame1
        else:
            infiles2 = kwargs['infiles2']
            gains2 = kwargs['gains2']
            bias_frame2 = kwargs['bias_frame2']
            dark_frame2 = kwargs['dark_frame2']

        infiles_list = [(infiles1[i], infiles2[i]) for i in range(len(infiles1))]
        
        all_amps = imutils.allAmps(infiles1[0])

        ## Create new matrix or modify existing
        if crosstalk_matrix_file is not None:
            crosstalk_matrix = CrosstalkMatrix.from_fits(crosstalk_matrix_file)
            outfile = crosstalk_matrix_file
        else:
            crosstalk_matrix = CrosstalkMatrix(sensor_id1, 
                                               victim_id=sensor_id2,
                                               namps=len(all_amps))
            outfile = self.config.outfile

        for infile1, infile2 in infiles_list:
            ccd1 = MaskedCCD(infile1, bias_frame=bias_frame1, 
                             dark_frame=dark_frame1)
            ccd2 = MaskedCCD(infile2, bias_frame=bias_frame2, 
                             dark_frame=dark_frame2)      
            num_aggressors = 0
            signals = []

            ## Search each amp for aggressor
            for i in all_amps:
                imarr1 = ccd1.unbiased_and_trimmed_image(i).getImage().getArray()*gains1[i]
                smoothed = gaussian_filter(imarr1, 20)
                y, x = np.unravel_index(smoothed.argmax(), smoothed.shape)
                stamp1 = make_stamp(imarr1, y, x)
                ly, lx = stamp1.shape
                Y, X = np.ogrid[-ly/2:ly/2, -lx/2:lx/2]
                mask = X*X + Y*Y <= 20*20
                signal = np.mean(stamp1[mask])

                if signal > self.config.threshold:
                    signals.append(signal)
                    row = {}

                    ## Calculate crosstalk for each victim amp
                    for j in all_amps:
                        imarr2 = ccd2.unbiased_and_trimmed_image(j).getImage().getArray()*gains2[j]

                        stamp2 = make_stamp(imarr2, y, x)
                        row[j] = crosstalk_fit(stamp1, stamp2, noise=7.0,
                                                     num_iter=self.config.num_iter,
                                                     nsig=self.config.nsig)

                    crosstalk_matrix.set_row(i, row)
                    if num_aggressors == self.config.aggressors_per_image: 
                        break

        crosstalk_matrix.signal = np.median(np.asarray(signals))
        if sensor_id1 == sensor_id2:
            crosstalk_matrix.set_diagonal(0.)
        crosstalk_matrix.write_fits(outfile, overwrite=True)

class CrosstalkSpotConfig(pipeBase.PipelineTaskConfig,
                          pipelineConnections=pipeBase.PipelineTaskConnections):

    database = pexConfig.Field(
        dtype=str,
        doc="SQL database DB file.", 
        default='crosstalk.db'
    )
    maskLengthX = pexConfig.Field(
        dtype=int,
        doc="Length of rectangular mask in x-direction.",
        default=200
    )
    maskLengthY = pexConfig.Field(
        dtype=int,
        doc="Length of rectangular mask in y-direction.", 
        default=200
    )
    threshold = pexConfig.Field(
        dtype=float,
        doc="Aggressor spot mean signal threshold.", 
        default=50000.
    )
    covarianceCorrectionType = pexConfig.ChoiceField(
        dtype=str,
        doc="The method for correcting correlated read noise.",
        default='NONE',
        allowed={
            "NONE": "No correction performed.",
            "RANDOM": "Correct using a random seed.",
            "SEED" : "Correct using a fixed seed."
        }
    )

class CrosstalkSpotTask(pipeBase.Task):

    ConfigClass = CrosstalkSpotConfig
    _DefaultName = "CrosstalkSpotTask"

    def run(self, sensor_name, infiles, bias_frame=None, dark_frame=None, linearity_correction=None):

        if not isinstance(infiles, list):
            infiles = [infiles]

        if self.config.covarianceCorrectionType == 'NONE':
            correct_covariance = False
            seed = None
        elif self.config.covarianceCorrectionType == 'RANDOM':
            correct_covariance = True
            seed = None
        elif self.config.covarianceCorrectionType == 'SEED':
            correct_covariance = True
            seed = 189

        ## Get sensor information from header
        all_amps = imutils.allAmps(infiles[0])
        with fits.open(infiles[0]) as hdulist:
            lsst_num = hdulist[0].header['LSST_NUM']
            teststand = hdulist[0].header['TSTAND']
            manufacturer = lsst_num[:3]

        ## Interface with SQL database
        database = self.config.database
        with db_session(database) as session:

            ## Get sensor from database
            try:
                sensor = Sensor.from_db(session, sensor_name=sensor_name)
            except NoResultFound:
                sensor = Sensor(sensor_name=sensor_name, lsst_num=lsst_num, manufacturer=manufacturer, 
                                namps=len(all_amps))
                sensor.segments = {i : Segment(segment_name=AMP2SEG[i], amplifier_number=i) for i in all_amps}
                sensor.add_to_db(session)
                session.commit()

            ## Set configuration and analysis settings
            if len(infiles) > 1:
                is_coadd = True
            else:
                is_coadd = False
            ccds = [MaskedCCD(infile, bias_frame=bias_frame, dark_frame=dark_frame,
                              linearity_correction=linearity_correction) for infile in infiles]

            ## Aggressor amplifiers
            for i in all_amps:

                aggressor_images = [ccd.unbiased_and_trimmed_image(i).getImage() for ccd in ccds]
                aggressor_imarr = imutils.stack(aggressor_images).getArray()

                ## Find aggressor regions
                smoothed = gaussian_filter(aggressor_imarr, 20)
                y, x = np.unravel_index(smoothed.argmax(), smoothed.shape)
                mask = rectangular_mask(aggressor_imarr, y, x, ly=self.config.maskLengthY, 
                                        lx=self.config.maskLengthX)
                signal = np.max(smoothed)
                if signal < self.config.threshold:
                    continue
    
                ## Victim amplifiers
                for j in all_amps:

                    covariance = calculate_covariance(ccds[0], i, j)/float(len(ccds))
                    victim_images = [ccd.unbiased_and_trimmed_image(j).getImage() for ccd in ccds]
                    victim_imarr = imutils.stack(victim_images).getArray()

                    ## Add crosstalk result to database
                    res = crosstalk_fit(aggressor_imarr, victim_imarr, mask, covariance=covariance,
                                        correct_covariance=correct_covariance, seed=seed)
                    result = Result(aggressor_id=sensor.segments[i].id, victim_id=sensor.segments[j].id,
                                    aggressor_signal=signal, coefficient=res[0], error=res[4],
                                    methodology='MODEL_LSQ', teststand=teststand, image_type='spot',
                                    analysis='CrosstalkSpotTask', is_coadd=is_coadd, z_offset=res[1], 
                                    y_tilt=res[2], x_tilt=res[3])
                    result.add_to_db(session)
                    
class CrosstalkColumnConfig(pipeBase.PipelineTaskConfig,
                            pipelineConnections=pipeBase.PipelineTaskConnections):

    database = pexConfig.Field(
        dtype=str,
        doc="SQL database DB file.", 
        default='crosstalk.db'
    )
    maskLengthX = pexConfig.Field(
        dtype=int,
        doc="Length of rectangular mask in x-direction.",
        default=100
    )
    maskLengthY = pexConfig.Field(
        dtype=int,
        doc="Length of postage stamp mask in y-direction.", 
        default=2000
    )
    threshold = pexConfig.Field(
        dtype=float,
        doc="Aggressor column mean signal threshold.",
        default=1000.
    )
    covarianceCorrectionType = pexConfig.ChoiceField(
        dtype=str,
        doc="The method for correcting correlated read noise.",
        default='NONE',
        allowed={
            "NONE": "No correction performed.",
            "RANDOM": "Correct using a random seed.",
            "SEED" : "Correct using a fixed seed."
        }
    )

class CrosstalkColumnTask(pipeBase.PipelineTask):

    ConfigClass = CrosstalkColumnConfig
    _DefaultName = "CrosstalkColumnTask"

    def run(self, sensor_name, infiles, bias_frame=None, dark_frame=None, linearity_correction=None):

        if not isinstance(infiles, list):
            infiles = [infiles]

        if self.config.covarianceCorrectionType == 'NONE':
            correct_covariance = False
            seed = None
        elif self.config.covarianceCorrectionType == 'RANDOM':
            correct_covariance = True
            seed = None
        elif self.config.covarianceCorrectionType == 'SEED':
            correct_covariance = True
            seed = 189

        ## Get sensor information from header
        all_amps = imutils.allAmps(infiles[0])
        with fits.open(infiles[0]) as hdulist:
            lsst_num = hdulist[0].header['LSST_NUM']
            teststand = hdulist[0].header['TSTAND']
            manufacturer = lsst_num[:3]

        ## Interface with SQL database
        database = self.config.database
        with db_session(database) as session:

            ## Get sensor from database
            try:
                sensor = Sensor.from_db(session, sensor_name=sensor_name)
            except NoResultFound:
                sensor = Sensor(sensor_name=sensor_name, lsst_num=lsst_num, manufacturer=manufacturer,
                                namps=len(all_amps))
                sensor.segments = {i : Segment(segment_name=AMP2SEG[i], amplifier_number=i) for i in all_amps}
                sensor.add_to_db(session)
                session.commit()

            if len(infiles) > 1:
                is_coadd = True
            else:
                is_coadd = False
            ccds = [MaskedCCD(infile, bias_frame=bias_frame, dark_frame=dark_frame,
                              linearity_correction=linearity_correction) for infile in infiles]

            ## Aggressor amplifiers
            for i in all_amps:

                ## Find aggressor regions
                exptime = 1
                gain = 1
                bp = BrightPixels(ccds[0], i, exptime, gain, ethresh=self.config.threshold)
                pixels, columns = bp.find()
                if len(columns) == 0:
                    continue
                col = columns[0]

                aggressor_images = [ccd.unbiased_and_trimmed_image(i).getImage() for ccd in ccds]
                aggressor_imarr = imutils.stack(aggressor_images).getArray()
                signal = np.mean(aggressor_imarr[:, col])
                mask = rectangular_mask(aggressor_imarr, 1000, col, ly=self.config.maskLengthY,
                                        lx=self.config.maskLengthX)
                
                ## Victim amplifiers
                for j in all_amps:

                    covariance = calculate_covariance(ccds[0], i, j)/float(len(ccds))
                    victim_images = [ccd.unbiased_and_trimmed_image(j).getImage() for ccd in ccds]
                    victim_imarr = imutils.stack(victim_images).getArray()
                    
                    ## Add crosstalk result to database
                    res = crosstalk_fit(aggressor_imarr, victim_imarr, mask, covariance=covariance,
                                        correct_covariance=correct_covariance, seed=seed)
                    result = Result(aggressor_id=sensor.segments[i].id, victim_id=sensor.segments[j].id,
                                    aggressor_signal=signal, coefficient=res[0], error=res[4], 
                                    methodology='MODEL_LSQ', image_type='brightcolumn',
                                    teststand=teststand, analysis='CrosstalkColumnTask', is_coadd=is_coadd,
                                    z_offset=res[1], y_tilt=res[2], x_tilt=res[3])
                    result.add_to_db(session)

class CrosstalkSatelliteConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=pipeBase.PipelineTaskConnections):

    database = pexConfig.Field(
        dtype=str,
        doc="SQL database DB file.", 
        default='crosstalk.db'
    )
    maskWidth = pexConfig.Field(
        dtype=int,
        doc="Width of satellite streak mask.", 
        default=100
    )
    cannySigma = pexConfig.Field(
        dtype=float,
        doc="Gaussian smoothing sigma for Canny edge detection.", 
        default=15.
    )
    thresholdLow = pexConfig.Field(
        dtype=float,
        doc="Low threshold for Canny edge detection.", 
        default=1
    )
    thresholdHigh = pexConfig.Field(
        dtype=float,
        doc="High threshold for Canny edge detection.", 
        default=15
    )
    restrictSide = pexConfig.Field(
        dtype=bool,
        doc="Restrict crosstalk measurement to segment pairs on a single side", 
        default=True
    )
    covarianceCorrectionType = pexConfig.ChoiceField(
        dtype=str,
        doc="The method for correcting correlated read noise.", 
        default='NONE',
        allowed={
            "NONE": "No correction performed.",
            "RANDOM": "Correct using a random seed.",
            "SEED" : "Correct using a fixed seed."
        }
    )

class CrosstalkSatelliteTask(pipeBase.Task):

    ConfigClass = CrosstalkSatelliteConfig
    _DefaultName = "CrosstalkSatelliteTask"

    def run(self, sensor_name, infiles, bias_frame=None, dark_frame=None, linearity_correction=None):

        if not isinstance(infiles, list):
            infiles = [infiles]

        if self.config.covarianceCorrectionType == 'NONE':
            correct_covariance = False
            seed = None
        elif self.config.covarianceCorrectionType == 'RANDOM':
            correct_covariance = True
            seed = None
        elif self.config.covarianceCorrectionType == 'SEED':
            correct_covariance = True
            seed = 189

        ## Get sensor information from header
        all_amps = imutils.allAmps(infiles[0])
        with fits.open(infiles[0]) as hdulist:
            lsst_num = hdulist[0].header['LSST_NUM']
            teststand = hdulist[0].header['TSTAND']
            manufacturer = lsst_num[:3]

        ## Interface with SQL database
        database = self.config.database
        with db_session(database) as session:

            ## Get sensor from database
            try:
                sensor = Sensor.from_db(session, sensor_name=sensor_name)
            except NoResultFound:
                sensor = Sensor(sensor_name=sensor_name, lsst_num=lsst_num, manufacturer=manufacturer,
                                namps=len(all_amps))
                sensor.segments = {i : Segment(segment_name=AMP2SEG[i], amplifier_number=i) for i in all_amps}
                sensor.add_to_db(session)
                session.commit()

            if len(infiles) > 1:
                is_coadd = True
            else:
                is_coadd = False
            ccds = [MaskedCCD(infile, bias_frame=bias_frame, dark_frame=dark_frame,
                              linearity_correction=linearity_correction) for infile in infiles]

            ## Aggressor amplifiers
            for i in all_amps:

                aggressor_images = [ccd.unbiased_and_trimmed_image(i).getImage() for ccd in ccds]
                aggressor_imarr = imutils.stack(aggressor_images).getArray()

                ## Find aggressor regions
                tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 1000)
                edges = feature.canny(aggressor_imarr, sigma=self.config.cannySigma, 
                                      low_threshold=self.config.thresholdLow, 
                                      high_threshold=self.config.thresholdHigh)
                h, theta, d = hough_line(edges, theta=tested_angles)
                _, angle, dist = hough_line_peaks(h, theta, d)

                if len(angle) != 2:
                    continue

                mean_angle = np.mean(angle)
                mean_dist = np.mean(dist)
                mask = satellite_mask(aggressor_imarr, mean_angle, mean_dist, width=self.config.maskWidth)
                signal = np.max(aggressor_imarr[~mask])
                
                ## Victim amplifiers
                if self.config.restrictSide:
                    if i < 9:
                        vic_amps = range(1, 9)
                    else:
                        vic_amps = range(9, 17)
                else:
                    vic_amps = all_amps

                for j in vic_amps:

                    covariance = calculate_covariance(ccds[0], i, j)/float(len(ccds))
                    victim_images = [ccd.unbiased_and_trimmed_image(j).getImage() for ccd in ccds]
                    victim_imarr = imutils.stack(victim_images).getArray()
                    res = crosstalk_fit(aggressor_imarr, victim_imarr, mask, covariance=covariance,
                                        correct_covariance=correct_covariance, seed=seed)

                    ## Add result to database
                    result = Result(aggressor_id=sensor.segments[i].id, victim_id=sensor.segments[j].id,
                                    aggressor_signal=signal, coefficient=res[0], error=res[4], 
                                    methodology='MODEL_LSQ', teststand=teststand, image_type='satellite',
                                    analysis='CrosstalkSatelliteTask', is_coadd=is_coadd, z_offset=res[1],
                                    y_tilt=res[2], x_tilt=res[3])
                    result.add_to_db(session)

"""MixCOATL Tasks for crosstalk analysis.

TODO:
    * Update CrosstalkSpotTask to interface with database and matrix.
    * Update CrosstalkMatrix as needed.
    * Add docstrings and confirm compliance with LSP coding style guide.
"""
import numpy as np
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter
from sqlalchemy.orm.exc import NoResultFound
import logging
from datetime import datetime
import os

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.eotest.image_utils as imutils
from lsst.eotest.sensor.MaskedCCD import MaskedCCD
from lsst.eotest.sensor.BrightPixels import BrightPixels

from mixcoatl.crosstalk import CrosstalkMatrix, make_stamp, crosstalk_fit
from mixcoatl.utils import AMP2SEG
from mixcoatl.database import Sensor, Segment, Result, db_session

class CrosstalkSpotConfig(pexConfig.Config):
    
    database = pexConfig.Field("SQL database DB file", str, default='test.db')
    length = pexConfig.Field("Length of postage stamps in y and x direction", int, default=200)
    nsig = pexConfig.Field("Outlier rejection sigma threshold", float, default=5.0)
    num_iter = pexConfig.Field("Number of least square iterations", int, default=3)
    threshold = pexConfig.Field("Aggressor spot mean signal threshold", float, default=40000.)
    verbose = pexConfig.Field("Turn verbosity on", bool, default=True)

class CrosstalkSpotTask(pipeBase.Task):

    ConfigClass = CrosstalkSpotConfig
    _DefaultName = "CrosstalkSpotTask"

    def run(self, sensor_name, infiles, gains, bias_frame=None, dark_frame=None, 
            crosstalk_matrix_file=None):

        all_amps = imutils.allAmps(infiles[0])
        with fits.open(infiles[0]) as hdulist:
            manufacturer = hdulist[0].header['CCD_MANU']
            lsst_num = hdulist[0].header['LSST_NUM']

        if crosstalk_matrix_file is not None:
            crosstalk_matrix = CrosstalkMatrix.from_fits(crosstalk_matrix_file)
            outfile = crosstalk_matrix_file
        else:
            crosstalk_matrix = CrosstalkMatrix(sensor_id1,
                                               namps=len(all_amps))
            outfile = 'test.fits'

        database = self.config.database
        with db_session(database) as session:

            ## Get (Add) sensor from (to) database
            try:
                sensor = Sensor.from_db(session, lsst_num=lsst_num)
            except NoResultFound:
                sensor = Sensor(sensor_name=sensor_name, lsst_num=lsst_num, manufacturer=manufacturer, 
                                namps=len(all_amps))
                sensor.segments = {i : Segment(segment_name=AMP2SEG[i], amplifier_number=i) for i in all_amps}
                sensor.add_to_db(session)
                session.commit()

            ## Get configuration and analysis settings
            length = self.config.length
            num_iter = self.config.num_iter
            nsig = self.config.nsig
            threshold = self.config.threshold

            ## Process image files
            for infile in infiles:

                ccd = MaskedCCD(infile, bias_frame=bias_frame, dark_frame=dark_frame)

                ## Determine aggressor amplifier and find spot
                for i in all_amps:
                    aggressor_imarr = ccd1.unbiased_and_trimmed_image(i).getImage().getArray()
                    smoothed = gaussian_filter(imarr1, 20)
                    y, x = np.unravel_index(smoothed.argmax(), smoothed.shape)
                    aggressor_stamp = make_stamp(aggressor_imarr, y, x, ly=length, lx=length)
                    ly, lx = aggressor_stamp.shape
                    Y, X = np.ogrid[-ly/2:ly/2, -lx/2:lx/2]
                    mask = X*X + Y*Y <= 20*20
                    signal = np.mean(stamp1[mask])

                    if signal < self.config.threshold:
                        continue
                    signals.append(signal)
                    row = {}

                    ## Calculate crosstalk for each victim amp
                    for j in all_amps:
                        victim_imarr = ccd2.unbiased_and_trimmed_image(j).getImage().getArray()

                        victim_stamp = make_stamp(victim_imarr, y, x)
                        res = crosstalk_fit(aggressor_stamp, victim_stamp, noise=7.0, num_iter=num_iter,
                                            nsig=nsig)
                        row[j] = res

                        ## Add result to database
                        result = Result(aggressor_id=sensor.segments[i].id, aggressor_signal=signal,
                                        coefficient=res[0], error=res[4], method='MODEL_LSQ',
                                        victim_id=sensor.segments[j].id)
                        result.add_to_db(session)

                    crosstalk_matrix.set_row(i, row)

        crosstalk_matrix.signal = np.median(np.asarray(signals))
        if sensor_id1 == sensor_id2:
            crosstalk_matrix.set_diagonal(0.)
        crosstalk_matrix.write_fits(outfile, overwrite=True)

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

class CrosstalkColumnConfig(pexConfig.Config):

    database = pexConfig.Field("SQL database DB file", str, default='crosstalk.db')
    nsig = pexConfig.Field("Outlier rejection sigma threshold", float, default=7.0)
    num_iter = pexConfig.Field("Number of least square iterations", int, default=1)
    length_y = pexConfig.Field("Length of postage stamps in y-direction", int, default=200)
    length_x = pexConfig.Field("Length of postage stamps in x-direction", int, default=20)
    verbose = pexConfig.Field("Turn verbosity on", bool, default=True)
    threshold = pexConfig.Field("Aggressor column mean signal threshold", float, default=1000.)

class CrosstalkColumnTask(pipeBase.Task):

    ConfigClass = CrosstalkColumnConfig
    _DefaultName = "CrosstalkColumnTask"

    def run(self, sensor_name, infiles, bias_frame=None, dark_frame=None, linearity_correction=None,
            noise=7.0):

        if not isinstance(infiles, list):
            infiles = [infiles]

        ## Get sensor information from header
        all_amps = imutils.allAmps(infiles[0])
        with fits.open(infiles[0]) as hdulist:
            lsst_num = hdulist[0].header['LSST_NUM']
            teststand = hdulist[0].header['TSTAND']
            manufacturer = lsst_num[:3]

        ## Interface with SQL database
        database = self.config.database
        logging.info("{0}  Running CrosstalkColumnTask using database {1}".format(datetime.now(), database))

        with db_session(database) as session:

            ## Get sensor from database
            try:
                sensor = Sensor.from_db(session, lsst_num=lsst_num)
            except NoResultFound:
                sensor = Sensor(sensor_name=sensor_name, lsst_num=lsst_num, manufacturer=manufacturer,
                                namps=len(all_amps))
                sensor.segments = {i : Segment(segment_name=AMP2SEG[i], amplifier_number=i) for i in all_amps}
                sensor.add_to_db(session)
                session.commit()
                logging.info("{0}  New sensor {1} added to database".format(datetime.now(), lsst_num))

            ## Get configuration and analysis settings
            ly = self.config.length_y
            lx = self.config.length_x
            num_iter = self.config.num_iter
            nsig = self.config.nsig
            threshold = self.config.threshold

            ccds = [MaskedCCD(infile, bias_frame=bias_frame, dark_frame=dark_frame,
                              linearity_correction=linearity_correction) for infile in infiles]

            logging.info("{0}  ".format(datetime.now()) + \
                         "Processing files: {}".format(' '.join(map(str, infiles))))
            for i in all_amps:

                ## Search for aggressor column
                exptime = 1
                gain = 1
                bp = BrightPixels(ccds[0], i, exptime, gain, ethresh=threshold)
                pixels, columns = bp.find()
                if len(columns) == 0:
                    continue
                col = columns[0]

                aggressor_images = [ccd.unbiased_and_trimmed_image(i).getImage() for ccd in ccds]
                aggressor_imarr = imutils.stack(aggressor_images).getArray()
                signal = np.mean(aggressor_imarr[:, col])
                aggressor_stamp = make_stamp(aggressor_imarr, 1000, col, ly=ly, lx=lx)
                
                ## Calculate crosstalk for each victim amp
                for j in all_amps:
                    victim_images = [ccd.unbiased_and_trimmed_image(j).getImage() for ccd in ccds]
                    victim_imarr = imutils.stack(victim_images).getArray()
                    victim_stamp = make_stamp(victim_imarr, 1000, col, ly=ly, lx=lx)
                    res = crosstalk_fit(aggressor_stamp, victim_stamp, noise=noise, num_iter=num_iter, 
                                        nsig=nsig)

                    ## Add result to database
                    result = Result(aggressor_id=sensor.segments[i].id, aggressor_signal=signal,
                                    coefficient=res[0], error=res[4], method='MODEL_LSQ',
                                    victim_id=sensor.segments[j].id, filename=os.path.basename(infiles[0]),
                                    teststand=teststand)
                    result.add_to_db(session)
                    logging.info("{0}  Injested C({1},{2}) for signal {3:.1f}".format(datetime.now(), i, j,
                                                                                      signal))
                logging.info("{0}  Task completed successfully.".format(datetime.now()))

class CrosstalkCoordConfig(pexConfig.Config):

    database = pexConfig.Field("SQL database DB file", str, default='crosstalk.db')
    nsig = pexConfig.Field("Outlier rejection sigma threshold", float, default=7.0)
    num_iter = pexConfig.Field("Number of least square iterations", int, default=1)
    length_y = pexConfig.Field("Length of postage stamps in y-direction", int, default=200)
    length_x = pexConfig.Field("Length of postage stamps in x-direction", int, default=100)
    verbose = pexConfig.Field("Turn verbosity on", bool, default=True)
    
class CrosstalkCoordTask(pipeBase.Task):

    ConfigClass = CrosstalkCoordConfig
    _DefaultName = "CrosstalkCoordTask"

    def run(self, sensor_name, infiles, aggressor_coordinates, bias_frame=None,
            dark_frame=None, linearity_correction=None, noise=7.0):

        if not isinstance(infiles, list):
            infiles = [infiles]
        if not isinstance(aggressor_coordinates, list):
            aggressor_coordinates = [aggressor_coordinates]

        ## Get sensor information from header
        all_amps = imutils.allAmps(infiles[0])
        with fits.open(infiles[0]) as hdulist:
            lsst_num = hdulist[0].header['LSST_NUM']
            teststand = hdulist[0].header['TSTAND']
            manufacturer = lsst_num[:3]

        ## Interface with SQL database
        database = self.config.database
        logging.info("{0}  Running CrosstalkCoordTask using database {1}".format(datetime.now(), database))

        with db_session(database) as session:

            ## Get sensor from database
            try:
                sensor = Sensor.from_db(session, lsst_num=lsst_num)
            except NoResultFound:
                sensor = Sensor(sensor_name=sensor_name, lsst_num=lsst_num, manufacturer=manufacturer,
                                namps=len(all_amps))
                sensor.segments = {i : Segment(segment_name=AMP2SEG[i], amplifier_number=i) for i in all_amps}
                sensor.add_to_db(session)
                session.commit()
                logging.info("{0}  New sensor {1} added to database".format(datetime.now(), lsst_num))

            ## Get configuration and analysis settings
            ly = self.config.length_y
            lx = self.config.length_x
            num_iter = self.config.num_iter
            nsig = self.config.nsig

            ccds = [MaskedCCD(infile, bias_frame=bias_frame, dark_frame=dark_frame,
                              linearity_correction=linearity_correction) for infile in infiles]

            logging.info("{0}  ".format(datetime.now()) + \
                         "Processing files: {}".format(' '.join(map(str, infiles))))

            for coord in aggressor_coordinates:

                i, y, x = coord
                aggressor_images = [ccd.unbiased_and_trimmed_image(i).getImage() for ccd in ccds]
                aggressor_imarr = imutils.stack(aggressor_images).getArray()
                aggressor_stamp = make_stamp(aggressor_imarr, y, x, ly=ly, lx=lx)
                signal = np.max(aggressor_stamp)
                
                ## Calculate crosstalk for uncontaminated victim amp
                for j in all_amps:
                    victim_images = [ccd.unbiased_and_trimmed_image(j).getImage() for ccd in ccds]
                    victim_imarr = imutils.stack(victim_images).getArray()
                    victim_stamp = make_stamp(victim_imarr, y, x, ly=ly, lx=lx)
                    if (np.std(victim_stamp) > 100.) and j != i:
                        continue
                    res = crosstalk_fit(aggressor_stamp, victim_stamp, noise=noise, num_iter=num_iter, 
                                        nsig=nsig)

                    ## Add result to database
                    result = Result(aggressor_id=sensor.segments[i].id, aggressor_signal=signal,
                                    coefficient=res[0], error=res[4], method='MODEL_LSQ',
                                    victim_id=sensor.segments[j].id, filename=os.path.basename(infiles[0]),
                                    teststand=teststand)
                    result.add_to_db(session)
                    logging.info("{0}  Injested C({1},{2}) for signal {3:.1f}".format(datetime.now(), i, j,
                                                                                      signal))
                logging.info("{0}  Task completed successfully.".format(datetime.now()))



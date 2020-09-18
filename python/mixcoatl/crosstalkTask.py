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

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.eotest.image_utils as imutils
from lsst.eotest.sensor.MaskedCCD import MaskedCCD
from lsst.eotest.sensor.BrightPixels import BrightPixels

from mixcoatl.crosstalk import CrosstalkMatrix, make_stamp, crosstalk_fit
from mixcoatl.utils import AMP2SEG
from mixcoatl.database import Sensor, Segment, Result, db_session

class CrosstalkSpotConfig(pexConfig.Config):
    
    nsig = pexConfig.Field("Outlier rejection sigma threshold", float, 
                           default=5.0)
    num_iter = pexConfig.Field("Number of least square iterations", int, 
                               default=3)
    threshold = pexConfig.Field("Aggressor spot mean signal threshold", float,
                                default=40000.)
    outfile = pexConfig.Field("Output filename", str, 
                              default='crosstalk_matrix.fits')
    verbose = pexConfig.Field("Turn verbosity on", bool, default=True)
    aggressors_per_image = pexConfig.Field("Number of aggressors per image",
                                           int, default=4)

class CrosstalkColumnConfig(pexConfig.Config):

    database = pexConfig.Field("SQL database DB file", str, default='test.db')
    nsig = pexConfig.Field("Outlier rejection sigma threshold", float, default=7.0)
    num_iter = pexConfig.Field("Number of least square iterations", int, default=1)
    length_y = pexConfig.Field("Length of postage stamps in y-direction", int, default=200)
    length_x = pexConfig.Field("Length of postage stamps in x-direction", int, default=50)
    verbose = pexConfig.Field("Turn verbosity on", bool, default=True)
    threshold = pexConfig.Field("Aggressor column mean signal threshold", float, default=1000.)

class CrosstalkCoordsConfig(pexConfig.Config):

    nsig = pexConfig.Field("Outlier rejection sigma threshold", float, 
                           default=5.0)
    num_iter = pexConfig.Field("Number of least square iterations", int, 
                               default=3)
    outfile = pexConfig.Field("Output filename", str, 
                              default='crosstalk_matrix.fits')
    verbose = pexConfig.Field("Turn verbosity on", bool, default=True)

class CrosstalkSpotTask(pipeBase.Task):

    ConfigClass = CrosstalkSpotConfig
    _DefaultName = "CrosstalkSpotTask"

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

class CrosstalkColumnTask(pipeBase.Task):

    ConfigClass = CrosstalkColumnConfig
    _DefaultName = "CrosstalkColumnTask"

    def run(self, sensor_name, infiles, bias_frame=None, dark_frame=None):

        ## Get sensor information from header
        all_amps = imutils.allAmps(infiles[0])
        with fits.open(infiles[0]) as hdulist:
            manufacturer = hdulist[0].header['CCD_MANU']
            lsst_num = hdulist[0].header['LSST_NUM']

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
            i, col = aggressor_info
            ly = self.config.length_y
            lx = self.config.length_x
            num_iter = self.config.num_iter
            nsig = self.config.nsig
            threshold = self.config.threshold

            ## Process image files
            for infile in infiles:

                ccd = MaskedCCD(infile, bias_frame=bias_frame, dark_frame=dark_frame)

                ## Determine aggressor amplifier and column
                for i in all_amps:
                    exptime = 1
                    gain = 1
                    bp = BrightPixels(ccd, i, exptime, gain, ethresh=threshold)
                    pixels, columns = bp.find()

                    if len(columns) == 0:
                        continue
                    col = columns[0]
                    aggressor_imarr = ccd.unbiased_and_trimmed_image(i).getImage().getArray()
                    signal = np.mean(aggressor_imarr[:, col])    
                    aggressor_stamp = make_stamp(aggressor_imarr, 2000, col, ly=ly, lx=lx)

                    ## Calculate crosstalk coefficient
                    for j in all_amps:

                        victim_imarr = ccd.unbiased_and_trimmed_image(j).getImage().getArray()
                        victim_stamp = make_stamp(victim_imarr, 2000, col, ly=ly, lx=lx)
                        res = crosstalk_fit(aggressor_stamp, victim_stamp, noise=7.0, num_iter=num_iter, 
                                        nsig=nsig)

                        ## Add result to database
                        result = Result(aggressor_id=sensor.segments[i].id, aggressor_signal=signal,
                                        coefficient=res[0], error=res[4], method='MODEL_LSQ',
                                        victim_id=sensor.segments[j].id)
                        result.add_to_db(session)

class CrosstalkCoordsTask(pipeBase.Task):

    ConfigClass = CrosstalkCoordsConfig
    _DefaultName = "CrosstalkCoordsTask"

    def run(self, sensor_id1, infile1, signal, aggressor_coords, gains1, 
            bias_frame1=None, dark_frame1=None, crosstalk_matrix_file=None, **kwargs):

        ## Parse kwargs for separate victim CCD inputs
        try:
            sensor_id2 = kwargs['sensor_id2']
        except KeyError:
            sensor_id2 = sensor_id1
            infile2 = infile1
            gains2 = gains1
            bias_frame2 = bias_frame1
            dark_frame2 = dark_frame1
        else:
            infile2 = kwargs['infile2']
            gains2 = kwargs['gains2']
            bias_frame2 = kwargs['bias_frame2']
            dark_frame2 = kwargs['dark_frame2']
        
        all_amps = imutils.allAmps(infile1)

        ## Create new matrix or modify existing
        if crosstalk_matrix_file is not None:
            crosstalk_matrix = CrosstalkMatrix.from_fits(crosstalk_matrix_file)
            outfile = crosstalk_matrix_file
        else:
            crosstalk_matrix = CrosstalkMatrix(sensor_id1, signal=signal, 
                                               victim_id=sensor_id2,
                                               namps=len(all_amps))
            outfile = self.config.outfile

        ccd1 = MaskedCCD(infile1, bias_frame=bias_frame1, 
                         dark_frame=dark_frame1)
        ccd2 = MaskedCCD(infile2, bias_frame=bias_frame2, 
                         dark_frame=dark_frame2)      

        ## Search each amp for aggressor
        for aggressor_coord in aggressor_coords:
            i, y, x = aggressor_coord
            imarr1 = ccd1.unbiased_and_trimmed_image(i).getImage().getArray()*gains1[i]
            stamp1 = make_stamp(imarr1, y, x)
            row = {}

            ## Calculate crosstalk for each victim amp
            for j in all_amps:
                imarr2 = ccd2.unbiased_and_trimmed_image(j).getImage().getArray()*gains2[j]

                stamp2 = make_stamp(imarr2, y, x)
                row[j] = crosstalk_fit(stamp1, stamp2, noise=7.0,
                                             num_iter=self.config.num_iter,
                                             nsig=self.config.nsig)

            crosstalk_matrix.set_row(i, row)

        if sensor_id1 == sensor_id2:
            crosstalk_matrix.set_diagonal(0.)
        crosstalk_matrix.write_fits(outfile, overwrite=True)
   

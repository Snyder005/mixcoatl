import numpy as np
from scipy.ndimage.filters import gaussian_filter
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

import lsst.eotest.image_utils as imutils
from lsst.eotest.sensor.MaskedCCD import MaskedCCD

from mixcoatl.crosstalk import CrosstalkMatrix, make_stamp, crosstalk_fit

class CrosstalkConfig(pexConfig.Config):
    
    nsig = pexConfig.Field("Outlier rejection sigma threshold", float, 
                           default=5.0)
    num_iter = pexConfig.Field("Number of least square iterations", int, 
                               default=3)
    threshold = pexConfig.Field("Aggressor spot mean signal threshold", float,
                                default=40000.)
    output_file = pexConfig.Field("Output filename", str, 
                                  default='crosstalk_matrix.fits')
    verbose = pexConfig.Field("Turn verbosity on", bool, default=True)

class CrosstalkTask(pipeBase.Task):

    ConfigClass = CrosstalkConfig
    _DefaultName = "CrosstalkTask"

    def run(self, sensor_id1, infile1, gains1, bias_frame1=None, dark_frame1=None,
            crosstalk_matrix_file=None, **kwargs):

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

        ccd1 = MaskedCCD(infile1, bias_frame=bias_frame1, dark_frame=dark_frame1)
        ccd2 = MaskedCCD(infile2, bias_frame=bias_frame2, dark_frame=dark_frame2)
        
        all_amps = imutils.allAmps(infile1)

        ## Create new matrix or modify existing
        if crosstalk_matrix_file is not None:
            crosstalk_matrix = CrosstalkMatrix(sensor_id1, victim_id=sensor_id2,
                                               filename=crosstalk_matrix_file,
                                               namps=len(all_amps))
            outfile = crosstalk_matrix_file
        else:
            outfile = self.config.outfile
            crosstalk_matrix = CrosstalkMatrix(sensor_id1, victim_id=sensor_id2,
                                               namps=len(all_amps))

        num_aggressors = 0

        ## Search each amp for aggressor
        for i in all_amps:
            imarr1 = ccd1.unbiased_and_trimmed_image(i).getImage().getArray()*gains1[i]
            smoothed = gaussian_filter(imarr1, 20)
            y, x = np.unravel_index(smoothed.argmax(), smoothed.shape)
            stamp1 = make_stamp(imarr1, y, x)
            ly, lx = stamp1.shape
            Y, X = np.ogrid[-ly/2:ly/2, -lx/2:lx/2]
            mask = X*X + Y*Y <= 20*20

            if np.mean(stamp1[mask]) > self.config.threshold:

                row = {}

                ## Calculate crosstalk for each victim amp
                for j in all_amps:
                    imarr2 = ccd2.unbiased_and_trimmed_image(j).getImage().getArray()*gains2[j]

                    stamp2 = make_stamp(imarr2, y, x)
                    row[j] = crosstalk_fit(stamp1, stamp2, noise=7.0,
                                                 num_iter=self.config.num_iter,
                                                 nsig=self.config.nsig)

                crosstalk_matrix.set_row(i, row)
                if num_aggressors == 4: break

        crosstalk_matrix.write_fits(outfile)
   

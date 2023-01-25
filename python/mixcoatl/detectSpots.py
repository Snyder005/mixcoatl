import numpy as np
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks

from lsst.utils.timer import timeMethod
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.maskStreaks import LineCollection

class DetectSpotsConfig(pexConfig.Config):

    threshold = Field(
        dtype=float,
        default=30000.,
        doc="Minimum mean value of crosstalk source."
    )
    maskLength = Field(
        dtype=float,
        default=250.,
        doc="Length of side of square mask."
    )
    doAnnularCutout = Field(
        dtype=bool,
        default=False,
        doc="Mask an annular cutout of the square mask."
    )
    annulusInnerRadius = Field(
        dtype=float,
        default=40.,
        doc="Inner radius of annulur mask used for cutout."
    )
    annulusOuterRadius = Field(
        dtype=float,
        default=100.,
        doc="Outer radius of annulur mask used for cutout."
    )
    
class DetectSpotsTask(pipeBase.Task):
    
    ConfigClass = DetectSpotsConfig
    _DefaultName = "detectLines"

    @timeMethod
    def run(self, maskedImage):

        imarr = maskedImage.getImage().getArray()
        smoothed = gaussian_filter(imarr, 20)
        y, x = np.unravel_index(smoothed.argmax(), smoothed.shape)
        
        signal = sigma_clipped_stats(imarr,
                                     mask=~mixCrosstalk.circular_mask(imarr, y, x, radius=10),
                                     cenfunc='median', stdfunc=median_absolute_deviation)[1]

        if signal < self.config.threshold:
            raise RuntimeError("No crosstalk source detected.")

        sourceMask = mixCrosstalk.rectangular_mask(imarr, y, x, ly=self.config.maskLength, 
                                                   lx=self.config.MaskLength)
        
        if self.config.doAnnularCutout:
            cutout = ~mixCrosstalk.annularMask(imarr, y, x,
                                               inner_radius=self.config.annulusInnerRadius,
                                               outer_radius=self.config.annulusOuterRadius)
            sourceMask = sourceMask*cutout

        return pipeBase.Struct(
            signal=signal,
            sourceMask=sourceMask)

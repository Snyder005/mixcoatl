import numpy as np
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks

from lsst.utils.timer import timeMethod
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.maskStreaks import LineCollection

class DetectStreaksConfig(pexConfig.Config):

    maskWidth = pexConfig.Field(
        dtype=float,
        default=80.,
        doc="Width of streak mask."
    )
    sigma = pexConfig.Field(
        dtype=float,
        default=15.,
        doc="Standard deviation of the Gaussian filter."
    )
    lowThreshold = pexConfig.Field(
        dtype=float,
        default=5.,
        doc="Lower bound for hysteresis thresholding (linking edges)."
    )
    highThreshold = pexConfig.Field(
        dtype=float,
        default=15.,
        doc="Upper bound for hysteresis thresholding (linking edges)."
    )
    
class DetectStreaksTask(pipeBase.Task):
    
    ConfigClass = DetectStreaksConfig
    _DefaultName = "detectStreaks"

    @timeMethod
    def run(self, maskedImage):

        imarr = maskedImage.getImage().getArray()
        tested_angles = np.linspace(-np.pi/2., np.pi/2., 1000)
        edges = feature.canny(imarr, sigma=self.config.sigma,
                              low_threshold=self.config.lowThreshold,
                              high_threshold=self.config.highThreshold)
        h, theta, d = hough_line(edges, theta=tested_angles)
        accum, angles, dists = hough_line_peaks(h, theta, d)

        if len(angles) != 2:
            raise RuntimeError("No crosstalk source detected.")

        dist = np.mean(dists)
        angle = np.mean(angles)
        theta = np.rad2deg(angle)
        rho = dist + x0*np.cos(angle) + y0*np.sin(angle)
        
        line = Line(rho, theta)
        sourceMask = mixCrosstalk.streak_mask(imarr, line, self.config.maskWidth)
        signal = sigma_clipped_stats(imarr,
                                     mask=~mixCrosstalk.streak_mask(imarr, line, 1.0),
                                     cenfunc='median', stdfunc=median_absolute_deviation)[1]

        return pipeBase.Struct(
            signal=signal,
            sourceMask=sourceMask
        )

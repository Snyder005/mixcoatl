import numpy as np
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks

from lsst.utils.timer import timeMethod
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.maskStreaks import LineCollection

class DetectLinesConfig(pexConfig.Config):

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
    
class DetectLinesTask(pipeBase.Task):
    
    ConfigClass = DetectLinesConfig
    _DefaultName = "detectLines"

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
            lines = LineCollection([], [])
        else:
            dist = np.mean(dists)
            angle = np.mean(angles)

            Ny, Nx = imarr.shape
            x0 = (Nx - 1)/2.
            y0 = (Ny - 1)/2.
            theta = np.rad2deg(angle)
            rho = dist + x0*np.cos(angle) + y0*np.sin(angle)

            lines = LineCollection([rho], [theta])

        return pipeBase.Struct(lines=lines)

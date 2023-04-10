import numpy as np
from skimage import feature
from sklearn.cluster import KMeans
from skimage.transform import hough_line, hough_line_peaks
from astropy.stats import median_absolute_deviation, sigma_clipped_stats

from lsst.utils.timer import timeMethod
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.maskStreaks import LineCollection
import lsst.kht

import mixcoatl.crosstalk as mixCrosstalk

class DetectStreaksConfig(pexConfig.Config):

    maskWidth = pexConfig.Field(
        dtype=float,
        default=80.,
        doc="Width of streak mask."
    )
    minimumKernelHeight = pexConfig.Field(
        doc="Minimum height of the streak-finding kernel relative to the tallest kernel",
        dtype=float,
        default=0.0,
    )
    absMinimumKernelHeight = pexConfig.Field(
        doc="Minimum absolute height of the streak-finding kernel",
        dtype=float,
        default=5,
    )
    clusterMinimumSize = pexConfig.Field(
        doc="Minimum size in pixels of detected clusters",
        dtype=int,
        default=50,
    )
    clusterMinimumDeviation = pexConfig.Field(
        doc="Allowed deviation (in pixels) from a straight line for a detected "
            "line",
        dtype=int,
        default=2,
    )
    delta = pexConfig.Field(
        doc="Stepsize in angle-radius parameter space",
        dtype=float,
        default=0.2,
    )
    nSigma = pexConfig.Field(
        doc="Number of sigmas from center of kernel to include in voting "
            "procedure",
        dtype=float,
        default=2,
    )
    rhoBinSize = pexConfig.Field(
        doc="Binsize in pixels for position parameter rho when finding "
            "clusters of detected lines",
        dtype=float,
        default=60,
    )
    thetaBinSize = pexConfig.Field(
        doc="Binsize in degrees for angle parameter theta when finding "
            "clusters of detected lines",
        dtype=float,
        default=2,
    )
    
class DetectStreaksTask(pipeBase.Task):
    
    ConfigClass = DetectStreaksConfig
    _DefaultName = "detectStreaks"

    @timeMethod
    def run(self, maskedImage):
        
        mask = maskedImage.getMask()
        imarr = maskedImage.getImage().getArray()
        detectionMask = (mask.array & mask.getPlaneBitMask('DETECTED'))
        filterData = detectionMask.astype(int) 

        edges = feature.canny(filterData, sigma=1.0, low_threshold=0, high_threshold=1)
        tested_angles = np.linspace(-np.pi/2., np.pi/2., 1000)
        h, theta, d = hough_line(edges, theta=tested_angles)
        accum, angles, dists = hough_line_peaks(h, theta, d)
        
        rhos = []
        thetas = []
        for i in range(len(angles)):
            angle = angles[i]
            dist = dists[i]
            Ny, Nx = imarr.shape
            x0 = (Nx-1)/2.
            y0 = (Ny-1)/2.
            thetas.append(np.rad2deg(angle))
            rhos.append(dist - x0*np.cos(angle) - y0*np.sin(angle))
            
        lines = LineCollection(rhos, thetas)

        # fix this
        if len(lines) < 2:
            raise RuntimeError("No crosstalk source detected.")
        
        result = self.findClusters(lines)

        if len(result) > 1:
            raise RuntimeError("Could not cluster lines to identify streak.")
        else:
            line = result[0]

        sourceMask = mixCrosstalk.streak_mask(imarr, line, self.config.maskWidth)
        signal = sigma_clipped_stats(imarr, mask=~mixCrosstalk.streak_mask(imarr, line, 1.0),
                                     cenfunc='median', stdfunc=median_absolute_deviation)[1]

        return pipeBase.Struct(
            signal=signal,
            sourceMask=sourceMask
        )

    def findClusters(self, lines):

        x = lines.rhos / self.config.rhoBinSize
        y = lines.thetas / self.config.thetaBinSize
        X = np.array([x, y]).T
        nClusters = 1

        while True:
            kmeans = KMeans(n_clusters=nClusters).fit(X)
            clusterStandardDeviations = np.zeros((nClusters, 2))
            for c in range(nClusters):
                inCluster = X[kmeans.labels_ == c]
                clusterStandardDeviations[c] = np.std(inCluster, axis=0)
            # Are the rhos and thetas in each cluster all below the threshold?
            if (clusterStandardDeviations <= 1.0).all():
                break
            nClusters += 1

        finalClusters = kmeans.cluster_centers_.T

        finalRhos = finalClusters[0] * self.config.rhoBinSize
        finalThetas = finalClusters[1] * self.config.thetaBinSize
        result = LineCollection(finalRhos, finalThetas)

        return result

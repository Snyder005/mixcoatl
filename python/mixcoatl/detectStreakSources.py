# This file is part of mixcoatl.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from skimage.feature import canny
from sklearn.cluster import KMeans
from skimage.transform import hough_line, hough_line_peaks
from astropy.stats import median_absolute_deviation, sigma_clipped_stats

from lsst.utils.timer import timeMethod
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms.maskStreaks import LineCollection
import lsst.kht

import mixcoatl.crosstalk as mixCrosstalk

class DetectStreakSourcesConfig(pexConfig.Config):

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
    
class DetectStreakSourcesTask(pipeBase.Task):
    
    ConfigClass = DetectStreakSourcesConfig
    _DefaultName = "detectStreakSources"

    @timeMethod
    def run(self, maskedImage):

        imarr = maskedImage.getImage().getArray()
        tested_angles = np.linspace(-np.pi/2., np.pi/2., 1000)
        edges = canny(imarr, sigma=self.config.sigma,
                              low_threshold=self.config.lowThreshold,
                              high_threshold=self.config.highThreshold)
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

        if len(lines) < 2:
            raise RuntimeError("No crosstalk source detected.")
        
        streaks = self.findClusters(lines)
        streaks = sorted(streaks, key=lambda x : sigma_clipped_stats(imarr, mask=~mixCrosstalk.make_streak_mask(imarr, x, 1.0), 
                                                                     cenfunc='median', stdfunc=median_absolute_deviation)[1])

        signals = [sigma_clipped_stats(imarr, mask=~mixCrosstalk.make_streak_mask(imarr, streaks[-1], 1.0),
                                       cenfunc='median', stdfunc=median_absolute_deviation)[1] for s in streaks]
        sourceMask = mixCrosstalk.make_streak_mask(imarr, streaks[-1], self.config.maskWidth)

        return pipeBase.Struct(
            signals=signals,
            sourceMask=sourceMask,
            streaks=streaks,
            edges=edges,
            hough=h
        )

    def findClusters(self, lines):

        x = lines.rhos / self.config.rhoBinSize
        y = lines.thetas / self.config.thetaBinSize
        X = np.array([x, y]).T
        nClusters = 1

        while True:
            kmeans = KMeans(n_clusters=nClusters, n_init='auto').fit(X)
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

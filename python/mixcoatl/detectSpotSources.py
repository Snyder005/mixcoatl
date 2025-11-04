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
from scipy.ndimage.filters import gaussian_filter
from astropy.stats import median_absolute_deviation, sigma_clipped_stats

from lsst.utils.timer import timeMethod
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

import mixcoatl.crosstalk as mixCrosstalk

class DetectSpotSourcesConfig(pexConfig.Config):

    threshold = pexConfig.Field(
        dtype=float,
        default=30000.,
        doc="Minimum mean value of crosstalk source."
    )
    maskLength = pexConfig.Field(
        dtype=float,
        default=250.,
        doc="Length of side of square mask."
    )
    doAnnularCutout = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Mask an annular cutout of the square mask."
    )
    annulusInnerRadius = pexConfig.Field(
        dtype=float,
        default=40.,
        doc="Inner radius of annulur mask used for cutout."
    )
    annulusOuterRadius = pexConfig.Field(
        dtype=float,
        default=100.,
        doc="Outer radius of annulur mask used for cutout."
    )
    
class DetectSpotSourcesTask(pipeBase.Task):
    
    ConfigClass = DetectSpotSourcesConfig
    _DefaultName = "detectLines"

    @timeMethod
    def run(self, maskedImage):

        imarr = maskedImage.getImage().getArray()
        smoothed = gaussian_filter(imarr, 20)
        y, x = np.unravel_index(smoothed.argmax(), smoothed.shape)
        
        signal = sigma_clipped_stats(imarr,
                                     mask=~mixCrosstalk.make_circular_mask(imarr, y, x, radius=10),
                                     cenfunc='median', stdfunc=median_absolute_deviation)[1]

        if signal < self.config.threshold:
            raise RuntimeError("No crosstalk source detected.")

        sourceMask = mixCrosstalk.make_rectangular_mask(imarr, y, x, ly=self.config.maskLength, 
                                                        lx=self.config.maskLength)
        
        if self.config.doAnnularCutout:
            cutout = ~mixCrosstalk.make_annular_mask(imarr, y, x,
                                                     inner_radius=self.config.annulusInnerRadius,
                                                     outer_radius=self.config.annulusOuterRadius)
            sourceMask = sourceMask*cutout

        return pipeBase.Struct(
            signal=signal,
            sourceMask=sourceMask)

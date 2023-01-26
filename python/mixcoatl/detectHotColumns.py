import numpy as np
from astropy.stats import median_absolute_deviation, sigma_clipped_stats

from lsst.utils.timer import timeMethod
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.detection import FootprintSet, Threshold

import mixcoatl.crosstalk as mixCrosstalk

class DetectHotColumnsConfig(pexConfig.Config):

    threshold = pexConfig.Field(
        dtype=float,
        default=30000.,
        doc="Minimum level of source pixels."
    )
    maskWidth = pexConfig.Field(
        dtype=float,
        default=10.,
        doc="Width of rectangular mask in serial direction."
    )
    
class DetectHotColumnsTask(pipeBase.Task):
    
    ConfigClass = DetectHotColumnsConfig
    _DefaultName = "detectHotColumns"

    @timeMethod
    def run(self, maskedImage):

        image = maskedImage.getImage()
        imarr = image.getArray()
        Ny, Nx = imarr.shape
        fp_set = FootprintSet(image, Threshold(threshold))    
        columns = dict([(x, []) for x in range(0, image.getWidth())])
        for footprint in fp_set.getFootprints():
            for span in footprint.getSpans():
                y = span.getY()
                for x in range(span.getX0(), span.getX1()+1):
                    columns[x].append(y)
                    
        bright_cols = []
        x0 = image.getX0()
        y0 = image.getY0()
        for x in columns:
            if self.bad_column(columns[x], 20):
                bright_cols.append(x - x0)
        #
        # Sort the output.
        #
        bright_cols.sort()

        if len(bright_cols) == 0:
            raise RuntimeError("No crosstalk source detected.")

        sourceMask = mixCrosstalk.rectangular_mask(sourceAmpArray, Ny//2, bright_cols[0],
                                                   ly=Ny, lx=self.config.maskWidth)
        signal = sigma_clipped_stats(imarr, cenfunc='median', stdfunc=median_absolute_deviation)

        return pipeBase.Struct(
            signal=signal,
            sourceMask=sourceMask
        )

    def bad_column(self, column_indices, threshold):
        """Identify bad columns by number of masked pixels.
        
        Parameters
        ----------
        column_indices : `list`
            List of column indices.
        threshold : `int`
            Number of bad pixels required to mark the column as bad.
        Returns
        -------
        is_bad_column : `bool`
            `True` if column is bad, `False` if not.
        """
        if len(column_indices) < threshold:
            # There are not enough masked pixels to mark this as a bad
            # column.
            return False
        # Fill an array with zeros, then fill with ones at mask locations.
        column = np.zeros(max(column_indices) + 1)
        column[(column_indices,)] = 1
        # Count pixels in contiguous masked sequences.
        masked_pixel_count = []
        last = 0
        for value in column:
            if value != 0 and last == 0:
                masked_pixel_count.append(1)
            elif value != 0 and last != 0:
                masked_pixel_count[-1] += 1
            last = value
        if len(masked_pixel_count) > 0 and max(masked_pixel_count) >= threshold:
            return True
        return False

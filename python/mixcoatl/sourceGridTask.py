import os
import numpy as np
from os.path import join, splitext
import scipy
from scipy.spatial import distance
from astropy.io import fits

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

from .sourcegrid import BaseGrid, DistortedGrid, grid_fit, coordinate_distances

camera = camMapper._makeCamera()
lct = LsstCameraTransforms(camera)

class SourceGridConfig(pexConfig.Config):
    """Configuration for GridFitTask."""

    max_displacement = pexConfig.Field("Maximum distance (pixels) between matched sources.",
                                       float, default=10.)
    nrows = pexConfig.Field("Number of grid rows.", int, default=49)
    ncols = pexConfig.Field("Number of grid columns.", int, default=49)
    y_kwd = pexConfig.Field("Source catalog y-position keyword", str, 
                            default='base_SdssShape_y')
    x_kwd = pexConfig.Field("Source catalog y-position keyword", str, 
                            default='base_SdssShape_x')
    yy_kwd = pexConfig.Field("Source catalog y-position keyword", str, 
                             default='base_SdssShape_yy')
    xx_kwd = pexConfig.Field("Source catalog y-position keyword", str, 
                             default='base_SdssShape_xx')
    flux_kwd = pexConfig.Field("Source catalog y-position keyword", str, 
                               default='base_SdssShape_instFlux')
    outfile = pexConfig.Field("Output filename", str, default="test.fits")

class SourceGridTask(pipeBase.Task):

    ConfigClass = SourceGridConfig
    _DefaultName = "SourceGridTask"

    @pipeBase.timeMethod
    def run(self, infile):

        basename = os.path.basename(infile)
        projector_y = float(basename.split('_')[-1][:-5]) # camera x/y coords
        projector_x = float(basename.split('_')[-2][:-1])

        ccd_name, ccd_x, ccd_y = lct.focalMmToCcdPixel(projector_y, projector_x)

        x0_guess = 2*509*4. - ccd_x
        y0_guess = ccd_y

        src = fits.getdata(infile)
        
        ## Get source information
        srcY = src[self.config.y_kwd]
        srcX = src[self.config.x_kwd]
        srcXX = src[self.config.xx_kwd]
        srcYY = src[self.config.yy_kwd]
        srcF = src[self.config.flux_kwd]

        grid = grid_fit(srcY, srcX, self.config.ncols, self.config.nrows,
                        y0_guess, x0_guess)
        gY, gX = grid.make_ideal_grid()

        indices, distances = coordinate_distances(gY, gX, srcY, srcX)
        nn_indices = indices[:, 0]

        ## Matched source information
        dy_array = srcY[nn_indices] - gY
        dx_array = srcX[nn_indices] - gX
        xx_array = np.zeros(gX.shape[0])
        yy_array = np.zeros(gY.shape[0])
        dxx_array = srcXX[nn_indices]
        dyy_array = srcYY[nn_indices]
        flux_array = np.zeros(gX.shape[0])
        dflux_array = srcF[nn_indices]

        ## Mask unmatched sources
        mask = np.hypot(dy_array, dx_array) >= self.config.max_displacement
        dx_array[mask] = np.nan
        dy_array[mask] = np.nan
        dxx_array[mask] = np.nan
        dyy_array[mask] = np.nan
        dflux_array[mask] = np.nan

        ## Construct source information dictionary
        data = {}
        data['X'] = gX
        data['Y'] = gY
        data['DX'] = dx_array
        data['DY'] = dy_array

        data['XX'] = xx_array
        data['YY'] = yy_array
        data['DXX'] = dxx_array
        data['DYY'] = dyy_array

        data['FLUX'] = flux_array
        data['DFLUX'] = dflux_array

        distorted_grid = DistortedGrid(grid.ystep, grid.xstep, grid.theta, 
                                       grid.y0, grid.x0, self.config.ncols, 
                                       self.config.nrows, data)
        distorted_grid.write_fits(self.config.outfile, overwrite=True)

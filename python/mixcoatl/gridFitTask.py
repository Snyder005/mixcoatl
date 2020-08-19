import os
import numpy as np
from os.path import join
from astropy.io import fits

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from .sourcegrid import DistortedGrid, grid_fit
from .utils import ITL_AMP_GEOM, E2V_AMP_GEOM

class GridFitConfig(pexConfig.Config):
    """Configuration for GridFitTask."""

    max_displacement = pexConfig.Field("Maximum distance [pix] between matched sources.",
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
    brute_search = pexConfig.Field("Perform prelim brute search", bool,
                                   default=False)
    vary_theta = pexConfig.Field("Vary theta parameter during fit", bool,
                                 default=False)
    outfile = pexConfig.Field("Output filename", str, default="test.fits")

class GridFitTask(pipeBase.Task):

    ConfigClass = GridFitConfig
    _DefaultName = "GridFitTask"

    @pipeBase.timeMethod
    def run(self, infile, grid_center_guess, ccd_type=None, 
            optics_grid_file=None):

        y0_guess, x0_guess = grid_center_guess

        ## Keywords for catalog
        x_kwd = self.config.x_kwd
        y_kwd = self.config.y_kwd
        xx_kwd = self.config.xx_kwd
        yy_kwd = self.config.yy_kwd
        flux_kwd = self.config.flux_kwd

        ## Get CCD geometry
        if ccd_type == 'ITL':
            ccd_geom = ITL_AMP_GEOM
        elif ccd_type == 'E2V':
            ccd_geom = E2V_AMP_GEOM
        else:
            ccd_geom = None

        ## Get source positions for fit
        src = fits.getdata(infile)

        srcY = src[x_kwd]
        srcX = src[y_kwd]

        ## Curate data here (remove bad shapes, fluxes, etc.)
        srcW = np.sqrt(np.square(src[xx_kwd]) + np.square(src[yy_kwd]))
        mask = (srcW > 4.)

        srcY = src[y_kwd][mask]
        srcX = src[x_kwd][mask]
        srcXX = src[xx_kwd][mask]
        srcYY = src[yy_kwd][mask]
        srcF = src[flux_kwd][mask]

        ## Optionally get existing normalized centroid shifts
        if optics_grid_file is not None:
            optics_grid = DistortedGrid.from_fits(optics_grid_file)
            normalized_shifts = (optics_grid.norm_dy, optics_grid.norm_dx)
        else:
            normalized_shifts = None

        ## Perform grid fit
        ncols = self.config.ncols
        nrows = self.config.nrows
        result = grid_fit(srcY, srcX, y0_guess, x0_guess, ncols, nrows,
                          brute_search=self.config.brute_search,
                          vary_theta=self.config.vary_theta,
                          normalized_shifts=normalized_shifts,
                          ccd_geom=ccd_geom)

        ## Make best fit source grid
        parvals = result.params.valuesdict()
        grid = DistortedGrid(parvals['ystep'], parvals['xstep'], parvals['theta'], 
                        parvals['y0'], parvals['x0'],
                        ncols, nrows, normalized_shifts=normalized_shifts)

        grid.write_fits(self.config.outfile, overwrite=True)

        return grid, result

import os
import numpy as np
from os.path import join
from astropy.io import fits

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from .sourcegrid import DistortedGrid, grid_fit, coordinate_distances
from .utils import ITL_AMP_GEOM, E2V_AMP_GEOM

class GridFitConfig(pexConfig.Config):
    """Configuration for GridFitTask."""

    nrows = pexConfig.Field("Number of grid rows.", int, default=49)
    ncols = pexConfig.Field("Number of grid columns.", int, default=49)
    y_kwd = pexConfig.Field("Source catalog y-position keyword", str, 
                            default='base_SdssCentroid_Y')
    x_kwd = pexConfig.Field("Source catalog y-position keyword", str, 
                            default='base_SdssCentroid_X')
    brute_search = pexConfig.Field("Perform prelim brute search", bool,
                                   default=False)
    vary_theta = pexConfig.Field("Vary theta parameter during fit", bool,
                                 default=False)
    outfile = pexConfig.Field("Output filename", str, default="test.cat")

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
        xx_kwd = 'base_SdssShape_XX'
        yy_kwd = 'base_SdssShape_YY'

        ## Get CCD geometry
        if ccd_type == 'ITL':
            ccd_geom = ITL_AMP_GEOM
        elif ccd_type == 'E2V':
            ccd_geom = E2V_AMP_GEOM
        else:
            ccd_geom = None

        ## Get source positions for fit
        with fits.open(infile) as src:

            all_srcY = src[1].data[y_kwd]
            all_srcX = src[1].data[x_kwd]

            ## Curate data here (remove bad shapes, fluxes, etc.)
            all_srcW = np.sqrt(np.square(src[1].data[xx_kwd]) + \
                                   np.square(src[1].data[yy_kwd]))
            mask = (all_srcW > 4.)

            srcY = all_srcY[mask]
            srcX = all_srcX[mask]

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
            grid = DistortedGrid(parvals['ystep'], parvals['xstep'], 
                                 parvals['theta'], parvals['y0'], 
                                 parvals['x0'], ncols, nrows, 
                                 normalized_shifts=normalized_shifts)

            ## Match grid to catalog
            gY, gX = grid.get_source_centroids()
            indices, dist = coordinate_distances(gY, gX, all_srcY, all_srcX)
            nn_indices = indices[:, 0]

            ## Populate grid information
            grid_index = np.full(all_srcX.shape[0], np.nan)
            grid_y = np.full(all_srcX.shape[0], np.nan)
            grid_x = np.full(all_srcX.shape[0], np.nan)
            grid_y[nn_indices] = gY
            grid_x[nn_indices] = gX
            grid_index[nn_indices] = np.arange(49*49)

            ## Merge tables
            new_cols = fits.ColDefs([fits.Column(name='spotgrid_index', 
                                                 format='D', array=grid_index),
                                     fits.Column(name='spotgrid_x', 
                                                 format='D', array=grid_x),
                                     fits.Column(name='spotgrid_y', 
                                                 format='D', array=grid_y)])
            cols = src[1].columns
            new_hdu = fits.BinTableHDU.from_columns(cols+new_cols)
            src[1] = new_hdu

            ## Append grid HDU
            grid_hdu = grid.make_grid_hdu()
            src.append(grid_hdu)
            src.writeto(self.config.outfile, overwrite=True)

        return grid, result

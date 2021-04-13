import os
import numpy as np
from os.path import join
from astropy.io import fits
from scipy.spatial import distance

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
    xx_kwd = pexConfig.Field('Source catalog xx-shape keyword', str, 
                             default="base_SdssShape_XX")
    yy_kwd = pexConfig.Field('Source catalog yy-shape keyword', str,
                             default="base_SdssShape_YY")
    vary_theta = pexConfig.Field("Vary theta parameter during fit", bool,
                                 default=True)
    fit_method = pexConfig.Field("Method for fit", str,
                                 default='least_squares')
    outfile = pexConfig.Field("Output filename", str, default="test.cat")

class GridFitTask(pipeBase.Task):

    ConfigClass = GridFitConfig
    _DefaultName = "GridFitTask" 
    
    @pipeBase.timeMethod
    def run(self, infile, ccd_type=None, optics_grid_file=None):

        ## Keywords for catalog
        x_kwd = self.config.x_kwd
        y_kwd = self.config.y_kwd
        xx_kwd = self.config.xx_kwd
        yy_kwd = self.config.yy_kwd

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
            allsrc_points = np.asarray([[x,y] for x,y in zip(all_srcX,all_srcY)])
            
            # Mask the bad grid points
            quality_mask = (src[1].data['base_SdssShape_XX'] > 4.5) \
                         * (src[1].data['base_SdssShape_XX'] < 7.)  \
                         * (src[1].data['base_SdssShape_YY'] > 4.5) \
                         * (src[1].data['base_SdssShape_YY'] < 7.)

            idx, dist = coordinate_distances(all_srcY, all_srcX, all_srcY, all_srcX)
            check = (dist < 100.) & (dist > 40.)
            outlier_mask = np.sum(check[:,1:3], axis=1) >= 2

            full_mask = quality_mask & outlier_mask

            srcY = all_srcY[full_mask]
            srcX = all_srcX[full_mask]
            
            ## Optionally get existing normalized centroid shifts
            if optics_grid_file is not None:
                optics_grid = DistortedGrid.from_fits(optics_grid_file)
                normalized_shifts = (optics_grid.norm_dy, optics_grid.norm_dx)
            else:
                normalized_shifts = None
                
            ## Perform grid fit
            ncols = self.config.ncols
            nrows = self.config.nrows
            grid, result = grid_fit(srcY, srcX, ncols, nrows,
                              vary_theta=self.config.vary_theta,
                              normalized_shifts=normalized_shifts,
                              method=self.config.fit_method,
                              ccd_geom=ccd_geom)

            ## Match grid to catalog
            gY, gX = grid.get_source_centroids()
            indices, dist = coordinate_distances(gY, gX, srcY, srcX)
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

        return grid, result, srcY, srcX
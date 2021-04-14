"""
To Do:
    1. Get CCD geometry information using DM tools and replace existing AMP_GEOM
    2. Add optics_distortion_grid information to Connections (as another source catalog, maybe?)
    3. Update modification of the source catalog to use DM tools.
"""
import os
import numpy as np
from os.path import join
from astropy.io import fits
from scipy.spatial import distance

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field

from .sourcegrid import DistortedGrid, grid_fit, coordinate_distances
from .utils import ITL_AMP_GEOM, E2V_AMP_GEOM

class GridFitConnections(pipeBase.PipelineTaskConnections, 
                         dimensions=("instrument", "exposure", "detector")):

    inputCat = cT.Input(
        doc="Source catalog produced by characterize spot task.",
        name='spotSrc',
        storageClass="SourceCatalog",
        dimensions=("instrument", "exposure", "detector")
    )

    vaSourceCat = cT.Output(
        doc="Value added source catalog produced by grid fit task.",
        name="vaSrc",
        storageClass="SourceCatalog",
        dimensions=("instrument", "exposure", "detector")
    )

class GridFitConfig(pipebase.PipelineTaskConfig,
                    pipelineConnections=GridFitConnections):
    """Configuration for GridFitTask."""

    numRows = Field(
        dtype=int,
        default=49,
        doc="Number of grid rows."
    )
    numColumns = Field(
        dtype=int,
        default=49,
        doc="Number of grid columns."
    )
    varyTheta = Field(
        dtype=bool,
        default=True,
        doc="Vary theta parameter during model fit."
    )
    fitMethod = Field(
        dtype=str,
        default='least_squares',
        doc="Minimization method for model fit."
    )

class GridFitTask(pipeBase.PipelineTask):

    ConfigClass = GridFitConfig
    _DefaultName = "GridFitTask" 
    
    @pipeBase.timeMethod
    def run(self, inputCat):

        ## Need to figure out how to add to connections
        optics_grid_file = None
        ccd_type = None ## will probably error since needs geom info

        ## Get CCD geometry
        if ccd_type == 'ITL':
            ccd_geom = ITL_AMP_GEOM
        elif ccd_type == 'E2V':
            ccd_geom = E2V_AMP_GEOM
        else:
            ccd_geom = None

        src = inputCat.asAstropy()

        all_srcY = src['base_SdssCentroid_Y']
        all_srcX = src['base_SdssCentroid_X']
        
        # Mask the bad grid points
        quality_mask = (src['base_SdssShape_XX'] > 4.5) \
                     * (src['base_SdssShape_XX'] < 7.)  \
                     * (src['base_SdssShape_YY'] > 4.5) \
                     * (src['base_SdssShape_YY'] < 7.)

        indices, distances = coordinate_distances(all_srcY, all_srcX, all_srcY, all_srcX)
        outlier_mask = ((distances[:,1] < 100.) & (distances[:,1] > 40.)) & ((distances[:,2] < 100.) & (distances[:,2] > 40.))

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
        grid, result = grid_fit(srcY, srcX, self.config.numColumns, self.config.numRows,
                          vary_theta=self.config.varyTheta,
                          normalized_shifts=normalized_shifts,
                          method=self.config.fitMethod,
                          ccd_geom=ccd_geom)

        ## Match grid to catalog
        gY, gX = grid.get_source_centroids()
        closest_indices, closest_distances = coordinate_distances(srcY, srcX, gY, gX)
        grid_y[full_mask] = gY[closest_indices[:, 0]]
        grid_x[full_mask] = gX[closest_indices[:, 0]]
        grid_index[full_mask] = closest_indices[:, 0]

        ## Merge tables
        ## How do I convert this to DM stuff?
        new_cols = fits.ColDefs([fits.Column(name='spotgrid_index', 
                                             format='D', array=grid_index),
                                 fits.Column(name='spotgrid_x', 
                                             format='D', array=grid_x),
                                 fits.Column(name='spotgrid_y', 
                                             format='D', array=grid_y)])
        cols = src.columns
        new_hdu = fits.BinTableHDU.from_columns(cols+new_cols)
        grid_hdu = grid.make_grid_hdu()
        new_src = fits.HDUList([new_hdu, grid_hdu])
        new_src.writeto('test.cat', overwrite=True)

    return pipeBase.Struct(vaSourceCat=src)

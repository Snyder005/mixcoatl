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

        all_srcY = inputCat['base_SdssCentroid_Y']
        all_srcX = inputCat['base_SdssCentroid_X']
        
        # Mask the bad grid points
        quality_mask = (inputCat['base_SdssShape_XX'] > 4.5) \
                     * (inputCat['base_SdssShape_XX'] < 7.)  \
                     * (inputCat['base_SdssShape_YY'] > 4.5) \
                     * (inputCat['base_SdssShape_YY'] < 7.)

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
        grid_y = np.full(all_srcY.shape[0], np.nan)
        grid_y = np.full(all_srcX.shape[0], np.nan)
        grid_index = np.full(all_srcX.shape[0], np.nan)

        gY, gX = grid.get_source_centroids()
        closest_indices, closest_distances = coordinate_distances(srcY, srcX, gY, gX)
        grid_y[full_mask] = gY[closest_indices[:, 0]]
        grid_x[full_mask] = gX[closest_indices[:, 0]]
        grid_index[full_mask] = closest_indices[:, 0]

        ## Add spot grid information to new source catalog
        schema = inputCat.getSchema()
        mapper = afwTable.SchemaMapper(schema)
        mapper.addMinimalSchema(schema, True)
        grid_y_col = mapper.editOutputSchema().addField('spotgrid_y', type=float,
                                                        doc='Y-position for ideal spot grid.')
        grid_x_col = mapper.editOutputSchema().addField('spotgrid_x', type=float,
                                                        doc='X-position for ideal spot grid.')
        grid_index_col = mapper.editOutputSchema().addField('spotgrid_index', type=int,
                                                            doc='Index of ideal spot grid.')
    
        outputCat = afwTable.SourceCatalog(mapper.getOutputSchema())
        outputCat.extend(inputCat, mapper=mapper)
        outputCat[grid_y_col][:] = grid_y
        outputCat[grid_x_col][:] = grid_x
        outputCat[grid_index_col][:] = grid_index
        
        ## Add grid parameters to metadata
        md = inputCat.getMetadata()
        md.add('GRID_X0', grid.x0)
        md.add('GRID_Y0', grid.y0)
        md.add('GRID_THETA', grid.theta)
        md.add('GRID_XSTEP', grid.xstep)
        md.add('GRID_YSTEP', grid.ystep)
        md.add('GRID_NCOLS', grid.ncols)
        md.add('GRID_NROWS', grid.nrows)

        outputCat.setMetadata(md)

    return pipeBase.Struct(vaSourceCat=outputCat)

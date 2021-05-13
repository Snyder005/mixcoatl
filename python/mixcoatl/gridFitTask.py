"""
To Do:
    1. Add optics_distortion_grid information to Connections (as another source catalog, maybe?)
    2. Determine how grid information is stored in DM butler world.
"""
import numpy as np
from scipy.spatial import distance

import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field

from .sourcegrid import DistortedGrid, grid_fit, coordinate_distances

class GridFitConnections(pipeBase.PipelineTaskConnections, 
                         dimensions=("instrument", "exposure", "detector")):

    inputCat = cT.Input(
        doc="Source catalog produced by characterize spot task.",
        name='spotSrc',
        storageClass="SourceCatalog",
        dimensions=("instrument", "exposure", "detector")
    )
    bbox = cT.Input(
        doc="Bounding box for CCD.",
        name="postISRCCD.bbox",
        storageClass="Box2I",
        dimensions=("instrument", "exposure", "detector")
    )
    gridSourceCat = cT.Output(
        doc="Source catalog produced by grid fit task.",
        name="gridSpotSrc",
        storageClass="SourceCatalog",
        dimensions=("instrument", "exposure", "detector")
    )

class GridFitConfig(pipeBase.PipelineTaskConfig,
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
    def run(self, inputCat, bbox, gridCalib=None):

        all_srcY = inputCat['base_SdssCentroid_y']
        all_srcX = inputCat['base_SdssCentroid_x']
        
        # Mask the bad grid points
        quality_mask = (inputCat['base_SdssShape_xx'] > 0.1) \
                     * (inputCat['base_SdssShape_xx'] < 50.)  \
                     * (inputCat['base_SdssShape_yy'] > 0.1) \
                     * (inputCat['base_SdssShape_yy'] < 50.)

        indices, distances = coordinate_distances(all_srcY, all_srcX, all_srcY, all_srcX)
        outlier_mask = ((distances[:,1] < 100.) & (distances[:,1] > 40.)) & \
            ((distances[:,2] < 100.) & (distances[:,2] > 40.))

        full_mask = quality_mask & outlier_mask
        srcY = all_srcY[full_mask]
        srcX = all_srcX[full_mask]
        
        ## Optionally get existing normalized centroid shifts
        if gridCalib is not None:
            normalized_shifts = (gridCalib.norm_dy, gridCalib.norm_dx)
        else:
            normalized_shifts = None
            
        ## Perform grid fit
        grid, result = grid_fit(srcY, srcX, self.config.numColumns, self.config.numRows,
                                vary_theta=self.config.varyTheta, normalized_shifts=normalized_shifts,
                                method=self.config.fitMethod, bbox=bbox)

        ## Match grid to catalog
        grid_y = np.full(all_srcY.shape[0], np.nan)
        grid_x = np.full(all_srcX.shape[0], np.nan)
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
        grid_index_col = mapper.editOutputSchema().addField('spotgrid_index', type=np.int32,
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

        return pipeBase.Struct(gridSourceCat=outputCat)

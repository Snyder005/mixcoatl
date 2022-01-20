"""
To Do:
    1. Add gridCalibTable information to Connections and setup optional Butler inclusion (runQuantum?).
"""
import numpy as np
from scipy.spatial import distance
import copy

import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field
from lsst.utils.timer import timeMethod

from .sourcegrid import DistortedGrid, grid_fit, coordinate_distances, find_midpoint_guess

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
    gridCalibTable = cT.PrerequisiteInput(
        doc="Calibration table for spot grid.",
        name="gridCalibration",
        storageClass="AstropyTable",
        dimensions=("instrument", "detector"),
        isCalibration=True
    )
    gridSourceCat = cT.Output(
        doc="Source catalog produced by grid fit task.",
        name="gridSpotSrc",
        storageClass="SourceCatalog",
        dimensions=("instrument", "exposure", "detector")
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.useGridCalibration is not True:
            self.prerequisiteInputs.discard("gridCalibTable")

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
    useGridCalibration = Field(
        dtype=bool,
        default=False,
        doc="Use centroid shifts from grid calibration table?"
    )
    shapeLowerBound = Field(
        dtype=float,
        default=2.0,
        doc="Lower bound on source second moment; used for masking."
    )
    shapeUpperBound = Field(
        dtype=float,
        default=20.,
        doc="Upper bound on source second moment; used for masking."
    )
    distanceFromVertex = Field(
        dtype=float,
        default=5.,
        doc="Allowable distance of each source from a possible grid vertex."
    )

class GridFitTask(pipeBase.PipelineTask):

    ConfigClass = GridFitConfig
    _DefaultName = "GridFitTask" 

    @timeMethod
    def run(self, inputCat, bbox, gridCalibTable=None):

        all_srcY = inputCat['slot_Centroid_y']
        all_srcX = inputCat['slot_Centroid_x']

        ## Mask sources by shape
        select_by_shape = (inputCat['slot_Shape_xx'] > self.config.shapeLowerBound) \
                        * (inputCat['slot_Shape_xx'] < self.config.shapeUpperBound) \
                        * (inputCat['slot_Shape_yy'] > self.config.shapeLowerBound) \
                        * (inputCat['slot_Shape_yy'] < self.config.shapeUpperBound)

        ## Estimate grid properties
        indices, distances = coordinate_distances(all_srcY, all_srcX, all_srcY, all_srcX)
        nn_indices = indices[select_by_shape, 1:5]
        nn_distances = distances[select_by_shape, 1:5]
        med_distance = np.median(nn_distances)

        nsources = all_srcY[select_by_shape].shape[0]
        dist1_array = np.full(nsources, np.nan)
        dist2_array = np.full(nsources, np.nan)
        theta_array = np.full(nsources, np.nan)

        for i in range(nsources):

            yc = all_srcY[select_by_shape][i]
            xc = all_srcX[select_by_shape][i]

            for j in range(4):

                nn_dist = nn_distances[i, j]
                if np.abs(nn_dist - med_distance) > 5.: 
                    continue
                y_nn = all_srcY[nn_indices[i, j]]
                x_nn = all_srcX[nn_indices[i, j]]

                if x_nn > xc:
                    if y_nn > yc:
                        dist1_array[i] = nn_dist
                        theta_array[i] = np.arctan((y_nn-yc)/(x_nn-xc))
                    else:
                        dist2_array[i] = nn_dist

        ## Use theta to determine x/y step direction
        theta = np.nanmedian(theta_array)
        if theta >= np.pi/4.:
            theta = theta - (np.pi/2.)
            xstep = np.nanmedian(dist2_array)
            ystep = np.nanmedian(dist1_array)
        else:
            xstep = np.nanmedian(dist1_array)
            ystep = np.nanmedian(dist2_array)

        ## Rotate to pixel x/y
        R = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
        points = (R @ np.vstack([all_srcX, all_srcY]))

        ## Mask by distance to possible grid vertex
        select_by_distance = np.zeros(len(all_srcX), bool)
        for masked_idx, main_idx in enumerate(np.argwhere(select_by_shape)[:, 0]):
        
            pt = points.T[main_idx]
            
            xs = np.delete(points[0][select_by_shape], masked_idx)
            ys = np.delete(points[1][select_by_shape], masked_idx)
            if np.any(xs, where=np.isclose(pt[0],xs, rtol=0, atol=self.config.distanceFromVertex)) and \
               np.any(ys, where=np.isclose(pt[1],ys, rtol=0, atol=self.config.distanceFromVertex)):
                select_by_distance[main_idx] = True
            else:
                select_by_distance[main_idx] = False
  
        select = select_by_shape*select_by_distance

        srcY = all_srcY[select]
        srcX = all_srcX[select]

        ## Calculate intial guess for grid center
        grid_center_guess = find_midpoint_guess(srcY, srcX, xstep, ystep, theta)
        y0, x0 = grid_center_guess[1], grid_center_guess[0]
       
        ## Optionally use normalized centroid shifts from calibration
        if gridCalibTable is not None:
            gridCalib = DistortedGrid.from_astropy(gridCalibTable)
            normalized_shifts = (gridCalib.norm_dy, gridCalib.norm_dx)
        else:
            normalized_shifts = None
            
        ## Perform grid fit
        params = (ystep, xstep, theta, y0, x0)
        grid, result = grid_fit(srcY, srcX, self.config.numColumns, self.config.numRows, params, bbox=bbox,
                                vary_theta=self.config.varyTheta, method=self.config.fitMethod,
                                normalized_shifts=normalized_shifts)

        ## Construct source catalog with new columns
        schema = inputCat.getSchema()
        mapper = afwTable.SchemaMapper(schema)
        mapper.addMinimalSchema(schema, True)
        gridYCol = mapper.editOutputSchema().addField('spotgrid_y', type=float,
                                                      doc='Y-position for spot grid source.')
        gridXCol = mapper.editOutputSchema().addField('spotgrid_x', type=float,
                                                      doc='X-position for spot grid source.')
        normDYCol = mapper.editOutputSchema().addField('spotgrid_normalized_dy', type=float,
                                                       doc='Normalized shift from spot grid source in Y.')
        normDXCol = mapper.editOutputSchema().addField('spotgrid_normalized_dx', type=float,
                                                       doc='Normalized shift from spot grid source in X.')
        gridIndexCol = mapper.editOutputSchema().addField('spotgrid_index', type=np.int32,
                                                          doc='Index of corresponding spot grid source.')

        outputCat = afwTable.SourceCatalog(mapper.getOutputSchema())
        outputCat.extend(inputCat, mapper=mapper)

        ## Match grid to catalog
        gridY, gridX = grid.get_centroids()
        match_indices, match_distances = coordinate_distances(srcY, srcX, gridY, gridX)

        ## Construct new column arrays
        numSrcs = all_srcY.shape[0]
        all_gridY = np.full(numSrcs, np.nan)
        all_gridX = np.full(numSrcs, np.nan)
        all_normDY = np.full(numSrcs, np.nan)
        all_normDX = np.full(numSrcs, np.nan)
        all_gridIndex = np.full(numSrcs, np.nan)

        all_gridY[select] = gridY[match_indices[:, 0]]
        all_gridX[select] = gridX[match_indices[:, 0]]
        all_gridIndex[select] = match_indices[:, 0] 
        dy = all_srcY - all_gridY
        dx = all_srcX - all_gridX
        all_normDY = (np.sin(-grid.theta)*dx + np.cos(-grid.theta)*dy)/grid.ystep
        all_normDX = (np.cos(-grid.theta)*dx - np.sin(-grid.theta)*dy)/grid.xstep 

        ## Assign new column arrays to catalog
        outputCat[gridYCol][:] = all_gridY
        outputCat[gridXCol][:] = all_gridX
        outputCat[normDYCol][:] = all_normDY
        outputCat[normDXCol][:] = all_normDX
        outputCat[gridIndexCol][:] = all_gridIndex
        
        ## Add grid parameters to metadata
        md = copy.deepcopy(inputCat.getMetadata())
        md.add('GRID_X0', grid.x0)
        md.add('GRID_Y0', grid.y0)
        md.add('GRID_THETA', grid.theta)
        md.add('GRID_XSTEP', grid.xstep)
        md.add('GRID_YSTEP', grid.ystep)
        md.add('GRID_NCOLS', grid.ncols)
        md.add('GRID_NROWS', grid.nrows)
        outputCat.setMetadata(md)

        return pipeBase.Struct(gridSourceCat=outputCat)

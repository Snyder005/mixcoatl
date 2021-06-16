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
        default=50.,
        doc="Upper bound on source second moment; used for masking."
    )
    neighborDistanceVariation = Field(
        dtype=float,
        default=5.,
        doc="Allowable variation of distance to nearest source neighbor; used for masking."
    )

class GridFitTask(pipeBase.PipelineTask):

    ConfigClass = GridFitConfig
    _DefaultName = "GridFitTask" 

    def run(self, inputCat, bbox, gridCalibTable=None):
        
        ## Mask sources by shape
        select_by_shape = (inputCat['slot_Shape_xx'] > self.config.shapeLowerBound) \
                        * (inputCat['slot_Shape_xx'] < self.config.shapeUpperBound) \
                        * (inputCat['slot_Shape_yy'] > self.config.shapeLowerBound) \
                        * (inputCat['slot_Shape_yy'] < self.config.shapeUpperBound)

        all_srcY = inputCat['slot_Centroid_y'][select_by_shape]
        all_srcX = inputCat['slot_Centroid_x'][select_by_shape]

        ## Estimate grid properties
        indices, distances = coordinate_distances(all_srcY, all_srcX, all_srcY, all_srcX)
        nn_indices = indices[:, 1:5]
        nn_distances = distances[:, 1:5]
        med_distance = np.median(nn_distances)

        nsources = all_srcY.shape[0]
        dist1_array = np.full(nsources, np.nan)
        dist2_array = np.full(nsources, np.nan)
        theta_array = np.full(nsources, np.nan)

        for i in range(nsources):

            yc = all_srcY[i]
            xc = all_srcX[i]

            for j in range(4):

                nn_dist = nn_distances[i, j]
                if np.abs(nn_dist - med_distance) > self.config.neighborDistanceVariation: continue
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
        points = (R @ np.vstack([all_srcX,all_srcY]))
        xs = points[0]
        ys = points[1]

        ## Mask by distance to possible grid vertex
        mask = np.zeros(len(all_srcX), bool)
        for i,pt in enumerate(points.T):
            xs = np.delete(points[0], i)
            ys = np.delete(points[1], i)
            if np.any(xs, where=np.isclose(pt[0],xs, rtol=0, atol=self.config.neighborDistanceVariation)) and \
               np.any(ys, where=np.isclose(pt[1],ys, rtol=0, atol=self.config.neighborDistanceVariation)):
                mask[i] = True
            else:
                mask[i] = False
                
        srcY = all_srcY[mask]
        srcX = all_srcX[mask]
       
        ## Optionally use normalized centroid shifts from calibration
        if gridCalibTable is not None:
            gridCalib = DistortedGrid.from_astropy(gridCalibTable)
            normalized_shifts = (gridCalib.norm_dy, gridCalib.norm_dx)
        else:
            normalized_shifts = None
            
        ## Perform grid fit
        grid, result = self.performGridFit(srcY, srcX, ystep, xstep, theta, bbox=bbox,
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

        all_gridY[mask] = gridY[match_indices[:, 0]]
        all_gridX[mask] = gridX[match_indices[:, 0]]
        all_gridIndex[mask] = match_indices[:, 0] 
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

    def performGridFit(self, src_y, src_x, ystep, xstep, theta, normalized_shifts=None, bbox=None):

        ncols = self.config.numCols
        nrows = self.config.numRows

        ## Find initial guess for grid center based on orientation
        grid_center_guess = mixCrosstalk.find_midpoint_guess(src_y, src_x, xstep, ystep, theta)
        y0_guess, x0_guess = grid_center_guess[1], grid_center_guess[0]

        ## Define fit parameters
        params = Parameters()
        params.add('ystep', value=ystep, vary=False)
        params.add('xstep', value=xstep, vary=False)
        params.add('y0', value=y0_guess, min=y0_guess-3., max=y0_guess+3., vary=True)
        params.add('x0', value=x0_guess, min=x0_guess-3., max=x0_guess+3., vary=True)
        params.add('theta', value=theta, min=theta-0.5*np.pi/180., max=theta+0.5*np.pi/180., vary=False)

        minner = Minimizer(mixCrosstalk.fit_error, params, fcn_args=(src_y, src_x, ncols, nrows),
                           fcn_kws={'normalized_shifts' : normalized_shifts,
                                    'bbox' : bbox}, nan_policy='omit')
        result = minner.minimize(params=params, method=self.config.fitMethod, max_nfev=None)
        x0result = result.params['x0']
        y0result = result.params['y0']

        if self.config.varyTheta:
            result_params = result.params
            result_values = result_params.valuesdict()
            params['y0'].set(value=result_values['y0'], vary=False)
            params['x0'].set(value=result_values['x0'], vary=False)
            params['theta'].set(vary=True)
            theta_minner = Minimizer(fit_error, params, fcn_args=(src_y, src_x, ncols, nrows),
                           fcn_kws={'normalized_shifts' : normalized_shifts,
                                    'bbox' : bbox}, nan_policy='omit')
            theta_result = theta_minner.minimize(params=params, method=self.config.fitMethod, max_nfev=None)
            result.params['theta'] = theta_result.params['theta']

        parvals = result.params.valuesdict()
        grid = mixCrosstalk.DistortedGrid(parvals['ystep'], parvals['xstep'], 
                                          parvals['theta'], parvals['y0'], 
                                          parvals['x0'], ncols, nrows, 
                                          normalized_shifts=normalized_shifts)
        
        return grid, result

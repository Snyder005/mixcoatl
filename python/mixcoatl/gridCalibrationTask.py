import numpy as np
from astropy.table import Table, vstack
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.utils.timer import timeMethod

cols_to_aggregate = ['spotgrid_index',
                     'spotgrid_normalized_dy',
                     'spotgrid_normalized_dx',
                     'base_SdssCentroid_xErr',
                     'base_SdssCentroid_yErr',
                     'base_SdssShape_xx',
                     'base_SdssShape_yy',
                     'base_SdssShape_xy',
                     'base_SdssShape_xxErr',
                     'base_SdssShape_yyErr',
                     'base_SdssShape_xyErr',
                     'base_SdssShape_x',
                     'base_SdssShape_y',
                     'base_SdssShape_instFlux',
                     'base_SdssShape_instFluxErr',
                     'base_SdssShape_psf_xx',
                     'base_SdssShape_psf_yy',
                     'base_SdssShape_psf_xy',
                     'base_SdssShape_instFlux_xx_Cov',
                     'base_SdssShape_instFlux_yy_Cov',
                     'base_SdssShape_instFlux_xy_Cov',
                     'ext_shapeHSM_HsmPsfMoments_x',
                     'ext_shapeHSM_HsmPsfMoments_y',
                     'ext_shapeHSM_HsmPsfMoments_xx',
                     'ext_shapeHSM_HsmPsfMoments_yy',
                     'ext_shapeHSM_HsmPsfMoments_xy',
                     'ext_shapeHSM_HsmShapeBj_e1',
                     'ext_shapeHSM_HsmShapeBj_e2',
                     'ext_shapeHSM_HsmShapeBj_sigma',
                     'ext_shapeHSM_HsmShapeBj_resolution',
                     'ext_shapeHSM_HsmShapeKsb_g1',
                     'ext_shapeHSM_HsmShapeKsb_g2',
                     'ext_shapeHSM_HsmShapeKsb_sigma',
                     'ext_shapeHSM_HsmShapeKsb_resolution',
                     'ext_shapeHSM_HsmShapeLinear_e1',
                     'ext_shapeHSM_HsmShapeLinear_e2',
                     'ext_shapeHSM_HsmShapeLinear_sigma',
                     'ext_shapeHSM_HsmShapeLinear_resolution',
                     'ext_shapeHSM_HsmShapeRegauss_e1',
                     'ext_shapeHSM_HsmShapeRegauss_e2',
                     'ext_shapeHSM_HsmShapeRegauss_sigma',
                     'ext_shapeHSM_HsmShapeRegauss_resolution',
                     'ext_shapeHSM_HsmSourceMoments_x',
                     'ext_shapeHSM_HsmSourceMoments_y',
                     'ext_shapeHSM_HsmSourceMoments_xx',
                     'ext_shapeHSM_HsmSourceMoments_yy',
                     'ext_shapeHSM_HsmSourceMoments_xy',
                     'base_CircularApertureFlux_3_0_instFlux',
                     'base_CircularApertureFlux_3_0_instFluxErr',
                     'base_CircularApertureFlux_4_5_instFlux',
                     'base_CircularApertureFlux_4_5_instFluxErr',
                     'base_CircularApertureFlux_6_0_instFlux',
                     'base_CircularApertureFlux_6_0_instFluxErr',
                     'base_CircularApertureFlux_9_0_instFlux',
                     'base_CircularApertureFlux_9_0_instFluxErr',
                     'base_CircularApertureFlux_12_0_instFlux',
                     'base_CircularApertureFlux_12_0_instFluxErr',
                     'base_CircularApertureFlux_17_0_instFlux',
                     'base_CircularApertureFlux_17_0_instFluxErr',
                     'base_CircularApertureFlux_25_0_instFlux',
                     'base_CircularApertureFlux_25_0_instFluxErr',
                     'base_CircularApertureFlux_35_0_instFlux',
                     'base_CircularApertureFlux_35_0_instFluxErr',
                     'base_CircularApertureFlux_50_0_instFlux',
                     'base_CircularApertureFlux_50_0_instFluxErr',
                     'base_CircularApertureFlux_70_0_instFlux',
                     'base_CircularApertureFlux_70_0_instFluxErr',
                     'base_GaussianFlux_instFlux',
                     'base_GaussianFlux_instFluxErr',
                     'base_PsfFlux_instFlux',
                     'base_PsfFlux_instFluxErr',
                     'base_PsfFlux_area',
                     'base_ClassificationExtendedness_value',
                     'base_FootprintArea_value']

class GridCalibrationConnections(pipeBase.PipelineTaskConnections,
                                 dimensions=("instrument", "detector")):

    inputCatalogs = cT.Input(
        doc="Source catalogs for fit spot grid.",
        name="gridSpotSrc",
        storageClass="SourceCatalog",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True
    )
    outputTable = cT.Output(
        doc="Calibration source catalog for spot grid.",
        name="gridCalibration",
        storageClass="AstropyTable",
        dimensions=("instrument", "detector"),
        multiple=False,
        isCalibration=True
    )

class GridCalibrationConfig(pipeBase.PipelineTaskConfig,
                            pipelineConnections=GridCalibrationConnections):
    """Configuration for Calibration Task"""
    
    numCols = pexConfig.Field("Number of ideal grid columns", int, default=49)
    numRows = pexConfig.Field("Number of ideal grid rows", int, default=49)

class GridCalibrationTask(pipeBase.PipelineTask):
    """Task to combine distorted grid fits."""
    
    ConfigClass = GridCalibrationConfig
    _DefaultName = "GridCalibrationTask"

    @timeMethod
    def run(self, inputCatalogs):

        ## Grid parameters
        all_x0 = np.zeros(len(inputCatalogs))
        all_y0 = np.zeros(len(inputCatalogs))
        all_xstep = np.zeros(len(inputCatalogs))
        all_ystep = np.zeros(len(inputCatalogs))
        all_theta = np.zeros(len(inputCatalogs))
        all_data_tables = []

        for i, inputCat in enumerate(inputCatalogs):
            
            md = inputCat.getMetadata()

            if (md['GRID_NROWS'],  md['GRID_NCOLS']) != (self.config.numRows, self.config.numCols):
                raise ValueError('Grid sizes do not match: ({0}, {1}), ({1}, {2})'\
                    .format(md['GRID_NROWS'], md['GRID_NCOLS'], self.config.numRows, self.config.numCols))

            all_x0[i] = md['GRID_X0']
            all_y0[i] = md['GRID_Y0']
            all_xstep[i] = md['GRID_XSTEP']
            all_ystep[i] = md['GRID_YSTEP']
            all_theta[i] = md['GRID_THETA']
            
            all_data_tables.append(inputCat.asAstropy())
            
        ## Compute data means
        meta = {'GRID_X0' : np.nanmedian(all_x0),
                'GRID_Y0' : np.nanmedian(all_y0),
                'GRID_XSTEP' : np.nanmedian(all_xstep),
                'GRID_YSTEP' : np.nanmedian(all_ystep),
                'GRID_THETA' : np.nanmedian(all_theta),
                'GRID_NROWS' : self.config.numRows,
                'GRID_NCOLS' : self.config.numCols}
        outputTable = vstack(all_data_tables, join_type='exact', metadata_conflicts='silent')\
                             .group_by('spotgrid_index')
        outputTable = outputTable[cols_to_aggregate].groups.aggregate(np.nanmedian)
        outputTable.remove_rows(np.where(outputTable['spotgrid_index'] < 0)[0])
        outputTable.meta.update(meta)

        return pipeBase.Struct(outputTable=outputTable)

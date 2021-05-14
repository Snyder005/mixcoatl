import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from mixcoatl.sourcegrid import DistortedGrid

cols_to_aggregate = ['spotgrid_index',
                    'base_SdssCentroid_xErr',
                    'base_SdssCentroid_yErr',
                    'base_Blendedness_old',
                    'base_Blendedness_raw',
                    'base_Blendedness_raw_child_instFlux',
                    'base_Blendedness_raw_parent_instFlux',
                    'base_Blendedness_abs',
                    'base_Blendedness_abs_child_instFlux',
                    'base_Blendedness_abs_parent_instFlux',
                    'base_Blendedness_raw_child_xx',
                    'base_Blendedness_raw_child_yy',
                    'base_Blendedness_raw_child_xy',
                    'base_Blendedness_raw_parent_xx',
                    'base_Blendedness_raw_parent_yy',
                    'base_Blendedness_raw_parent_xy',
                    'base_Blendedness_abs_child_xx',
                    'base_Blendedness_abs_child_yy',
                    'base_Blendedness_abs_child_xy',
                    'base_Blendedness_abs_parent_xx',
                    'base_Blendedness_abs_parent_yy',
                    'base_Blendedness_abs_parent_xy',
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
                    'base_LocalBackground_instFlux',
                    'base_LocalBackground_instFluxErr',
                    'base_PsfFlux_instFlux',
                    'base_PsfFlux_instFluxErr',
                    'base_PsfFlux_area',
                    'base_Variance_value',
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
        name="gridCalibTable",
        storageClass="AstropyTable",
        dimensions=("instrument", "detector"),
        multiple=False,
        isCalibration=True
    )

class GridCalibrationConfig(pipeBase.PipelineTaskConfig,
                            pipelineConnections=GridCalibationConnections):
    """Configuration for Calibration Task"""
    
    ncols = pexConfig.Field("Number of ideal grid columns", int, default=49)
    nrows = pexConfig.Field("Number of ideal grid rows", int, default=49)

class GridCalibrationTask(pipeBase.PipelineTask):
    """Task to combine distorted grid fits."""
    
    ConfigClass = GridCalibrationConfig
    _DefaultName = "GridCalibrationTask"

    def run(self, inputCatalogs):

        ## Initialize data from distorted_grid
        all_x0 = np.zeros(len(infiles))
        all_y0 = np.zeros(len(infiles))
        all_xstep = np.zeros(len(infiles))
        all_ystep = np.zeros(len(infiles))
        all_theta = np.zeros(len(infiles))
        all_data_tables = []

        for i, infile in enumerate(infiles):
            grid = DistortedGrid.from_fits(infile)

            all_x0[i] = grid.x0
            all_y0[i] = grid.y0
            all_xstep[i] = grid.xstep
            all_ystep[i] = grid.ystep
            all_theta[i] = grid.theta

            with fits.open(infile) as hdulist:
                all_data_tables.append(Table(hdulist[1].data))


        ## Compute data means
        mean_x0 = np.nanmean(all_x0)
        mean_y0 = np.nanmean(all_y0)
        mean_xstep = np.nanmean(all_xstep)
        mean_ystep = np.nanmean(all_ystep)
        mean_theta = np.nanmean(all_theta)
        mean_data_table = vstack(all_data_tables, join_type='exact').group_by('spotgrid_index')
        mean_data_table = mean_data_table[cols_to_aggregate].groups.aggregate(np.nanmean)


        ## Create and save optical distortions grid
        grid = DistortedGrid(mean_ystep, mean_xstep, mean_theta, mean_y0, mean_x0, 
                             self.config.ncols, self.config.nrows)
        grid.make_source_grid()
        gY = grid._y
        gX = grid._x

        optics_grid = DistortedGrid(mean_ystep, mean_xstep, mean_theta, mean_y0, mean_x0, 
                                    self.config.ncols, self.config.nrows, (gY,gX))

        prihdr = fits.Header()
        prihdu = fits.PrimaryHDU(header=prihdr)
        gridhdu = optics_grid.make_grid_hdu()
        tablehdu = fits.BinTableHDU(mean_data_table)

        hdulist = fits.HDUList([prihdu, tablehdu, gridhdu])
        hdulist.writeto(os.path.join(self.config.output_dir, self.config.outfile), overwrite=True)

        return

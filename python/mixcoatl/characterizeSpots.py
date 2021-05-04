import numpy as np
import traceback
from scipy.ndimage import maximum_filter

from lsstDebug import getDebugFrame
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetect
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.pipe.base.connectionTypes as cT
import lsst.meas.extensions.shapeHSM
from lsst.afw.table import SourceTable, SourceCatalog
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask
from lsst.obs.base import ExposureIdInfo
from lsst.meas.base import SingleFrameMeasurementTask, CatalogCalculationTask
from lsst.pipe.tasks.repair import RepairTask
from lsst.pex.exceptions import LengthError

class CharacterizeSpotsConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", "exposure", "detector")):
    exposure = cT.Input(
        doc="Input exposure data",
        name="postISRCCD",
        storageClass="Exposure",
        dimensions=["instrument", "exposure", "detector"],
    )
    sourceCat = cT.Output(
        doc="Output source catalog.",
        name="spotSrc",
        storageClass="SourceCatalog",
        dimensions=["instrument", "exposure", "detector"],
    )
    outputSchema = cT.InitOutput(
        doc="Schema of the catalog produced by CharacterizeSpots",
        name="spotSrc_schema",
        storageClass="SourceCatalog",
    )

class CharacterizeSpotsConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=CharacterizeSpotsConnections):

    """!Config for CharacterizeSpotsTask"""
    doWrite = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Persist results?",
    )
    thresholdValue = pexConfig.Field(
        dtype=float,
        default=300.0,
        doc="Threshold applied to image value for footprints"
    )
    footprintGrowValue = pexConfig.Field(
        dtype=int,
        default=10,
        doc="Value to grow detected footprints"
    )
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Measure sources"
    )
    catalogCalculation = pexConfig.ConfigurableField(
        target=CatalogCalculationTask,
        doc="Subtask to run catalogCalculation plugins on catalog"
    )
    installSimplePsf = pexConfig.ConfigurableField(
        target=InstallGaussianPsfTask,
        doc="Install a simple PSF model",
    )
    repair = pexConfig.ConfigurableField(
        target=RepairTask,
        doc="Remove cosmic rays",
    )
    checkUnitsParseStrict = pexConfig.Field(
        doc="Strictness of Astropy unit compatibility check, can be 'raise', 'warn' or 'silent'",
        dtype=str,
        default="raise",
    )

    def setDefaults(self):
        super().setDefaults()
        # minimal set of measurements needed to determine PSF
        self.measurement.plugins.names = [
            "base_PixelFlags",
            "base_SdssCentroid",
            "base_SdssShape",
            "base_GaussianFlux",
            "base_PsfFlux",
            "base_CircularApertureFlux",
            "ext_shapeHSM_HsmShapeBj",
            "ext_shapeHSM_HsmShapeLinear",
            "ext_shapeHSM_HsmShapeKsb",
            "ext_shapeHSM_HsmShapeRegauss",
            "ext_shapeHSM_HsmSourceMoments",
            "ext_shapeHSM_HsmPsfMoments"
        ]
        
class CharacterizeSpotsTask(pipeBase.PipelineTask):

    ConfigClass = CharacterizeSpotsConfig
    _DefaultName = "characterizeSpots"

    def __init__(self, butler=None, schema=None, **kwargs):
        """!Construct a CharacterizeSpotsTask
        @param[in] butler  A butler object is passed to the refObjLoader constructor in case
            it is needed to load catalogs.  May be None if a catalog-based star selector is
            not used, if the reference object loader constructor does not require a butler,
            or if a reference object loader is passed directly via the refObjLoader argument.
        @param[in,out] schema  initial schema (an lsst.afw.table.SourceTable), or None
        @param[in,out] kwargs  other keyword arguments for lsst.pipe.base.CmdLineTask
        """
        super().__init__(**kwargs)

        if schema is None:
            schema = SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("installSimplePsf")
        self.makeSubtask("repair")
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask('measurement', schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask('catalogCalculation', schema=self.schema)
        self._initialFrame = getDebugFrame(self._display, "frame") or 1
        self._frame = self._initialFrame
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)
        self.outputSchema = afwTable.SourceCatalog(self.schema)
        
        try:
            self.repair.config.doCosmicRay = False
        except:
            ### YU is not sure why this happens but to get it going
            traceback.print_exc()
            

    def getInitOutputDatasets(self):
        outputCatSchema = afwTable.SourceCatalog(self.schema)
        outputCatSchema.getTable().setMetadata(self.algMetadata)
        return {'outputSchema': outputCatSchema}

    @pipeBase.timeMethod
    def run(self, exposure, exposureIdInfo=None):
        
        if not exposure.hasPsf():
            self.installSimplePsf.run(exposure=exposure)

        if exposureIdInfo is None:
            exposureIdInfo = ExposureIdInfo()

        try:
            self.repair.run(exposure=exposure, keepCRs=True)
        except LengthError:
            self.log.info("Skipping cosmic ray detection: Too many CR pixels (max %0.f)" % repair.cosmicray.nCrPixelMax)

        sourceIdFactory = exposureIdInfo.makeSourceIdFactory()
        table = SourceTable.make(self.schema, sourceIdFactory)
        table.setMetadata(self.algMetadata)

        filtered = maximum_filter(exposure.getImage().array, footprint=np.ones((50, 50)))
        detected = (filtered == exposure.getImage().getArray()) & (filtered > self.config.thresholdValue)

        detectedImage = afwImage.ImageF(detected.astype(np.float32))
        fps = afwDetect.FootprintSet(detectedImage, afwDetect.Threshold(0.5))
        fp_ctrl = afwDetect.FootprintControl(True, True)
        fps = afwDetect.FootprintSet(fps, self.config.footprintGrowValue, fp_ctrl)

        sources = afwTable.SourceCatalog(table)
        fps.makeSources(sources)

        self.measurement.run(measCat=sources, exposure=exposure, exposureId=exposureIdInfo.expId)
        self.catalogCalculation.run(sources)
        
        self.display("measure", exposure=exposure, sourceCat=sources)

        return pipeBase.Struct(sourceCat=sources) 

    def display(self, itemName, exposure, sourceCat=None):
        """Display exposure and sources on next frame, if display of itemName has been requested
        @param[in] itemName  name of item in debugInfo
        @param[in] exposure  exposure to display
        @param[in] sourceCat  source catalog to display
        """
        val = getDebugFrame(self._display, itemName)
        if not val:
            return

        displayAstrometry(exposure=exposure, sourceCat=sourceCat, frame=self._frame, pause=False)
        self._frame += 1

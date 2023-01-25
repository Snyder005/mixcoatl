import numpy as np
from collections import defaultdict

from lsst.utils.timer import timeMethod
from lsstDebug import getDebugFrame
from lsst.afw.detection import FootprintSet, Threshold
from lsst.afw.display import getDisplay
from lsst.cp.pipe.utils import ddict2dict
import lsst.afw.image as afwImage
from lsst.ip.isr import CrosstalkCalib

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field, ChoiceField, ListField, ConfigurableField

import mixcoatl.crosstalk as mixCrosstalk
from mixcoatl.detectStreaks import DetectStreaksTask
from mixcoatl.detectSpots import DetectSpotsTask
from mixcoatl.crosstalk import CrosstalkModelFitTask

class CrosstalkTaskConnections(pipeBase.PipelineTaskConnections,
                               dimensions=("instrument", "exposure", "detector")):

    inputExp = cT.Input(
        name="crosstalkInputs",
        doc="Input post-ISR processed exposure to measure crosstalk from.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=False,
    )
    rawExp = cT.Input(
        name="raw",
        doc="Input raw exposure to measure noise covariance from.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=False,
    )
    outputLinearCoefficients = cT.Output(
        name="crosstalkLinearCoefficients",
        doc="Crosstalk linear coefficients from model fit.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )
    outputNonLinearCoefficients = cT.Output(
        name="crosstalkNonLinearCoefficients",
        doc="Crosstalk first-order nonlinear coefficients from model fit.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )
    outputSignals = cT.Output(
        name="crosstalkSignals",
        doc="Crosstalk source signal from model fit.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector")
    )
    outputRatios = cT.Output(
        name="crosstalkRatios",
        doc="Extracted crosstalk pixel ratios.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )
    outputFluxes = cT.Output(
        name="crosstalkFluxes",
        doc="Source pixel fluxes used in ratios.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )
    outputZOffsets = cT.Output(
        name="crosstalkBackgroundZOffsets",
        doc="Z offset parameters used in background model.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )
    outputYTilts = cT.Output(
        name="crosstalkBackgroundYTilts",
        doc="Y tilt parameters used in background model.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )
    outputXTilts = cT.Output(
        name="crosstalkBackgroundXTilts",
        doc="X tilt parameters used in background model.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )
    outputCoefficientErrors = cT.Output(
        name="crosstalkCoefficientErrors",
        doc="Standard error of the crosstalk coefficients.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
        )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.correctNoiseCovariance is not True:
            self.prerequisiteInputs.discard("rawExp")

class CrosstalkTaskConfig(pipeBase.PipelineTaskConfig,
                          pipelineConnections=CrosstalkSatelliteConnections):
    """Configuration for the measurement of pixel ratios.
    """
    threshold = Field(
        dtype=float,
        default=10000.,
        doc="Minimum level of source pixels for which to measure ratios."
    )
    maskWidth = Field(
        dtype=float,
        default=80.,
        doc="Width of satellite streak mask."
    )
    ignoreSaturatedPixels = Field(
        dtype=bool,
        default=False,
        doc="Should saturated pixels be ignored?"
    )
    badMask = ListField(
        dtype=str,
        default=["BAD", "INTRP"],
        doc="Mask planes to ignore when identifying source pixels."
    )
    isTrimmed = Field(
        dtype=bool,
        default=True,
        doc="Is the input exposure trimmed?"
    )
    sourceType = ChoiceField(
        dtype=str,
        default="streak",
        doc="Type of crosstalk source.",
        allowed={
            "streak" : "Streak as a crosstalk source.",
            "spot" : "Large spot as a crosstalk source."
        }
    )
    detectStreaks = ConfigurableField(
        target=DetectStreaksTask,
        doc="Detect streaks as crosstalk sources."
    )
    detectSpots = ConfigurableField(
        target=DetectSpotsTasks,
        doc="Detect spots as crosstalk sources."
    )
    crosstalkSolve = ConfigurableField(
        target=CrosstalkModelFitTask,
        doc="Solve for crosstalk."
    )

    def validate(self):
        super().validate()

        # Ensure the handling of the SAT mask plane is consistent
        # with the ignoreSaturatedPixels value.
        if self.ignoreSaturatedPixels:
            if 'SAT' not in self.badMask:
                self.badMask.append('SAT')
        else:
            if 'SAT' in self.badMask:
                self.badMask = [mask for mask in self.badMask if mask != 'SAT']

class CrosstalkTask(pipeBase.PipelineTask):
    """Task to measure pixel ratios to find crosstalk.
    """
    ConfigClass = CrosstalkTaskConfig
    _DefaultName = 'cpCrosstalk'

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.makeSubtask('detectStreaks')
        self.makeSubtask('detectSpots')
        self.makeSubtask('crosstalkSolve')

    @timeMethod
    def run(self, inputExp, rawExp=None, sourceExps=[]):

        outputLinearCoefficients = defaultdict(lambda: defaultdict(dict))
        outputNonLinearCoefficients = defaultdict(lambda: defaultdict(dict))
        outputSignals = defaultdict(lambda: defaultdict(dict))
        outputRatios = defaultdict(lambda: defaultdict(dict))
        outputFluxes = defaultdict(lambda: defaultdict(dict))
        outputZOffsets = defaultdict(lambda: defaultdict(dict))
        outputYTilts = defaultdict(lambda: defaultdict(dict))
        outputXTilts = defaultdict(lambda: defaultdict(dict))
        outputCoefficientErrors = defaultdict(lambda: defaultdict(dict))

        threshold = self.config.threshold
        badPixels = list(self.config.badMask)

        targetDetector = inputExp.getDetector()
        targetChip = targetDetector.getName()

        sourceExtractExps = [inputExp]
        sourceExtractExps.extend(sourceExps)

        targetIm = inputExp.getMaskedImage()
        FootprintSet(targetIm, Threshold(threshold), "DETECTED")
        detected = targetIm.getMask().getPlaneBitMask("DETECTED")

        for sourceExp in sourceExtractExps:

            sourceDetector = sourceExp.getDetector()
            sourceChip = sourceDetector.getName()
            sourceIm = sourceExp.getMaskedImage()
            bad = sourceIm.getMask().getPlaneBitMask(badPixels)
            self.log.info("Measuring crosstalk from source: %s", sourceChip)

            if sourceExp != inputExp:
                FootprintSet(sourceIm, Threshold(threshold), "DETECTED")
                detected = sourceIm.getMask().getPlaneBitMask("DETECTED")            

            linearCoefficientDict = defaultdict(lambda: defaultdict(list))
            nonLinearCoefficientDict = defaultdict(lambda: defaultdict(list))
            ratioDict = defaultdict(lambda: defaultdict(list))
            zoffsetDict = defaultdict(lambda: defaultdict(list))
            ytiltDict = defaultdict(lambda: defaultdict(list))
            xtiltDict = defaultdict(lambda: defaultdict(list))
            coefficientErrorDict = defaultdict(lambda: defaultdict(list))
            extractedCount = 0

            for sourceAmp in sourceDetector:
                sourceAmpName = sourceAmp.getName()
                sourceAmpBBox = sourceAmp.getBBox() if self.config.isTrimmed else sourceAmp.getRawDataBBox()
                sourceAmpImage = sourceIm[sourceAmpBBox]
                sourceAmpMask = sourceAmpImage.mask.array
                sourceAmpArray = sourceAmpImage.image.array

                ## Find source
                try:
                    if self.config.sourceType == 'streak':
                        detectedSourceResults = self.detectStreaks.run(sourceAmpImage)
                    elif self.config.sourceType == 'spot':
                        detectedSourceResults = self.detectSpots.run(sourceAmpImage)
                except RuntimeError:
                    continue

                sourceMask = detectedSourceResults.sourceMask
                signal = detectedSourceResults.signal
            
                model_select = (sourceMask & (sourceAmpMask & bad == 0)
                                & np.isfinite(sourceAmpImage.image.array))
                ratio_select = (model_select & (sourceAmpMask & detected > 0))
                count = np.sum(ratio_select)
                self.log.debug("  Source amplifier: %s", sourceAmpName)

                outputSignals[sourceChip][sourceAmpName] = [float(signal)]
                outputFluxes[sourceChip][sourceAmpName] = sourceAmpImage.image.array[ratio_select].tolist()

                for targetAmp in targetDetector:
                    # iterate over targetExposure
                    targetAmpName = targetAmp.getName()
                    if sourceAmpName == targetAmpName and sourceChip == targetChip:
                        linearCoefficientDict[sourceAmpName][targetAmpName] = []
                        nonLinearCoefficientDict[sourceAmpName][targetAmpName] = []
                        ratioDict[sourceAmpName][targetAmpName] = []
                        zoffsetDict[targetAmpName][sourceAmpName] = []
                        ytiltDict[targetAmpName][sourceAmpName] = []
                        xtiltDict[targetAmpName][sourceAmpName] = []
                        coefficientErrorDict[targetAmpName][sourceAmpName] = []
                        continue
                    self.log.debug("    Target amplifier: %s", targetAmpName)

                    ## Correct noise covariance
                    if rawExp:
                        covariance = mixCrosstalk.calculate_covariance(rawExp, sourceAmp, targetAmp)
                    else:
                        noise = np.asarray([[sourceAmp.getReadNoise()/sourceAmp.getGain(), 0.],
                                            [0., targetAmp.getReadNoise()/targetAmp.getGain()]])
                        covariance = np.square(noise)

                    ## Perform model fit
                    targetAmpImage = CrosstalkCalib.extractAmp(targetIm, targetAmp, sourceAmp,
                                                               isTrimmed=self.config.isTrimmed)
                    targetAmpArray = targetAmpImage.image.array
                    results = self.crosstalkSolve.run(sourceAmpArray, targetAmpArray, model_select, 
                                                      covariance=covariance, seed=189)

                    ## Calculate background-subtracted ratios
                    bg = results.background
                    ratios = (targetAmpArray-bg)[ratio_select]/sourceAmpArray[ratio_select]

                    linearCoefficientDict[targetAmpName][sourceAmpName] = [float(results.coefficient)]
                    nonLinearCoefficientDict[targetAmpName][sourceAmpName] = [float(results.nonLinearCoefficient)]
                    ratioDict[targetAmpName][sourceAmpName] = ratios.tolist()
                    zoffsetDict[targetAmpName][sourceAmpName] = [float(results.backgroundParameters[0])]
                    ytiltDict[targetAmpName][sourceAmpName] = [float(results.backgroundParameters[1])]
                    xtiltDict[targetAmpName][sourceAmpName] = [float(results.backgroundParameters[2])]
                    coefficientErrorDict[targetAmpName][sourceAmpName] = [float(results.coefficientError)]
                    extractedCount += count

            self.log.info("Extracted %d pixels from %s -> %s",
                          extractedCount, sourceChip, targetChip)
            outputLinearCoefficients[targetChip][sourceChip] = linearCoefficientDict
            outputNonLinearCoefficients[targetChip][sourceChip] = nonLinearCoefficientDict
            outputRatios[targetChip][sourceChip] = ratioDict
            outputZOffsets[targetChip][sourceChip] = zoffsetDict
            outputYTilts[targetChip][sourceChip] = ytiltDict
            outputXTilts[targetChip][sourceChip] = xtiltDict
            outputCoefficientErrors[targetChip][sourceChip] = coefficientErrorDict

        return pipeBase.Struct(
            outputLinearCoefficients=ddict2dict(outputLinearCoefficients),
            outputNonLinearCoefficients=ddict2dict(outputNonLinearCoefficients),
            outputSignals=ddict2dict(outputSignals),
            outputRatios=ddict2dict(outputRatios),
            outputFluxes=ddict2dict(outputFluxes),
            outputZOffsets=ddict2dict(outputZOffsets),
            outputYTilts=ddict2dict(outputYTilts),
            outputXTilts=ddict2dict(outputXTilts),
            outputCoefficientErrors=ddict2dict(outputCoefficientErrors)
        )

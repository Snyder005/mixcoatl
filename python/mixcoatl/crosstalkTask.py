import numpy as np
import copy
from astropy.stats import median_absolute_deviation, sigma_clipped_stats
from collections import defaultdict
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import canny

from lsst.utils.timer import timeMethod
from lsstDebug import getDebugFrame
from lsst.afw.detection import FootprintSet, Threshold
from lsst.afw.display import getDisplay
from lsst.cp.pipe.utils import ddict2dict
import lsst.afw.image as afwImage
from lsst.ip.isr import CrosstalkCalib
import lsst.kht

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field, ListField, ConfigurableField

import mixcoatl.crosstalk as mixCrosstalk

class CrosstalkColumnConnections(pipeBase.PipelineTaskConnections,
                                 dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="crosstalkInputs",
        doc="Input post-ISR processed exposure to measure crosstalk from.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=False,
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
    outputRatioErrors = cT.Output(
        name="crosstalkRatioErrors",
        doc="Parameter error on extracted crosstalk pixel ratios.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

class CrosstalkColumnConfig(pipeBase.PipelineTaskConfig,
                            pipelineConnections=CrosstalkColumnConnections):
    """Configuration for the measurement of pixel ratios.
    """
    threshold = Field(
        dtype=float,
        default=30000,
        doc="Minimum level of source pixels for which to measure crosstalk."
    )
    maskLengthX = Field(
        dtype=float,
        default=10.,
        doc="Length of rectangular mask in x-direction."
    )
    maskLengthY = Field(
        dtype=float,
        default=2000.,
        doc="Length of postage stamp mask in y-direction."
    )
    ignoreSaturatedPixels = Field(
        dtype=bool,
        default=True,
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

class CrosstalkColumnTask(pipeBase.PipelineTask,
                          pipeBase.CmdLineTask):
    """Task to measure pixel ratios to find crosstalk.
    """
    ConfigClass = CrosstalkColumnConfig
    _DefaultName = 'cpCrosstalkColumn'

    @timeMethod
    def run(self, inputExp):

        ## run() method
        outputRatios = defaultdict(lambda: defaultdict(dict))
        outputFluxes = defaultdict(lambda: defaultdict(dict))
        outputZOffsets = defaultdict(lambda: defaultdict(dict))
        outputYTilts = defaultdict(lambda: defaultdict(dict))
        outputXTilts = defaultdict(lambda: defaultdict(dict))
        outputRatioErrors = defaultdict(lambda: defaultdict(dict))

        badPixels = list(self.config.badMask)

        targetDetector = inputExp.getDetector()
        targetChip = targetDetector.getName()
        targetIm = inputExp.getMaskedImage()

        ## loop on sourceExp would go here
        sourceExp = inputExp ## Ignore other exposures for now
        sourceDetector = sourceExp.getDetector()
        sourceChip = sourceDetector.getName()
        sourceIm = sourceExp.getMaskedImage()
        bad = sourceIm.getMask().getPlaneBitMask(badPixels)
        self.log.info("Measuring crosstalk from source: %s", sourceChip)

        ratioDict = defaultdict(lambda: defaultdict(list))
        zoffsetDict = defaultdict(lambda: defaultdict(list))
        ytiltDict = defaultdict(lambda: defaultdict(list))
        xtiltDict = defaultdict(lambda: defaultdict(list))
        ratioErrorDict = defaultdict(lambda: defaultdict(list))
        extractedCount = 0

        for sourceAmp in sourceDetector:
            sourceAmpName = sourceAmp.getName()
            sourceAmpImage = sourceIm[sourceAmp.getBBox()]
            sourceMask = sourceAmpImage.mask.array
            sourceAmpArray = sourceAmpImage.image.array

            columns = mixCrosstalk.find_bright_columns(sourceAmpArray, self.config.threshold)
            if len(columns) == 0: continue
            select = mixCrosstalk.rectangular_mask(sourceAmpArray, 1000, columns[0],
                                                   ly=self.config.maskLengthY, lx=self.config.maskLengthX)
            signal = np.median(sourceAmpArray[:, columns[0]])
            self.log.debug("  Source amplifier: %s", sourceAmpName)

            brightColumnArray = np.zeros(sourceAmpArray.shape)
            brightColumnArray[:, columns[0]] = signal
            outputFluxes[sourceChip][sourceAmpName] = [float(signal)]

            for targetAmp in targetDetector:
                # iterate over targetExposure
                targetAmpName = targetAmp.getName()
                if sourceAmpName == targetAmpName and sourceChip == targetChip:
                    ratioDict[sourceAmpName][targetAmpName] = []
                    zoffsetDict[targetAmpName][sourceAmpName] = []
                    ytiltDict[targetAmpName][sourceAmpName] = []
                    xtiltDict[targetAmpName][sourceAmpName] = []
                    ratioErrorDict[targetAmpName][sourceAmpName] = []
                    continue
                self.log.debug("    Target amplifier: %s", targetAmpName)

                noise = np.asarray([[0., 0.],
                                    [0., targetAmp.getReadNoise()/targetAmp.getGain()]])
                covariance = np.square(noise)

                targetAmpImage = CrosstalkCalib.extractAmp(targetIm, targetAmp, sourceAmp,
                                                           isTrimmed=self.config.isTrimmed)
                targetAmpArray = targetAmpImage.image.array
                results = mixCrosstalk.crosstalk_fit(brightColumnArray, targetAmpArray, select, 
                                                     covariance=covariance,
                                                     correct_covariance=False, 
                                                     seed=189)

                ratioDict[targetAmpName][sourceAmpName] = [float(results[0])]
                zoffsetDict[targetAmpName][sourceAmpName] = [float(results[1])]
                ytiltDict[targetAmpName][sourceAmpName] = [float(results[2])]
                xtiltDict[targetAmpName][sourceAmpName] = [float(results[3])]
                ratioErrorDict[targetAmpName][sourceAmpName] = [float(results[4])]
                extractedCount += 1

        self.log.info("Extracted %d pixels from %s -> %s",
                      extractedCount, sourceChip, targetChip)
        outputRatios[targetChip][sourceChip] = ratioDict
        outputZOffsets[targetChip][sourceChip] = zoffsetDict
        outputYTilts[targetChip][sourceChip] = ytiltDict
        outputXTilts[targetChip][sourceChip] = xtiltDict
        outputRatioErrors[targetChip][sourceChip] = ratioErrorDict
        
        return pipeBase.Struct(
            outputRatios=ddict2dict(outputRatios),
            outputFluxes=ddict2dict(outputFluxes),
            outputZOffsets=ddict2dict(outputZOffsets),
            outputYTilts=ddict2dict(outputYTilts),
            outputXTilts=ddict2dict(outputXTilts),
            outputRatioErrors=ddict2dict(outputRatioErrors)
        )

class CrosstalkSatelliteConnections(pipeBase.PipelineTaskConnections,
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
    outputCoefficients = cT.Output(
        name="crosstalkCoefficients",
        doc="Crosstalk coefficients from model fit.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
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
    outputRatioErrors = cT.Output(
        name="crosstalkRatioErrors",
        doc="Standard error of the crosstalk pixel ratios.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.correctNoiseCovariance is not True:
            self.prerequisiteInputs.discard("rawExp")

class CrosstalkSatelliteConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=CrosstalkSatelliteConnections):
    """Configuration for the measurement of pixel ratios.
    """
    correctNoiseCovariance = Field(
        dtype=bool,
        default=False,
        doc="Correct the effect of correlated read noise."
    )
    thresholdLow = Field(
        dtype=float,
        default=1,
        doc="Low threshold for Canny edge detection."
    )
    thresholdHigh = Field(
        dtype=float,
        default=15,
        doc="High threshold for Canny edge detection."
    )
    minimumKernelHeight = Field(
        doc="Minimum height of the streak-finding kernel relative to the tallest kernel",
        dtype=float,
        default=0.0,
    )
    absMinimumKernelHeight = Field(
        doc="Minimum absolute height of the streak-finding kernel",
        dtype=float,
        default=5,
    )
    clusterMinimumSize = Field(
        doc="Minimum size in pixels of detected clusters",
        dtype=int,
        default=50,
    )
    clusterMinimumDeviation = Field(
        doc="Allowed deviation (in pixels) from a straight line for a detected "
            "line",
        dtype=int,
        default=2,
    )
    delta = Field(
        doc="Stepsize in angle-radius parameter space",
        dtype=float,
        default=0.2,
    )
    nSigma = Field(
        doc="Number of sigmas from center of kernel to include in voting "
            "procedure",
        dtype=float,
        default=2,
    )
    maskWidth = Field(
        dtype=float,
        default=40.,
        doc="One-sided width of satellite streak mask."
    )
    cannySigma = Field(
        dtype=float,
        doc="Gaussian smoothing sigma for Canny edge detection.", 
        default=15.
    )
    ignoreSaturatedPixels = Field(
        dtype=bool,
        default=True,
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

class CrosstalkSatelliteTask(pipeBase.PipelineTask,
                             pipeBase.CmdLineTask):
    """Task to measure pixel ratios to find crosstalk.
    """
    ConfigClass = CrosstalkSatelliteConfig
    _DefaultName = 'cpCrosstalkSatellite'

    @timeMethod
    def run(self, inputExp, rawExp):

        outputCoefficients = defaultdict(lambda: defaultdict(dict))
        outputRatios = defaultdict(lambda: defaultdict(dict))
        outputFluxes = defaultdict(lambda: defaultdict(dict))
        outputZOffsets = defaultdict(lambda: defaultdict(dict))
        outputYTilts = defaultdict(lambda: defaultdict(dict))
        outputXTilts = defaultdict(lambda: defaultdict(dict))
        outputCoefficientErrors = defaultdict(lambda: defaultdict(dict))
        outputRatioErrors = defaultdict(lambda: defaultdict(dict))

        badPixels = list(self.config.badMask)

        targetDetector = inputExp.getDetector()
        targetChip = targetDetector.getName()
        targetIm = inputExp.getMaskedImage()

        ## loop on sourceExp would go here
        sourceExp = inputExp ## Ignore other exposures for now
        sourceDetector = sourceExp.getDetector()
        sourceChip = sourceDetector.getName()
        sourceIm = sourceExp.getMaskedImage()
        bad = sourceIm.getMask().getPlaneBitMask(badPixels)
        self.log.info("Measuring crosstalk from source: %s", sourceChip)

        coefficientDict = defaultdict(lambda: defaultdict(list))
        ratioDict = defaultdict(lambda: defaultdict(list))
        zoffsetDict = defaultdict(lambda: defaultdict(list))
        ytiltDict = defaultdict(lambda: defaultdict(list))
        xtiltDict = defaultdict(lambda: defaultdict(list))
        coefficientErrorDict = defaultdict(lambda: defaultdict(list))
        ratioErrorDict = defaultdict(lambda: defaultdict(list))
        extractedCount = 0

        for sourceAmp in sourceDetector:
            sourceAmpName = sourceAmp.getName()
            sourceAmpImage = sourceIm[sourceAmp.getBBox()]
            sourceMask = sourceAmpImage.mask.array
            sourceAmpArray = sourceAmpImage.image.array

            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 1000)
            edges = canny(sourceAmpArray, sigma=self.config.cannySigma, 
                                  low_threshold=self.config.thresholdLow, 
                                  high_threshold=self.config.thresholdHigh)
            lines = lsst.kht.find_lines(edges, self.config.clusterMinimumSize, 
                                        self.config.clusterMinimumDeviation, 
                                        self.config.delta, self.config.minimumKernelHeight, 
                                        self.config.nSigma, self.config.absMinimumKernelHeight)

            angle = np.asarray([l[1]*np.pi/180. for l in lines])
            dist = np.asarray([l[0] for l in lines])
            dist = -(dist + 1000*np.sin(angle) + 256*np.cos(angle))
            angle = angle - np.pi            

            if len(angle) != 2:
                continue

            mean_angle = np.mean(angle)
            mean_dist = np.mean(dist)
            select = mixCrosstalk.satellite_mask(sourceAmpArray, mean_angle, mean_dist, 
                                                 width=self.config.maskWidth)
            
            ## Calculate median signal of satellite
            signal_select = mixCrosstalk.satellite_mask(sourceAmpArray, mean_angle, mean_dist,
                                                        width=0.5)
            signal = sigma_clipped_stats(sourceAmpArray[signal_select], cenfunc='median',
                                         stdfunc=median_absolute_deviation)[1]
            self.log.debug("  Source amplifier: %s", sourceAmpName)

            outputFluxes[sourceChip][sourceAmpName] = [float(signal)]

            for targetAmp in targetDetector:
                # iterate over targetExposure
                targetAmpName = targetAmp.getName()
                if sourceAmpName == targetAmpName and sourceChip == targetChip:
                    coefficientDict[sourceAmpName][targetAmpName] = []
                    ratioDict[sourceAmpName][targetAmpName] = []
                    zoffsetDict[targetAmpName][sourceAmpName] = []
                    ytiltDict[targetAmpName][sourceAmpName] = []
                    xtiltDict[targetAmpName][sourceAmpName] = []
                    coefficientErrorDict[targetAmpName][sourceAmpName] = []
                    ratioErrorDict[targetAmpName][sourceAmpName] = []
                    continue
                self.log.debug("    Target amplifier: %s", targetAmpName)

                if self.config.correctNoiseCovariance:
                    covariance = mixCrosstalk.calculate_covariance(rawExp, sourceAmp, targetAmp)
                else:
                    noise = np.asarray([[sourceAmp.getReadNoise()/sourceAmp.getGain(), 0.],
                                        [0., targetAmp.getReadNoise()/targetAmp.getGain()]])
                    covariance = np.square(noise)

                targetAmpImage = CrosstalkCalib.extractAmp(targetIm, targetAmp, sourceAmp,
                                                           isTrimmed=self.config.isTrimmed)
                targetAmpArray = targetAmpImage.image.array
                ratios = targetAmpArray[signal_select]/signal
                _, ratio, ratio_stdev = sigma_clipped_stats(ratios, cenfunc='median',
                                                            stdfunc=median_absolute_deviation)
                ratio_error = ratio_stdev/np.sqrt(len(ratios))
                results = mixCrosstalk.crosstalk_fit(sourceAmpArray, targetAmpArray, select, 
                                                     covariance=covariance, 
                                                     correct_covariance=self.config.correctNoiseCovariance, 
                                                     seed=189)

                coefficientDict[targetAmpName][sourceAmpName] = [float(results[0])]
                ratioDict[targetAmpName][sourceAmpName] = [float(ratio)]
                zoffsetDict[targetAmpName][sourceAmpName] = [float(results[1])]
                ytiltDict[targetAmpName][sourceAmpName] = [float(results[2])]
                xtiltDict[targetAmpName][sourceAmpName] = [float(results[3])]
                coefficientErrorDict[targetAmpName][sourceAmpName] = [float(results[4])]
                ratioErrorDict[targetAmpName][sourceAmpName] = [float(ratio_error)]
                extractedCount += 1

        self.log.info("Extracted %d pixels from %s -> %s",
                      extractedCount, sourceChip, targetChip)
        outputCoefficients[targetChip][sourceChip] = coefficientDict
        outputRatios[targetChip][sourceChip] = ratioDict
        outputZOffsets[targetChip][sourceChip] = zoffsetDict
        outputYTilts[targetChip][sourceChip] = ytiltDict
        outputXTilts[targetChip][sourceChip] = xtiltDict
        outputCoefficientErrors[targetChip][sourceChip] = coefficientErrorDict
        outputRatioErrors[targetChip][sourceChip] = ratioErrorDict

        return pipeBase.Struct(
            outputCoefficients=ddict2dict(outputCoefficients),
            outputRatios=ddict2dict(outputRatios),
            outputFluxes=ddict2dict(outputFluxes),
            outputZOffsets=ddict2dict(outputZOffsets),
            outputYTilts=ddict2dict(outputYTilts),
            outputXTilts=ddict2dict(outputXTilts),
            outputCoefficientErrors=ddict2dict(outputCoefficientErrors),
            outputRatioErrors=ddict2dict(outputRatioErrors)
        )

class CrosstalkSpotConnections(pipeBase.PipelineTaskConnections,
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
    outputCoefficients = cT.Output(
        name="crosstalkCoefficients",
        doc="Crosstalk coefficients from model fit.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
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
    outputRatioErrors = cT.Output(
        name="crosstalkRatioErrors",
        doc="Standard error of the crosstalk pixel ratios.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
        )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.correctNoiseCovariance is not True:
            self.prerequisiteInputs.discard("rawExp")

class CrosstalkSpotConfig(pipeBase.PipelineTaskConfig,
                          pipelineConnections=CrosstalkSpotConnections):

    correctNoiseCovariance = Field(
        dtype=bool,
        default=False,
        doc="Correct the effect of correlated read noise."
    )
    threshold = Field(
        dtype=float,
        default=30000.,
        doc="Minimum level of source pixels for which to measure crosstalk."
    )
    maskRadius = Field(
        dtype=float,
        default=10.,
        doc="Radius of circular mask for source signal calculation."
    )
    maskLength = Field(
        dtype=float,
        default=250.,
        doc="Length of side of square mask."
    )
    doAnnularCutout = Field(
        dtype=bool,
        default=False,
        doc="Mask an annular cutout of the square mask."
    )
    annulusInnerRadius = Field(
        dtype=float,
        default=40.,
        doc="Inner radius of annulur mask used for cutout."
    )
    annulusOuterRadius = Field(
        dtype=float,
        default=100.,
        doc="Outer radius of annulur mask used for cutout."
    )
    ignoreSaturatedPixels = Field(
        dtype=bool,
        default=True,
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

class CrosstalkSpotTask(pipeBase.PipelineTask, 
                        pipeBase.CmdLineTask):

    ConfigClass = CrosstalkSpotConfig
    _DefaultName = 'cpCrosstalkSpot'

    @timeMethod
    def run(self, inputExp, rawExp):

        outputCoefficients = defaultdict(lambda: defaultdict(dict))
        outputRatios = defaultdict(lambda: defaultdict(dict))
        outputFluxes = defaultdict(lambda: defaultdict(dict))
        outputZOffsets = defaultdict(lambda: defaultdict(dict))
        outputYTilts = defaultdict(lambda: defaultdict(dict))
        outputXTilts = defaultdict(lambda: defaultdict(dict))
        outputCoefficientErrors = defaultdict(lambda: defaultdict(dict))
        outputRatioErrors = defaultdict(lambda: defaultdict(dict))

        badPixels = list(self.config.badMask)

        targetDetector = inputExp.getDetector()
        targetChip = targetDetector.getName()
        targetIm = inputExp.getMaskedImage()

        ## loop on sourceExp would go here
        sourceExp = inputExp ## Ignore other exposures for now
        sourceDetector = sourceExp.getDetector()
        sourceChip = sourceDetector.getName()
        sourceIm = sourceExp.getMaskedImage()
        bad = sourceIm.getMask().getPlaneBitMask(badPixels)
        self.log.info("Measuring crosstalk from source: %s", sourceChip)

        coefficientDict = defaultdict(lambda: defaultdict(list))
        ratioDict = defaultdict(lambda: defaultdict(list))
        zoffsetDict = defaultdict(lambda: defaultdict(list))
        ytiltDict = defaultdict(lambda: defaultdict(list))
        xtiltDict = defaultdict(lambda: defaultdict(list))
        coefficientErrorDict = defaultdict(lambda: defaultdict(list))
        ratioErrorDict = defaultdict(lambda: defaultdict(list))
        extractedCount = 0

        for sourceAmp in sourceDetector:

            sourceAmpName = sourceAmp.getName()
            sourceAmpImage = sourceIm[sourceAmp.getBBox()]
            sourceMask = sourceAmpImage.mask.array
            sourceAmpArray = sourceAmpImage.image.array

            smoothed = gaussian_filter(sourceAmpArray, 20)
            y, x = np.unravel_index(smoothed.argmax(), smoothed.shape)
            select = mixCrosstalk.rectangular_mask(sourceAmpArray, y, x,
                                                   ly=self.config.maskLength, lx=self.config.maskLength)
            if self.config.doAnnularCutout:
                cutout = ~mixCrosstalk.annular_mask(sourceAmpArray, y, x, 
                                                    inner_radius=self.config.annulusInnerRadius,
                                                    outer_radius=self.config.annulusOuterRadius)
                select = select*cutout
            signal_select = mixCrosstalk.circular_mask(sourceAmpArray, y, x, radius=self.config.maskRadius)
            signal = sigma_clipped_stats(sourceAmpArray[signal_select], cenfunc='median', 
                                         stdfunc=median_absolute_deviation)[1]
            if signal < self.config.threshold: continue
            self.log.debug("  Source amplifier: %s", sourceAmpName)

            outputFluxes[sourceChip][sourceAmpName] = [float(signal)]

            for targetAmp in targetDetector:
                # iterate over targetExposure
                targetAmpName = targetAmp.getName()
                if sourceAmpName == targetAmpName and sourceChip == targetChip:
                    coefficientDict[sourceAmpName][targetAmpName] = []
                    ratioDict[sourceAmpName][targetAmpName] = []
                    zoffsetDict[targetAmpName][sourceAmpName] = []
                    ytiltDict[targetAmpName][sourceAmpName] = []
                    xtiltDict[targetAmpName][sourceAmpName] = []
                    coefficientErrorDict[targetAmpName][sourceAmpName] = []
                    ratioErrorDict[targetAmpName][sourceAmpName] = []
                    continue
                self.log.debug("    Target amplifier: %s", targetAmpName)

                if self.config.correctNoiseCovariance:
                    covariance = mixCrosstalk.calculate_covariance(rawExp, sourceAmp, targetAmp)
                else:
                    noise = np.asarray([[sourceAmp.getReadNoise()/sourceAmp.getGain(), 0.],
                                        [0., targetAmp.getReadNoise()/targetAmp.getGain()]])
                    covariance = np.square(noise)

                targetAmpImage = CrosstalkCalib.extractAmp(targetIm, targetAmp, sourceAmp,
                                                           isTrimmed=self.config.isTrimmed)
                targetAmpArray = targetAmpImage.image.array
                ratios = targetAmpArray[signal_select]/signal
                _, ratio, ratio_stdev = sigma_clipped_stats(ratios, cenfunc='median', 
                                                           stdfunc=median_absolute_deviation)
                ratio_error = ratio_stdev/np.sqrt(len(ratios))
                results = mixCrosstalk.crosstalk_fit(sourceAmpArray, targetAmpArray, select, 
                                                     covariance=covariance, 
                                                     correct_covariance=self.config.correctNoiseCovariance, 
                                                     seed=189)

                coefficientDict[targetAmpName][sourceAmpName] = [float(results[0])]
                ratioDict[targetAmpName][sourceAmpName] = [float(ratio)]
                zoffsetDict[targetAmpName][sourceAmpName] = [float(results[1])]
                ytiltDict[targetAmpName][sourceAmpName] = [float(results[2])]
                xtiltDict[targetAmpName][sourceAmpName] = [float(results[3])]
                coefficientErrorDict[targetAmpName][sourceAmpName] = [float(results[4])]
                ratioErrorDict[targetAmpName][sourceAmpName] = [float(ratio_error)]
                extractedCount += 1

        self.log.info("Extracted %d pixels from %s -> %s",
                      extractedCount, sourceChip, targetChip)
        outputCoefficients[targetChip][sourceChip] = coefficientDict
        outputRatios[targetChip][sourceChip] = ratioDict
        outputZOffsets[targetChip][sourceChip] = zoffsetDict
        outputYTilts[targetChip][sourceChip] = ytiltDict
        outputXTilts[targetChip][sourceChip] = xtiltDict
        outputCoefficientErrors[targetChip][sourceChip] = coefficientErrorDict
        outputRatioErrors[targetChip][sourceChip] = ratioErrorDict

        return pipeBase.Struct(
            outputCoefficients=ddict2dict(outputCoefficients),
            outputRatios=ddict2dict(outputRatios),
            outputFluxes=ddict2dict(outputFluxes),
            outputZOffsets=ddict2dict(outputZOffsets),
            outputYTilts=ddict2dict(outputYTilts),
            outputXTilts=ddict2dict(outputXTilts),
            outputCoefficientErrors=ddict2dict(outputCoefficientErrors),
            outputRatioErrors=ddict2dict(outputRatioErrors)
        )


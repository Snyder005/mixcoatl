"""DM Crosstalk Tasks.

To Do: 
    1. Determine method to restrict satellite crosstalk measurement to single side of sensor.
    2. Clean up pipetask connection for rawExp.
"""
import numpy as np
import copy
from collections import defaultdict
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks

from lsstDebug import getDebugFrame
from lsst.afw.detection import FootprintSet, Threshold
from lsst.afw.display import getDisplay
from lsst.cp.pipe.utils import ddict2dict
import lsst.afw.image as afwImage
from lsst.ip.isr import CrosstalkCalib

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field, ListField, ConfigurableField

from mixcoatl.crosstalk import crosstalk_fit, satellite_mask, rectangular_mask

def find_bright_columns(imarr, threshold):
    
    image = afwImage.ImageF(imarr)
    
    fp_set = FootprintSet(image, Threshold(threshold))    
    columns = dict([(x, []) for x in range(0, image.getWidth())])
    for footprint in fp_set.getFootprints():
        for span in footprint.getSpans():
            y = span.getY()
            for x in range(span.getX0(), span.getX1()+1):
                columns[x].append(y)
                
    bright_cols = []
    x0 = image.getX0()
    y0 = image.getY0()
    for x in columns:
        if bad_column(columns[x], 20):
            bright_cols.append(x - x0)
    #
    # Sort the output.
    #
    bright_cols.sort()
    
    return bright_cols

def bad_column(column_indices, threshold):
    """
    Count the sizes of contiguous sequences of masked pixels and
    return True if the length of any sequence exceeds the threshold
    number.
    """
    if len(column_indices) < threshold:
        # There are not enough masked pixels to mark this as a bad
        # column.
        return False
    # Fill an array with zeros, then fill with ones at mask locations.
    column = np.zeros(max(column_indices) + 1)
    column[(column_indices,)] = 1
    # Count pixels in contiguous masked sequences.
    masked_pixel_count = []
    last = 0
    for value in column:
        if value != 0 and last == 0:
            masked_pixel_count.append(1)
        elif value != 0 and last != 0:
            masked_pixel_count[-1] += 1
        last = value
    if len(masked_pixel_count) > 0 and max(masked_pixel_count) >= threshold:
        return True
    return False

class CrosstalkColumnConnections(pipeBase.PipelineTaskConnections,
                                 dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="crosstalkInputs",
        doc="Input post-ISR processed exposure to measure crosstalk from.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=False,
    )
    rawExp = cT.Input(
        name="rawInputs",
        doc="Input raw exposure to measure noise covariance from.",
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

    def __init__(self, *, config=None):
        super().__init__(config=config)

class CrosstalkColumnConfig(pipeBase.PipelineTaskConfig,
                            pipelineConnections=CrosstalkColumnConnections):
    """Configuration for the measurement of pixel ratios.
    """
    correctNoiseCovariance = Field(
        dtype=bool,
        default=False,
        doc="Correct the effect of correlated read noise."
    )
    threshold = Field(
        dtype=float,
        default=30000,
        doc="Minimum level of source pixels for which to measure crosstalk."
    )
    maskLengthX = Field(
        dtype=int,
        default=100,
        doc="Length of rectangular mask in x-direction."
    )
    maskLengthY = Field(
        dtype=int,
        default=2000,
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

    @pipeBase.timeMethod
    def run(self, inputExp, rawExp):

        ## run() method
        outputRatios = defaultdict(lambda: defaultdict(dict))
        outputFluxes = defaultdict(lambda: defaultdict(dict))
        outputZOffsets = defaultdict(lambda: defaultdict(dict))
        outputYTilts = defaultdict(lambda: defaultdict(dict))
        outputXTilts = defaultdict(lambda: defaultdict(dict))

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

        extractedCount = 0

        for sourceAmp in sourceDetector:
            sourceAmpName = sourceAmp.getName()
            sourceAmpImage = sourceIm[sourceAmp.getBBox()]
            sourceMask = sourceAmpImage.mask.array
            sourceAmpArray = sourceAmpImage.image.array

            columns = find_bright_columns(sourceAmpArray, self.config.threshold)
            if len(columns) == 0: continue
            select = rectangular_mask(sourceAmpArray, 1000, columns[0], ly=self.config.maskLengthY, 
                                      lx=self.config.maskLengthX)
            signal = np.mean(sourceAmpArray[:, columns[0]])
            self.log.debug("  Source amplifier: %s", sourceAmpName)

            outputFluxes[sourceChip][sourceAmpName] = [float(signal)]

            for targetAmp in targetDetector:
                # iterate over targetExposure
                targetAmpName = targetAmp.getName()
                if sourceAmpName == targetAmpName and sourceChip == targetChip:
                    ratioDict[sourceAmpName][targetAmpName] = []
                    continue
                self.log.debug("    Target amplifier: %s", targetAmpName)

                if self.config.correctNoiseCovariance:
                    covariance = self.calculate_covariance(rawExp, sourceAmp, targetAmp)
                else:
                    covariance = np.asarray([[7.0, 0.0], [0.0, 7.0]])

                targetAmpImage = CrosstalkCalib.extractAmp(targetIm.image,
                                                           targetAmp, sourceAmp,
                                                           isTrimmed=self.config.isTrimmed)
                targetAmpArray = targetAmpImage.array
                results = crosstalk_fit(sourceAmpArray, targetAmpArray, select, covariance=covariance, 
                                        correct_covariance=self.config.correctNoiseCovariance, seed=189)

                ratioDict[targetAmpName][sourceAmpName] = [float(results[0])]
                zoffsetDict[targetAmpName][sourceAmpName] = [float(results[1])]
                ytiltDict[targetAmpName][sourceAmpName] = [float(results[2])]
                xtiltDict[targetAmpName][sourceAmpName] = [float(results[3])]
                extractedCount += 1

        self.log.info("Extracted %d pixels from %s -> %s",
                      extractedCount, sourceChip, targetChip)
        outputRatios[targetChip][sourceChip] = ratioDict
        outputZOffsets[targetChip][sourceChip] = zoffsetDict
        outputYTilts[targetChip][sourceChip] = ytiltDict
        outputXTilts[targetChip][sourceChip] = xtiltDict
        
        return pipeBase.Struct(
            outputRatios=ddict2dict(outputRatios),
            outputFluxes=ddict2dict(outputFluxes),
            outputZOffsets=ddict2dict(outputZOffsets),
            outputYTilts=ddict2dict(outputYTilts),
            outputXTilts=ddict2dict(outputXTilts)
        )

    def calculate_covariance(self, exposure, amp1, amp2):

        ccd = exposure.getDetector()

        oscanImage1 = exposure.maskedImage[amp1.getRawHorizontalOverscanBBox()]
        oscanImage2 = exposure.maskedImage[amp2.getRawHorizontalOverscanBBox()]

        arr1 = oscanImage1.getImage().getArray().flatten()
        arr2 = oscanImage2.getImage().getArray().flatten()

        cov = np.cov(np.vstack([arr1, arr2]))

        return cov

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
        name="rawInputs",
        doc="Input raw exposure to measure noise covariance from.",
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

    def __init__(self, *, config=None):
        super().__init__(config=config)

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
    maskWidth = Field(
        dtype=int,
        default=50,
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
    _DefaultName = 'cpSatelliteColumn'

    @pipeBase.timeMethod
    def run(self, inputExp, rawExp):

        ## run() method
        outputRatios = defaultdict(lambda: defaultdict(dict))
        outputFluxes = defaultdict(lambda: defaultdict(dict))
        outputZOffsets = defaultdict(lambda: defaultdict(dict))
        outputYTilts = defaultdict(lambda: defaultdict(dict))
        outputXTilts = defaultdict(lambda: defaultdict(dict))

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

        extractedCount = 0

        for sourceAmp in sourceDetector:
            sourceAmpName = sourceAmp.getName()
            sourceAmpImage = sourceIm[sourceAmp.getBBox()]
            sourceMask = sourceAmpImage.mask.array
            sourceAmpArray = sourceAmpImage.image.array

            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 1000)
            edges = feature.canny(sourceAmpArray, sigma=self.config.cannySigma, 
                                  low_threshold=self.config.thresholdLow, 
                                  high_threshold=self.config.thresholdHigh)
            h, theta, d = hough_line(edges, theta=tested_angles)
            _, angle, dist = hough_line_peaks(h, theta, d)

            if len(angle) != 2:
                continue

            mean_angle = np.mean(angle)
            mean_dist = np.mean(dist)
            select = satellite_mask(sourceAmpArray, mean_angle, mean_dist, width=self.config.maskWidth)
            signal = np.max(aggressor_imarr[select])
            self.log.debug("  Source amplifier: %s", sourceAmpName)

            outputFluxes[sourceChip][sourceAmpName] = [float(signal)]

            for targetAmp in targetDetector:
                # iterate over targetExposure
                targetAmpName = targetAmp.getName()
                if sourceAmpName == targetAmpName and sourceChip == targetChip:
                    ratioDict[sourceAmpName][targetAmpName] = []
                    continue
                self.log.debug("    Target amplifier: %s", targetAmpName)

                if self.config.correctNoiseCovariance:
                    covariance = self.calculate_covariance(rawExp, sourceAmp, targetAmp)
                else:
                    covariance = np.asarray([[7.0, 0.0], [0.0, 7.0]])

                targetAmpImage = CrosstalkCalib.extractAmp(targetIm.image,
                                                           targetAmp, sourceAmp,
                                                           isTrimmed=self.config.isTrimmed)
                targetAmpArray = targetAmpImage.array
                results = crosstalk_fit(sourceAmpArray, targetAmpArray, select, covariance=covariance, 
                                        correct_covariance=self.config.correctNoiseCovariance, seed=189)

                ratioDict[targetAmpName][sourceAmpName] = [float(results[0])]
                zoffsetDict[targetAmpName][sourceAmpName] = [float(results[1])]
                ytiltDict[targetAmpName][sourceAmpName] = [float(results[2])]
                xtiltDict[targetAmpName][sourceAmpName] = [float(results[3])]
                extractedCount += 1

        self.log.info("Extracted %d pixels from %s -> %s",
                      extractedCount, sourceChip, targetChip)
        outputRatios[targetChip][sourceChip] = ratioDict
        outputZOffsets[targetChip][sourceChip] = zoffsetDict
        outputYTilts[targetChip][sourceChip] = ytiltDict
        outputXTilts[targetChip][sourceChip] = xtiltDict
        
        return pipeBase.Struct(
            outputRatios=ddict2dict(outputRatios),
            outputFluxes=ddict2dict(outputFluxes),
            outputZOffsets=ddict2dict(outputZOffsets),
            outputYTilts=ddict2dict(outputYTilts),
            outputXTilts=ddict2dict(outputXTilts)
        )

    def calculate_covariance(self, exposure, amp1, amp2):

        ccd = exposure.getDetector()

        oscanImage1 = exposure.maskedImage[amp1.getRawHorizontalOverscanBBox()]
        oscanImage2 = exposure.maskedImage[amp2.getRawHorizontalOverscanBBox()]

        arr1 = oscanImage1.getImage().getArray().flatten()
        arr2 = oscanImage2.getImage().getArray().flatten()

        cov = np.cov(np.vstack([arr1, arr2]))

        return cov
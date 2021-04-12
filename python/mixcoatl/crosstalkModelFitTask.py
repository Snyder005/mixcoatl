import numpy as np
from collections import defaultdict

from lsst.afw.detection import FootprintSet, Threshold
from lsst.cp.pipe.utils import ddict2dict
import lsst.afw.image as afwImage
from lsst.ip.isr import CrosstalkCalib

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field, ListField

from mixcoatl.crosstalk import crosstalk_fit, rectangular_mask

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

class CrosstalkModelFitConnections(pipeBase.PipelineTaskConnections,
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

    def __init__(self, *, config=None):
        super().__init__(config=config)

class CrosstalkModelFitConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=CrosstalkModelFitConnections):
    """Configuration for the measurement of pixel ratios.
    """
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

class CrosstalkModelFitTask(pipeBase.PipelineTask,
                            pipeBase.CmdLineTask):
    """Task to measure pixel ratios to find crosstalk.
    """
    ConfigClass = CrosstalkModelFitConfig
    _DefaultName = 'cpCrosstalkModelFit'

    def run(self, inputExp):
        
        covariance = [[7.0, 0.0], [0.0, 7.0]] 

        ## run() method
        outputRatios = defaultdict(lambda: defaultdict(dict))
        outputFluxes = defaultdict(lambda: defaultdict(dict))

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

                targetAmpImage = CrosstalkCalib.extractAmp(targetIm.image,
                                                           targetAmp, sourceAmp,
                                                           isTrimmed=self.config.isTrimmed)
                targetAmpArray = targetAmpImage.array
                results = crosstalk_fit(sourceAmpArray, targetAmpArray, select, covariance=covariance)

                ratioDict[targetAmpName][sourceAmpName] = [float(results[0])]
                extractedCount += 1

        self.log.info("Extracted %d pixels from %s -> %s",
                      extractedCount, sourceChip, targetChip)
        outputRatios[targetChip][sourceChip] = ratioDict
        
        return pipeBase.Struct(
            outputRatios=ddict2dict(outputRatios),
            outputFluxes=ddict2dict(outputFluxes)
        )

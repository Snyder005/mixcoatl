"""DM Crosstalk Tasks.

To Do: 
    1. Determine method to get covariance measurement into CrosstalkModelFitTask.
    2. Maybe implement satellite finding into CrosstalkModelFitTask?
    3. Maybe add output files for background model parameters (yslope, xslope, zoffset)?
    2. Improvements to background in the crosstalkExtractTask.
"""
import numpy as np
import copy
from collections import defaultdict

from lsstDebug import getDebugFrame
from lsst.afw.detection import FootprintSet, Threshold
from lsst.afw.display import getDisplay
from lsst.cp.pipe.utils import ddict2dict
import lsst.afw.image as afwImage
from lsst.ip.isr import CrosstalkCalib
from lsst.meas.algorithms.subtractBackground import SubtractBackgroundConfig, SubtractBackgroundTask

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field, ListField, ConfigurableField

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

    @pipebase.timeMethod
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

class CrosstalkExtractConnections(pipeBase.PipelineTaskConnections,
                                  dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="crosstalkInputs",
        doc="Input post-ISR processed exposure to measure crosstalk from.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=False,
    )
    # To Do: Depends on DM-21904.
    sourceExp = cT.Input(
        name="crosstalkSource",
        doc="Post-ISR exposure to measure for inter-chip crosstalk onto inputExp.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
        # lookupFunction=None,
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
    background = cT.Output(
        doc="Output background model.",
        name="icCrosstalkBackground",
        storageClass="Background",
        dimensions=["instrument", "exposure", "detector"],
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        # Discard sourceExp until DM-21904 allows full interchip
        # measurements.
        self.inputs.discard("sourceExp")
        
class CrosstalkExtractConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=CrosstalkExtractConnections):
    """Configuration for the measurement of pixel ratios.
    """
    doMeasureInterchip = Field(
        dtype=bool,
        default=False,
        doc="Measure inter-chip crosstalk as well?",
    )
    threshold = Field(
        dtype=float,
        default=30000,
        doc="Minimum level of source pixels for which to measure crosstalk."
    )
    background = ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Configuration for initial background estimation",
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

class CrosstalkExtractTask(pipeBase.PipelineTask,
                            pipeBase.CmdLineTask):
    """Task to measure pixel ratios to find crosstalk.
    """
    ConfigClass = CrosstalkExtractConfig
    _DefaultName = 'cpCrosstalkExtract'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("background")

    def run(self, inputExp, sourceExps=[]):
        """Measure pixel ratios between amplifiers in inputExp.
        Extract crosstalk ratios between different amplifiers.
        For pixels above ``config.threshold``, we calculate the ratio
        between each background-subtracted target amp and the source
        amp. We return a list of ratios for each pixel for each
        target/source combination, as nested dictionary containing the
        ratio.
        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Input exposure to measure pixel ratios on.
        sourceExp : `list` [`lsst.afw.image.Exposure`], optional
            List of chips to use as sources to measure inter-chip
            crosstalk.
        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:
            ``outputRatios`` : `dict` [`dict` [`dict` [`dict` [`list`]]]]
                 A catalog of ratio lists.  The dictionaries are
                 indexed such that:
                 outputRatios[targetChip][sourceChip][targetAmp][sourceAmp]
                 contains the ratio list for that combination.
            ``outputFluxes`` : `dict` [`dict` [`list`]]
                 A catalog of flux lists.  The dictionaries are
                 indexed such that:
                 outputFluxes[sourceChip][sourceAmp]
                 contains the flux list used in the outputRatios.
        Notes
        -----
        The lsstDebug.Info() method can be rewritten for __name__ =
        `lsst.cp.pipe.measureCrosstalk`, and supports the parameters:
        debug.display['extract'] : `bool`
            Display the exposure under consideration, with the pixels used
            for crosstalk measurement indicated by the DETECTED mask plane.
        debug.display['pixels'] : `bool`
            Display a plot of the ratio calculated for each pixel used in this
            exposure, split by amplifier pairs.  The median value is listed
            for reference.
        """
        outputRatios = defaultdict(lambda: defaultdict(dict))
        outputFluxes = defaultdict(lambda: defaultdict(dict))

        threshold = self.config.threshold
        badPixels = list(self.config.badMask)

        targetDetector = inputExp.getDetector()
        targetChip = targetDetector.getName()

        # Always look at the target chip first, then go to any other supplied exposures.
        sourceExtractExps = [copy.deepcopy(inputExp)]
        sourceExtractExps.extend(sourceExps)

        self.log.info("Measuring full detector background for target: %s", targetChip)
        targetIm = inputExp.getMaskedImage()
        FootprintSet(targetIm, Threshold(threshold), "DETECTED")
        detected = targetIm.getMask().getPlaneBitMask("DETECTED")
        background = self.background.run(inputExp).background
        self.debugView('extract', inputExp)

        for sourceExp in sourceExtractExps:
            sourceDetector = sourceExp.getDetector()
            sourceChip = sourceDetector.getName()
            sourceIm = sourceExp.getMaskedImage()
            bad = sourceIm.getMask().getPlaneBitMask(badPixels)
            self.log.info("Measuring crosstalk from source: %s", sourceChip)

            if sourceExp != inputExp:
                FootprintSet(sourceIm, Threshold(threshold), "DETECTED")
                detected = sourceIm.getMask().getPlaneBitMask("DETECTED")

            # The dictionary of amp-to-amp ratios for this pair of source->target detectors.
            ratioDict = defaultdict(lambda: defaultdict(list))
            extractedCount = 0

            for sourceAmp in sourceDetector:
                sourceAmpName = sourceAmp.getName()
                sourceAmpBBox = sourceAmp.getBBox() if self.config.isTrimmed else sourceAmp.getRawDataBBox()
                sourceAmpImage = sourceIm[sourceAmpBBox]
                sourceMask = sourceAmpImage.mask.array
                select = ((sourceMask & detected > 0)
                          & (sourceMask & bad == 0)
                          & np.isfinite(sourceAmpImage.image.array))
                count = np.sum(select)
                self.log.debug("  Source amplifier: %s", sourceAmpName)

                outputFluxes[sourceChip][sourceAmpName] = sourceAmpImage.image.array[select].tolist()

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
                    ratios = (targetAmpImage.array[select])/sourceAmpImage.image.array[select]
                    ratioDict[targetAmpName][sourceAmpName] = ratios.tolist()
                    extractedCount += count

                    self.debugPixels('pixels',
                                     sourceAmpImage.image.array[select],
                                     targetAmpImage.array[select],
                                     sourceAmpName, targetAmpName)

            self.log.info("Extracted %d pixels from %s -> %s",
                          extractedCount, sourceChip, targetChip)
            outputRatios[targetChip][sourceChip] = ratioDict

        return pipeBase.Struct(
            outputRatios=ddict2dict(outputRatios),
            outputFluxes=ddict2dict(outputFluxes),
            background=background
        )

    def debugView(self, stepname, exposure):
        """Utility function to examine the image being processed.
        Parameters
        ----------
        stepname : `str`
            State of processing to view.
        exposure : `lsst.afw.image.Exposure`
            Exposure to view.
        """
        frame = getDebugFrame(self._display, stepname)
        if frame:
            display = getDisplay(frame)
            display.scale('asinh', 'zscale')
            display.mtv(exposure)

            prompt = "Press Enter to continue: "
            while True:
                ans = input(prompt).lower()
                if ans in ("", "c",):
                    break

    def debugPixels(self, stepname, pixelsIn, pixelsOut, sourceName, targetName):
        """Utility function to examine the CT ratio pixel values.
        Parameters
        ----------
        stepname : `str`
            State of processing to view.
        pixelsIn : `np.ndarray`
            Pixel values from the potential crosstalk source.
        pixelsOut : `np.ndarray`
            Pixel values from the potential crosstalk target.
        sourceName : `str`
            Source amplifier name
        targetName : `str`
            Target amplifier name
        """
        frame = getDebugFrame(self._display, stepname)
        if frame:
            import matplotlib.pyplot as plt
            figure = plt.figure(1)
            figure.clear()

            axes = figure.add_axes((0.1, 0.1, 0.8, 0.8))
            axes.plot(pixelsIn, pixelsOut / pixelsIn, 'k+')
            plt.xlabel("Source amplifier pixel value")
            plt.ylabel("Measured pixel ratio")
            plt.title(f"(Source {sourceName} -> Target {targetName}) median ratio: "
                      f"{(np.median(pixelsOut / pixelsIn))}")
            figure.show()

            prompt = "Press Enter to continue: "
            while True:
                ans = input(prompt).lower()
                if ans in ("", "c",):
                    break
            plt.close()

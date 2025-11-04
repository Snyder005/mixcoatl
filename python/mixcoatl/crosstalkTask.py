# This file is part of mixcoatl.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    outputCrosstalkResults = cT.Output(
        name="crosstalkResults",
        doc="Crosstalk coefficient and error results from model fit.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure", "detector"),
    )
    outputBackgroundResults = cT.Output(
        name="backgroundResults",
        doc="Background coefficient and error results from model fit.",
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

    def __init__(self, *, config=None):

        if config.doWriteRatios is not True:

            del self.outputRatios
            del self.outputFluxes

class CrosstalkTaskConfig(pipeBase.PipelineTaskConfig,
                          pipelineConnections=CrosstalkTaskConnections):
    """Configuration for the measurement of pixel ratios.
    """
    doWriteRatios = Field(
        dtype=bool,
        default=True,
        doc="Persist outputRatios and outputFluxes?"
    )
    threshold = Field(
        dtype=float,
        default=10000.,
        doc="Minimum level of source pixels for which to measure ratios."
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
        target=DetectSpotsTask,
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

        outputCrosstalkResults = defaultdict(lambda: defaultdict(dict))
        outputBackgroundResults = defaultdict(lambda: defaultdict(dict))
        outputSignals = defaultdict(lambda: defaultdict(dict))
        outputRatios = defaultdict(lambda: defaultdict(dict))
        outputFluxes = defaultdict(lambda: defaultdict(dict))

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

            crosstalkResultsDict = defaultdict(lambda: defaultdict(list))
            backgroundResultsDict = defaultdict(lambda: defaultdict(list))
            ratioDict = defaultdict(lambda: defaultdict(list))
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
                        sourceMask = detectedSourceResults.sourceMask
                        signal = detectedSourceResults.signals[-1]

                    elif self.config.sourceType == 'spot':

                        detectedSourceResults = self.detectSpots.run(sourceAmpImage)
                        sourceMask = detectedSourceResults.sourceMask
                        signal = detectedSourceResults.signal
                except RuntimeError:
                    continue


                sourcePixels = (sourceMask & (sourceAmpMask & bad == 0) 
                    & np.isfinite(sourceAmpImage.image.array))
                ratioPixels = sourcePixels & (sourceAmpMask & detected > 0)
                count = np.sum(ratioPixels)
                self.log.debug("  Source amplifier: %s", sourceAmpName)

                outputSignals[sourceChip][sourceAmpName] = [float(signal)]
                outputFluxes[sourceChip][sourceAmpName] = sourceAmpImage.image.array[ratioPixels].tolist()

                for targetAmp in targetDetector:
                    # iterate over targetExposure
                    targetAmpName = targetAmp.getName()
                    if sourceAmpName == targetAmpName and sourceChip == targetChip:
                        crosstalkResultsDict[targetAmpName][sourceAmpName] = {}
                        backgroundResultsDict[targetAmpName][sourceAmpName] = {}
                        ratioDict[targetAmpName][sourceAmpName] = []
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
                    targetAmpMask = targetAmpImage.mask.array
                    modelPixels = sourcePixels & (targetAmpMask & bad ==0) & (targetAmpMask & detected == 0)

                    try:
                        results = self.crosstalkSolve.run(sourceAmpArray, targetAmpArray, modelPixels, 
                                                      covariance=covariance, seed=189)
                    except np.linalg.LinAlgError:
                        continue
    
                    ## Calculate background-subtracted ratios
                    bg = results.background
                    ratios = (targetAmpArray-bg)[ratioPixels]/sourceAmpArray[ratioPixels]

                    crosstalkResultsDict[targetAmpName][sourceAmpName] = results.crosstalkResults
                    backgroundResultsDict[targetAmpName][sourceAmpName] = results.backgroundResults
                    ratioDict[targetAmpName][sourceAmpName] = ratios.tolist()
                    extractedCount += count

            self.log.info("Extracted %d pixels from %s -> %s",
                          extractedCount, sourceChip, targetChip)
            outputCrosstalkResults[targetChip][sourceChip] = crosstalkResultsDict
            outputBackgroundResults[targetChip][sourceChip] = backgroundResultsDict
            outputRatios[targetChip][sourceChip] = ratioDict

        return pipeBase.Struct(
            outputCrosstalkResults=ddict2dict(outputCrosstalkResults),
            outputBackgroundResults=ddict2dict(outputBackgroundResults),
            outputSignals=ddict2dict(outputSignals),
            outputRatios=ddict2dict(outputRatios),
            outputFluxes=ddict2dict(outputFluxes),
        )

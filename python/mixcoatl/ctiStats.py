from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lsst.afw.math as afw_math
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes as cT
from lsst.eo.pipe.isr_utils import apply_minimal_isr
from lsst.eo.pipe.eperTask import compute_ctis

class CtiStatsTaskConnections(pipeBase.PipelineTaskConnections,
                               dimensions=("instrument", "detector")):
    raws = cT.Input(
        name="raw",
        doc="Raw pixel data from flat dataset.",
        storageClass="Exposure",
        dimensions=("instrument", "detector", "exposure"),
        multiple=True,
        deferLoad=True)

    bias = cT.Input(
        name="bias_frame",
        doc="Combined bias frame",
        storageClass="Exposure",
        dimensions=("instrument", "detector"),
        isCalibration=True)

    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera used in observations",
        storageClass="Camera",
        isCalibration=True,
        dimensions=("instrument",))

    cti_vs_flux = cT.Output(
        name="cti_vs_flux",
        doc="Serial and parallel CTI values as a function of flux.",
        storageClass="DataFrame",
        dimensions=("instrument", "detector"))


class CtiStatsTaskConfig(pipeBase.PipelineTaskConfig,
                          pipelineConnections=CtiStatsTaskConnections):
    nx_skip = pexConfig.Field(
        doc=("Number columns at the leading and trailing edges of "
             "the serial overscan to omit when estimating the "
             "serial overscan correction."),
        default=4,
        dtype=int)
    overscan_pixels = pexConfig.Field(
        doc=("Number of overscan rows or columns to use for "
             "evaluating the trailed signal in the overscan regions."),
        default=3,
        dtype=int)
    oscan_method = pexConfig.ChoiceField(
        doc="Overscan modeling method",
        default="median_per_row",
        dtype=str,
        allowed={
            "mean": "Mean of all selected pixels in overscan region",
            "median": "Median of all selected pixels in overscan region",
            "median_per_row": "Median of each row of selected pixels",
            "1d_poly": "1D polynomial of degree 2 fit to median_per_row data"})
    polynomial_degree = pexConfig.Field(
        doc="Degree of polynomial to fit to overscan row medians",
        default=2,
        dtype=int)
    do_parallel_oscan = pexConfig.Field(
        doc="Flag to do parallel overscan correction in addition to serial",
        default=True,
        dtype=bool)

class CtiStatsTask(pipeBase.PipelineTask):
    """Task to measure serial and parallel CTI as a function of incident flux
    using the EPER method."""
    ConfigClass = CtiStatsTaskConfig
    _DefaultName = "ctiStatsTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dx = self.config.nx_skip
        self.npix = self.config.overscan_pixels
        self.oscan_method = self.config.oscan_method
        self.deg = self.config.polynomial_degree
        self.do_parallel = self.config.do_parallel_oscan

    def run(self, raws, bias, camera):
        det = camera[raws[0].dataId['detector']]
        det_name = det.getName()

        # Compute the serial and parallel CTIs over all flats and amps.
        data = defaultdict(list)
        for raw in raws:
            for amp, amp_info in enumerate(det):
                amp_name = amp_info.getName()
                dark = None  # Don't apply dark subtraction for EPER analysis
                flat = apply_minimal_isr(raw.get(), bias, dark, amp, dx=self.dx,
                                         oscan_method=self.oscan_method,
                                         deg=self.deg,
                                         do_parallel=self.do_parallel)
                signal = np.median(flat.array)
                scti, pcti = compute_ctis(flat, det[amp],
                                          npix=self.npix)
                data['det_name'].append(det_name)
                data['amp_name'].append(amp_name)
                data['signal'].append(signal)
                data['scti'].append(scti)
                data['pcti'].append(pcti)
        df0 = pd.DataFrame(data)
        df0.sort_values('signal', inplace=True)

        return pipeBase.Struct(cti_vs_flux=df0)

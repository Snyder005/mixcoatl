from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import cv2
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.utils.timer import timeMethod
from lsst.pipe.base import connectionTypes
from lsst.meas.algorithms import MaskStreaksTask
from lsst.meas.algorithms.maskStreaks import LineCollection

class StreakFinderConfig(pexConfig.Config):
    """Configurable parameters for StreakFinderTask.
    """
    binsize = pexConfig.Field(
        dtype=int,
        default=4,
        doc="Size of pixel binning of the image.",
    )

    kernel = pexConfig.Field(
        dtype=int,
        default=11,
        doc="Size in pixels of Gaussian kernel for initial smoothing.",
    )

    sigma = pexConfig.Field(
        dtype=float,
        default=12.0,
        doc="Standard deviation of the Gaussian kernel used to compute the Hessian second derivatives.",
    )

    edge = pexConfig.Field(
        dtype=int,
        default=20,
        doc="Number of edge pixels to zero out in the binned image before thresholding.",
    )

    threshold = pexConfig.Field(
        dtype=float,
        default=-0.05,
        doc="Threshold applied to the Hessian minima ridges eigenvalue array.",
    )

    aspect = pexConfig.Field(
        dtype=float,
        default=8.0,
        doc="Lower bound for the aspect ratio of regions, after thresholding",
    )

    streakWidth = pexConfig.Field(
        dtype=int,
        default=100,
        doc="Width of the streak which is drawn on the image",
    )

    limit = pexConfig.Field(
        dtype=float,
        default=0.10,
        doc="Limit of the deviation from horizontal/vertical orientation for streak exclusion",
    )

class StreakFinderTask(pipeBase.Task):
    ConfigClass = StreakFinderConfig
    _DefaultName = "streakFinder"

    @timeMethod
    def run(self, exposure):

        arr = exposure.getImage().getArray()    

        # Bin original image down to binxbin pixels
        binsize = self.config.binsize
        arr = np.clip(arr, a_min=0, a_max=100)
        new_shape = (int(arr.shape[0] / binsize), int(arr.shape[1] / binsize))
        
        # Rebin by averaging
        bin_arr = arr.reshape(
            new_shape[0],
            arr.shape[0] // new_shape[0],
            new_shape[1],
            arr.shape[1] // new_shape[1]
        ).mean(-1).mean(1)
    
        # Use the Hessian matrix to find streaks
        # The minima ridges output has been most effective
        # in finding the streaks
        kernel = self.config.kernel
        gauss = cv2.GaussianBlur(bin_arr, (kernel, kernel), 0) # Blur with Gaussian kernel

        sigma = self.config.sigma
        H_elems = hessian_matrix(gauss, sigma=sigma, order='rc', use_gaussian_derivatives=False)
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
        
        # Now we create a binary image 
        # Setting this threshold has been tricky
        threshold = self.config.threshold
        binary_ridges = minima_ridges < threshold
        binary_ridges = binary_ridges.astype(np.uint8)
        
        # Set edges of binary_ridges to zero
        edge = self.config.edge
        binary_ridges[:,0:edge] = 0
        binary_ridges[:,-edge:-1] = 0
        binary_ridges[0:edge,:] = 0
        binary_ridges[-edge:-1,:] = 0
        
        # Convert to 0 -> 255
        _, binary = cv2.threshold(binary_ridges, 0.5, 255, cv2.THRESH_BINARY)
        
        # Find connected regions
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Sort to find regions with long aspect ratios
        long_labels = []
        aspect = self.config.aspect
        for i in range(num_labels):
            mask = np.uint8(labels == i)
            # Extract points (x,y) of this component
            ys, xs = np.where(mask > 0)
            points = np.column_stack((xs, ys))
            rect = cv2.minAreaRect(points)
            (center, (width, height), angle) = rect
            if height > 0:
                aspect_ratio = max(width, height) / min(width, height)
            else:
                aspect_ratio = 0  # Handle division by zero for flat regions
            if aspect_ratio > aspect:
                long_labels.append(i)
    
        c, r = arr.shape
        # Fit lines to the longest ones
        rhos = []
        thetas = []
        lines = LineCollection([], [])
        for label in long_labels:
            mask = np.uint8(labels == label)
            # Extract points (x,y) of this component
            ys, xs = np.where(mask > 0)
            points = np.column_stack((xs, ys))
            
            # Fit a line through the points
            # x0, y0 are shape centroid
            # vx, vy are a normlized vector in the direction of the line
            # Resize x0 and y0 to the original image
            [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = vx[0]; vy = vy[0]; x0 = x0[0] * binsize; y0 = y0[0] * binsize
            
            # Weed out near horizontal or vertical lines
            limit = self.config.limit
            if (abs(vx) < limit) or (abs(vy) < limit):
                continue
                
            # Now find rho, theta for lsst.meas.algorithms.maskStreaks.Line class
            theta = -np.atan2(vx, vy)
            rho = (x0 - c/2) * np.cos(theta) + (y0 - r/2) * np.sin(theta)
            theta *= 180.0 / np.pi # Comvert to degrees
            rhos.append(rho)
            thetas.append(theta)
            lines = LineCollection(np.array(rhos), np.array(thetas))

        return pipeBase.Struct(
            lines=lines,
            minima_ridges=minima_ridges,
            binary_ridges=binary_ridges,
        )

class DetectStreaksTaskConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", "visit", "detector")):

    exposure = connectionTypes.Input(
        doc="Background-subtracted exposure to detect streaks on.",
        name="preliminary_visit_image",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "detector"),
    )
    detectedLines = connectionTypes.Output(
        doc="Lines detected in the input exposure.",
        name="detected_lines",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "visit", "detector"),
    )   
    

class DetectStreaksTaskConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=DetectStreaksTaskConnections):
    """Configuration parameters for DetectStreaksTask.
    """
    detectionAlgorithm = pexConfig.ChoiceField(
        dtype=str,
        default="canny-hough",
        doc="Line detection algorithm to use.",
        allowed={
            "canny-hough" : "Canny edge and Hough transform.",
            "hessian" : "Hessian matrix.",
        }
    )
    maskStreaks = pexConfig.ConfigurableField(
        target=MaskStreaksTask,
        doc="Detect streaks using Canny edge and Kernel Hough Transform."
    )
    streakFinder = pexConfig.ConfigurableField(
        target=StreakFinderTask,
        doc="Detect streaks using Hessian matrix line detection."
    )

class DetectStreaksTask(pipeBase.PipelineTask):

    ConfigClass = DetectStreaksTaskConfig
    _DefaultName = "detectStreaks"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.makeSubtask('maskStreaks')
        self.makeSubtask('streakFinder')

    @timeMethod
    def run(self, exposure):

        if self.config.detectionAlgorithm == 'canny-hough':
            detectedLineResults = self.maskStreaks.run(exposure.getMaskedImage())
        elif self.config.detectionAlgorithm == 'hessian':
            detectedLineResults = self.streakFinder.run(exposure)

        lines = detectedLineResults.lines
        linesDict = {'rhos' : lines.rhos.tolist(),
                     'thetas' : lines.thetas.tolist(),
                     'sigmas' : lines.sigmas.tolist(),
                    }

        return pipeBase.Struct(
            detectedLines=linesDict,
        )

import copy
import numpy as np
from astropy.io import fits

from lsst.utils.timer import timeMethod
import lsst.afw.image as afwImage
from lsst.afw.detection import FootprintSet, Threshold
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig

def calculate_covariance(exposure, amp1, amp2):
    """Calculate read noise covariance between amplifiers.
    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Exposure from which to measure read noise covariance.
    amp1 : `lsst.afw.cameraGeom.Amplifier`
        First amplifier to use in covariance calculation.
    amp2 : `lsst.afw.cameraGeom.Amplifier`
        Second amplifier to use in covariance calculation.
    Returns
    -------
    cov : `numpy.ndarray`, (2, 2)
        A 2-d array representing the covariance matrix.
    """
    ccd = exposure.getDetector()

    oscanImage1 = exposure.maskedImage[amp1.getRawHorizontalOverscanBBox()]
    oscanImage2 = exposure.maskedImage[amp2.getRawHorizontalOverscanBBox()]

    arr1 = oscanImage1.getImage().getArray().flatten()
    arr2 = oscanImage2.getImage().getArray().flatten()

    cov = np.cov(np.vstack([arr1, arr2]))

    return cov

def find_bright_columns(imarr, threshold):
    """Find bright columns in an image array.
    
    Parameters
    ----------
    imarr : `numpy.ndarrawy`, (Nx, Ny)
        An array representing an image to analyze.
    threshold : `float`
        Pixel value threshold defining a bright column.
    Returns
    -------
    bright_cols : `list`
        List of column indices corresponding to bright columns.
    """
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
    """Identify bad columns by number of masked pixels.
    
    Parameters
    ----------
    column_indices : `list`
        List of column indices.
    threshold : `int`
        Number of bad pixels required to mark the column as bad.
    Returns
    -------
    is_bad_column : `bool`
        `True` if column is bad, `False` if not.
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

def rectangular_mask(imarr, y_center, x_center, lx, ly):
    """Make a rectangular pixel mask.
    Parameters
    ----------
    imarr : `numpy.ndarray`, (Ny, Nx)
        2-D image pixel array.
    y_center : `int`
        Y-axis position of rectangle center.
    x_center : `int`
        X-axis position of rectangle center.
    lx : `int`
        Length of rectangle along X-axis.
    ly : `int`
        Length of rectangle along Y-axis.
    Returns
    -------
    mask : `numpy.ndarray`, (Ny, Nx)
        2-D mask boolean array.
    """
    Ny, Nx = imarr.shape
    Y, X = np.ogrid[:Ny, :Nx]
    select = (np.abs(Y - y_center) < ly/2.) & (np.abs(X - x_center) < lx/2.)

    return select

def circular_mask(imarr, y_center, x_center, radius):
    """Make a circular pixel mask.
    Parameters
    ----------
    imarr : `numpy.ndarray`, (Ny, Nx)
        2-D image pixel array.
    y_center : `int`
        Y-axis position of circle center.
    x_center : `int`
        X-axis position of circle center.
    radius : `float`
        Radius of the circle.
    Returns
    -------
    mask : `numpy.ndarray`, (Ny, Nx)
        2-D mask boolean array.
    """
    Ny, Nx = imarr.shape
    Y, X = np.ogrid[:Ny, :Nx]
    select = np.sqrt(np.square(Y - y_center) + np.square(X - x_center)) < radius

    return select

def annular_mask(imarr, y_center, x_center, inner_radius, outer_radius):
    """Make an annular pixel mask.
    Parameters
    ----------
    imarr : `numpy.ndarray`, (Ny, Nx)
        2-D image pixel array.
    y_center : `int`
        Y-axis position of annulus center.
    x_center : `int`
        X-axis position of annulus center.
    inner_radius : `float`
        Inner radius of the annulus.
    outer_radius : `float`
        Outer radius of the annulus.
    Returns
    -------
    mask : `numpy.ndarray`, (Ny, Nx)
        2-X mask boolean array.
    """
    if outer_radius <= inner_radius:
        raise ValueError('outer_radius {0.1f} must be greater then inner_radius {1:.1f').format(inner_radius,
                                                                                                outer_radius)
    Ny, Nx = imarr.shape
    Y, X = np.ogrid[:Ny, :Nx]
    R = np.sqrt(np.square(X-x_center) + np.square(Y-y_center))
    select = (R >= inner_radius) & (R < outer_radius)

    return select

def streak_mask(imarr, line, width):
    """Make a pixel mask along a straight line.
    Parameters
    ----------
    imarr : `numpy.ndarray`, (Ny, Nx)
        2-D image pixel array.
    line : `Line`
        Parameters of a line profile from which to derive the pixel mask.
    width : `float`
        Width of the mask.
    
    Returns
    -------
    mask : `numpy.ndarray`, (Ny, Nx)
        2-D mask boolean array.
    """
    Ny, Nx = imarr.shape
    Y, X = np.ogrid[:Ny, :Nx]
    theta = np.deg2rad(line.theta)
    x0 = -(Nx-1)/2.
    y0 = -(Ny-1)/2.
    
    select = np.abs(((X-x0)*np.cos(theta) + (Y-y0)*np.sin(theta)) - line.rho) < width/2.

    return select

def background_model(params, shape, order=1):
    """Create background model.
    Parameters
    ----------
    params : array-like, (3,)
        Input background model parameters:
        - Y-axis tilt.
        - X-axis tilt.
        - Constant offset.
    shape : array-like, (2,)
        Dimensions of 2-D background model pixel array.
    Returns
    -------
    model : `numpy.ndarray`, (shape)
        2-D background model pixel array.
    """

    model = np.ones(shape)*params[0]
    if order >= 1:
        Ny, Nx = shape
        Y, X = np.mgrid[:Ny, :Nx]
        model += params[1]*Y + params[2]*X
        if order == 2:
            model += params[3]*Y*Y + params[4]*X*X + params[5]*X*Y
    else:
        raise ValueError("Order must be an integer greater than zero: {0}".format(order))

    return model

def crosstalk_model(params, source_imarr, order=1):
    """Create crosstalk target model.
    Parameters
    ----------
    params : array-like, (4,)
        Input target model parameters:
        - crosstalk coefficient.
        - Y-axis tilt.
        - X-axis tilt.
        - Constant offset.
    aggressor_imarr : `numpy.ndarray`, (Ny, Nx)
        2-D source image pixel array.
    Returns
    -------
    model : `numpy.ndarray`, (Ny, Nx)
        2-D target model pixel array.
    """
    ## Model parameters
    crosstalk_coeff = params[0]
    bg = background_model(params[1:], source_imarr.shape, order=1)

    ## Construct model
    model = crosstalk_coeff*source_imarr + bg
    
    return model

class CrosstalkModelFitConfig(pexConfig.Config):
    
    correctCovariance = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Correct the effect of correlated read noise between amplifiers."
    )
    backgroundOrder = pexConfig.ChoiceField(
        dtype=int,
        default=1,
        doc="2D polynomial order for background model.",
        allowed={
            1 : "1st-order 2D polynomial background (sloped plane).",
            2 : "2nd-order 2D polynomial background."
        }
    )
    doNonLinearCrosstalk = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Include signal-dependent crosstalk in model fit."
    )

class CrosstalkModelFitTask(pipeBase.Task):

    ConfigClass = CrosstalkModelFitConfig
    _DefaultName = "crosstalkModelFit"

    @timeMethod
    def run(self, sourceAmpArray, targetAmpArray, sourceMask, covariance, seed=None):

        noise = np.sqrt(np.trace(covariance))

        if self.config.correctCovariance:
            
            diag = np.diag(covariance)
            invCovariance = -1*covariance
            np.fill_diagonal(invCovariance, diag)

            rng = np.random.default_rng(seed)
            correction = rng.multivariate_normal([0.0, 0.0], invCovariance, size=sourceAmpArray.shape)

            sourceAmpArray = sourceAmpArray + correction[:, :, 0]
            targetAmpArray = targetAmpArray + correction[:, :, 1]
            noise *= np.sqrt(2)

        targetStamp = targetAmpArray[sourceMask]
        sourceStamp = sourceAmpArray[sourceMask]

        if self.config.doNonLinearCrosstalk:
            bases = [sourceStamp, np.abs(sourceStamp)*sourceStamp]
            i = 1
        else:
            bases = [sourceStamp]
            i = 0
        
        ay, ax = sourceAmpArray.shape
        bases.append(np.ones((ay, ax))[sourceMask])

        if self.config.backgroundOrder >= 1:
             
            Y, X = np.mgrid[:ay, :ax]
            bases.append(Y[sourceMask])
            bases.append(X[sourceMask])

            if self.config.backgroundOrder == 2:

                bases.append((Y*Y)[sourceMask])
                bases.append((X*X)[sourceMask])
                bases.append((X*Y)[sourceMask])

        b = targetStamp/noise
        A = np.vstack(bases).T/noise
        params, res, rank, s = np.linalg.lstsq(A, b, rcond=-1)
        covar = np.linalg.inv(np.dot(A.T, A))
        errors = np.sqrt(covar.diagonal())
        dof = b.shape[0]

        c0 = params[0]
        c0Error = errors[0]
        if self.config.doNonLinearCrosstalk:
            c1 = params[1]
            c1Error = errors[1]
            i = 1
        else:
            c1 = 0.0
            c1Error = 0.0
            i = 0
        backgroundParameters = params[1+i:]
        backgroundParameterErrors = errors[1+i:]
        background = background_model(params[1+i:], sourceAmpArray.shape, order=self.config.backgroundOrder)

        return pipeBase.Struct(
            coefficient=c0,
            coefficientError=c0Error,
            nonLinearCoefficient=c1,
            nonLinearCoefficientError=c1Error,
            background=background,
            backgroundParameters=backgroundParameters,
            backgroundParameterErrors=backgroundParameterErrors,
            residuals=res,
            degreesOfFreedom=dof
        )
                    

def crosstalk_fit(source_array, target_array, select, covariance,
                  order=1, correct_covariance=False, seed=None):
    """Perform crosstalk target model least-squares minimization.
    Parameters
    ----------
    source_array: `numpy.ndarray`, (Ny, Nx)
        2-D source pixel array.
    target_array: `numpy.ndarray`, (Ny, Nx)
        2-D target pixel array.
    select: `numpy.ndarray`, (Ny, Nx)
        2-D mask boolean array.
    covariance : `numpy.ndarray`, (2, 2)
        Covariance between read noise of amplifiers.
    correct_covariance : 'bool'
        Correct covariance between read noise of amplifiers.
    seed : `int`
        Seed to initialize random generator.
    Returns
    -------
    results : `numpy.ndarray`, (10,)
        Results of least-squares minimization:
        - crosstalk coefficient.
        - Y-axis tilt.
        - X-axis tilt.
        - Constant offset.
        - Error estimate for crosstalk coefficient.
        - Error estimate for Y-axis tilt.
        - Error estimate for X-axis tilt.
        - Error estimate for constant offset.
        - Sum of residuals.
        - Reduced degrees of freedom.
    """    
    noise = np.sqrt(np.trace(covariance))
    source_imarr = copy.deepcopy(source_array)
    target_imarr = copy.deepcopy(target_array)

    ## Reduce correlated noise
    if correct_covariance:

        diag = np.diag(covariance)
        reverse_covariance = -1*covariance
        np.fill_diagonal(reverse_covariance, diag)

        rng = np.random.default_rng(seed)
        correction = rng.multivariate_normal([0.0, 0.0], reverse_covariance, size=source_imarr.shape)

        source_imarr += correction[:, :, 0]
        target_imarr += correction[:, :, 1]
        noise *= np.sqrt(2)

    target_stamp = target_imarr[select]

    ## Construct masked, compressed basis arrays
    ay, ax = source_imarr.shape
    bases = [source_imarr[select]]
    bases.append(np.ones((ay, ax))[select])
    if order >= 1:
        Y, X = np.mgrid[:ay, :ax]
        bases.append(Y[select])
        bases.append(X[select])
        if order == 2:
            bases.append((Y*Y)[select])
            bases.append((X*X)[select])
            bases.append((X*Y)[select])
    else:
        raise ValueError("Order must be an integer greater than zero: {0}".format(order))

    ## Perform least squares parameter estimation
    b = target_stamp/noise
    A = np.vstack(bases).T/noise
    params, res, rank, s = np.linalg.lstsq(A, b, rcond=-1)
    covar = np.linalg.inv(np.dot(A.T, A))
    errors = np.sqrt(covar.diagonal())
    dof = b.shape[0] - 4
    
    return pipeBase.Struct(
        coefficient = params[0],
        coefficientError = errors[0],
        backgroundParameters = params[1:],
        backgroundParameterErrors = errors[1:],
        residuals=res,
        degreesOfFreedom=dof
    )

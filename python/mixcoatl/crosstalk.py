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

import copy
import numpy as np
from astropy.io import fits
from scipy.ndimage import shift

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

def make_rectangular_mask(imarr, y_center, x_center, lx, ly):
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

def make_circular_mask(imarr, y_center, x_center, radius):
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

def make_annular_mask(imarr, y_center, x_center, inner_radius, outer_radius):
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

def make_streak_mask(imarr, line, width):
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
    x0 = (Nx-1)/2.
    y0 = (Ny-1)/2.
    
    select = np.abs(((X-x0)*np.cos(theta) + (Y-y0)*np.sin(theta)) - line.rho) < width/2.

    return select

def make_background_model(params, shape):
    """Create background model.
    Parameters
    ----------
    params : `dict`
        Background model parameters dictionary with keys:

        `"b00"`
            Constant offset term (`float`).
        `"b01"`
            First order y term (`float`).
        `"b10"`
            First order x term (`float`).
        `"b02"`
            Second order y term (`float`).
        `"b20"`
            Second order x term (`float`).
        `b11"`
            Second order xy term (`float`).
    shape : array-like, (2,)
        Dimensions of 2-D background model pixel array.
    Returns
    -------
    model : `numpy.ndarray`, (shape)
        2-D background model pixel array.
    """

    b00 = params['b00']
    b01 = params.get('b01', 0.0)
    b10 = params.get('b10', 0.0)
    b02 = params.get('b02', 0.0)
    b20 = params.get('b20', 0.0)
    b11 = params.get('b11', 0.0)

    ay, ax = shape
    Y, X = np.mgrid[:ay, :ax]
    model = b00 + b01*Y + b10*X + b02*Y*Y + b20*X*X + b11*X*Y

    return model

def make_crosstalk_model(crosstalk_params, background_params, source_imarr):
    """Create crosstalk target model.

    Parameters
    ----------
    crosstalk_params: `dict`
        Crosstalk model parameters dictionary with keys:
        
        `"c0"`
            Linear crosstalk term (`float`).
        `"c1"`
            First-order nonlinear crosstalk term (`float`).
        `"c2"`
            Second-order nonlinear crosstalk term (`float`).
        `"delay"`
            Crosstalk response delay term (`float`)
    background_params : `dict`
        Background model parameters dictionary with keys:

        `"b00"`
            Constant offset term (`float`).
        `"b01"`
            First order y term (`float`).
        `"b10"`
            First order x term (`float`).
        `"b02"`
            Second order y term (`float`).
        `"b20"`
            Second order x term (`float`).
        `b11"`
            Second order xy term (`float`).
    source_imarr : `numpy.ndarray`, (Ny, Nx)
        2-D source image pixel array.

    Returns
    -------
    model : `numpy.ndarray`, (Ny, Nx)
        2-D target model pixel array.
    """
    ## Model parameters
    c0 = crosstalk_params['c0']
    c1 = crosstalk_params.get('c1', 0.0)
    c2 = crosstalk_params.get('c2', 0.0)
    d = crosstalk_params.get('delay', 0.0)
    bg = make_background_model(background_params, source_imarr.shape)

    ## Construct model
    s = source_imarr*(1-d)+shift(source_imarr, (0, -1))*d
    model = bg + c0*s + c1*np.abs(s)*s + c2*s*s*s
    
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
            0 : "Constant background.",
            1 : "1st-order 2D polynomial background.",
            2 : "2nd-order 2D polynomial background."
        }
    )
    crosstalkOrder = pexConfig.ChoiceField(
        dtype=int,
        default=1,
        doc="Order for crosstalk model.",
        allowed={
            0 : "Constant (linear) crosstalk.",
            1 : "1st-order (non-linear) crosstalk.",
            2 : "2nd-order (non-linear) crosstalk."
        }
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
            correction = rng.multivariate_normal([0.0, 0.0], invCovariance, 
                                                 size=sourceAmpArray.shape)

            sourceAmpArray = sourceAmpArray + correction[:, :, 0]
            targetAmpArray = targetAmpArray + correction[:, :, 1]
            noise *= np.sqrt(2)

        targetStamp = targetAmpArray[sourceMask]
        sourceStamp = sourceAmpArray[sourceMask]

        ## Construct crosstalk basis polynomials
        crosstalkVectors = [sourceStamp]
        if self.config.crosstalkOrder >=1:
            crosstalkVectors.append(np.abs(sourceStamp)*sourceStamp)
            
            if self.config.crosstalkOrder == 2:
                crosstalkVectors.append(sourceStamp*sourceStamp*sourceStamp)        

        ## Construct background basis polynomials
        ay, ax = sourceAmpArray.shape
        backgroundVectors = [np.ones((ay, ax))[sourceMask]]
        if self.config.backgroundOrder >= 1:
             
            Y, X = np.mgrid[:ay, :ax]
            backgroundVectors.append(Y[sourceMask])
            backgroundVectors.append(X[sourceMask])

            if self.config.backgroundOrder == 2:

                backgroundVectors.append((Y*Y)[sourceMask])
                backgroundVectors.append((X*X)[sourceMask])
                backgroundVectors.append((X*Y)[sourceMask])

        ## Perform least squares fit
        b = targetStamp/noise
        A = np.vstack(crosstalkVectors + backgroundVectors).T/noise
        params, res, rank, s = np.linalg.lstsq(A, b, rcond=-1)
        covar = np.linalg.inv(np.dot(A.T, A))
        errors = np.sqrt(covar.diagonal())
        dof = b.shape[0]

        crosstalkParams, bgParams = np.split(params, [len(crosstalkVectors)])
        crosstalkErrors, bgErrors = np.split(errors, [len(crosstalkVectors)])

        ## Assign crosstalk results
        crosstalkResults = {'c0' : crosstalkParams[0], 
                            'c0Error' : crosstalkErrors[0]}
        if self.config.crosstalkOrder >= 1:
            crosstalkResults.update({'c1' : crosstalkParams[1],
                                     'c1Error' : crosstalkErrors[1]})
        if self.config.crosstalkOrder == 2:
            crosstalkResults.update({'c2' : crosstalkParams[2],
                                     'c2Error' : crosstalkErrors[2]})

        ## Assign background results
        backgroundResults = {'b00' : bgParams[0],
                             'b00Error' : bgErrors[0]}
        if self.config.backgroundOrder >= 1:
            backgroundResults.update({'b01' : bgParams[1],
                                      'b10' : bgParams[2],
                                      'b01Error' : bgErrors[1],
                                      'b10Error' : bgErrors[2]})
        if self.config.backgroundOrder == 2: 
            backgroundResults.update({'b02' : bgParams[3],
                                      'b20' : bgParams[4],
                                      'b11' : bgParams[5],
                                      'b02Error' : bgErrors[3],
                                      'b20Error' : bgErrors[4],
                                      'b11Error' : bgErrors[5]})
        background = make_background_model(backgroundResults, sourceAmpArray.shape)

        return pipeBase.Struct(
            crosstalkResults=crosstalkResults,
            backgroundResults=backgroundResults,
            background=background,
            residuals=res,
            degreesOfFreedom=dof
        )

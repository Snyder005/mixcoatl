"""Utility objects and functions for MixCOATL.

To Do:
    * Fix AMP2SEG and SEG2AMP definitions.
"""
from astropy.io import fits

import lsst.afw.math as afwMath
import lsst.eotest.image_utils as imutils
from lsst.eotest.sensor.MaskedCCD import MaskedCCD
from lsst.eotest.sensor.AmplifierGeometry import AmplifierGeometry, amp_loc

ITL_AMP_GEOM = AmplifierGeometry(prescan=3, nx=509, ny=2000, 
                                 detxsize=4608, detysize=4096,
                                 amp_loc=amp_loc['ITL'], vendor='ITL')
"""AmplifierGeometry: Amplifier geometry parameters for LSST ITL CCD sensors."""

E2V_AMP_GEOM = AmplifierGeometry(prescan=10, nx=512, ny=2002,
                                 detxsize=4688, detysize=4100,
                                 amp_loc=amp_loc['E2V'], vendor='E2V')
"""AmplifierGeometry: Amplifier geometry parameters for LSST E2V CCD sensors."""

AMP2SEG = {1 : 'C17', 2 : 'C16', 3 : 'C15', 4 : 'C14', 5 : 'C13', 6 : 'C12', 7 : 'C11', 8 : 'C10',
           9 : 'C00', 10 : 'C01', 11 : 'C02', 12 : 'C03', 13 : 'C04', 14 : 'C05', 15 : 'C06', 16 : 'C07'}
"""dict: Dictionary mapping from CCD amplifier number to segment names."""

SEG2AMP = {'C00' : 9, 'C01' : 10, 'C02' : 11, 'C03' : 12, 'C04' : 13, 'C05' : 14, 'C06' : 15, 'C07' : 16,
           'C10' : 8, 'C11' : 7, 'C12' : 6, 'C13' : 5, 'C14' : 4, 'C15' : 3, 'C16' : 2, 'C17' : 1}
"""dict: Dictionary mapping from CCD segment names to amplifier number."""

def calibrated_stack(infiles, outfile, bias_frame=None, dark_frame=None, 
                     linearity_correction=None, bitpix=32):
    """Make a calibrated coadd image and write FITS image file."""

    ccds = [MaskedCCD(infile, bias_frame=bias_frame, 
                      dark_frame=dark_frame, 
                      linearity_correction=linearity_correction) for infile in infiles]

    all_amps = imutils.allAmps(infiles[0])

    amp_images = {}
    for amp in all_amps:
        amp_ims = [ccd.bias_subtracted_image(amp) for ccd in ccds]
        amp_images[amp] = imutils.stack(amp_ims).getImage()

    imutils.writeFits(amp_images, outfile, infiles[0], bitpix=bitpix)

def make_superbias(sbias_frame, bias_frames):
    """Make a superbias image."""

    ## Get overscan geometry
    bias = MaskedCCD(bias_frames[0])
    overscan = bias.amp_geom.serial_overscan
    all_amps = imutils.allAmps(bias_frames[0])

    ## Make superbias
    bias_ampims = {amp : imutils.superbias(bias_frames, overscan, hdu=amp) for amp in all_amps}
    template_file = bias_frames[0]
    bitpix = 32

    with fits.open(template_file) as template:

        output = fits.HDUList()
        output.append(fits.PrimaryHDU())
        for amp in all_amps:
            output.append(fits.ImageHDU(data=bias_ampims[amp].getArray()))
        
        output[0].header.update(template[0].header)
        output[0].header['FILENAME'] = sbias_frame
        metadata = bias_ampims.get('METADATA', None)
        if metadata is not None:
            for key, val in metadata.items():
                output[0].header[key] = val
        for amp in all_amps:
            output[amp].header.update(template[amp].header)
            imutils.set_bitpix(output[amp], bitpix)
            
        output.writeto(sbias_frame, overwrite=True, checksum=True)

def calculate_read_noise(ccd, amp):
    """Calculate amplifier read noise from serial overscan."""
    
    overscan = ccd.amp_geom.serial_overscan
    
    im = ccd.bias_subtracted_image(amp)
    stdev = afwMath.makeStatistics(im.Factory(im, overscan), afwMath.STDEV).getValue()
    
    return stdev

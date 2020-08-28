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

def calibrated_stack(infiles, outfile, bias_frame=None, dark_frame=None, 
                     linearity_correction=None, bitpix=32):

    ccds = [MaskedCCD(infile, bias_frame=bias_frame, 
                      dark_frame=dark_frame, 
                      linearity_correction=linearity_correction) for infile in infiles]

    all_amps = imutils.allAmps(infiles[0])

    amp_images = {}
    for amp in all_amps:
        amp_ims = [ccd.unbiased_and_trimmed_image(amp) for ccd in ccds]
        amp_images[amp] = imutils.stack(amp_ims).getImage()

    imutils.writeFits(amp_images, outfile, infiles[0], bitpix=bitpix)

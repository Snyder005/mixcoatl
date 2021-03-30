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

    ccds = [MaskedCCD(infile, bias_frame=bias_frame, 
                      dark_frame=dark_frame, 
                      linearity_correction=linearity_correction) for infile in infiles]

    all_amps = imutils.allAmps(infiles[0])

    amp_images = {}
    for amp in all_amps:
        amp_ims = [ccd.bias_subtracted_image(amp) for ccd in ccds]
        amp_images[amp] = imutils.stack(amp_ims).getImage()

    imutils.writeFits(amp_images, outfile, infiles[0], bitpix=bitpix)

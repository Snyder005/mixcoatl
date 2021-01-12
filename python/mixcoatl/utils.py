"""Utility objects and functions for MixCOATL.

To Do:
    * Fix AMP2SEG and SEG2AMP definitions.
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets

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

class CrosstalkResults(widgets.VBox):
    
    def __init__(self, results, agg, vic):
        super().__init__()
        self.results = results
        output = widgets.Output()
        
        x, y, yerr = self.results[(agg, vic)]
        
        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(7.5, 5))
        self.ax.errorbar(x, y/y[-1], yerr=yerr/(np.sqrt(18)*y[-1]), c='blue', marker='o')
        
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.set_ylabel('Normalized Crosstalk Coefficient', fontsize=12)
        self.ax.set_xlabel('Signal [ADU]', fontsize=12)
        self.ax.grid(True, which='major', axis='both')
        self.ax.set_title('Aggressor Amp{0}, Victim Amp{1}'.format(agg, vic), fontsize=12)
        
        self.aggressor_slider = widgets.IntSlider(value=agg, min=1, max=8, step=1, description='Aggressor:',
                                                  continuous_update=False, orientation='horizontal')
        self.victim_slider = widgets.IntSlider(value=vic, min=1, max=8, step=1, description='Victim:',
                                               continuous_update=False, orientation='horizontal')
        self.norm_checkbox = widgets.Checkbox(value=True, description='Normalized:')
        
        self.aggressor_slider.observe(self.update_agg, 'value')
        self.victim_slider.observe(self.update_vic, 'value')
        self.norm_checkbox.observe(self.toggle_norm, 'value')
        
        controls = widgets.TwoByTwoLayout(top_left=self.aggressor_slider, 
                                          bottom_left=self.victim_slider,
                                          top_right=self.norm_checkbox)
        
        out_box = widgets.Box([output])
        
        self.children = [output, controls]
        
    def update_agg(self, change):
        """Remove old lines from plot and plot new one"""
        [l.remove() for l in self.ax.lines]
        [l.remove() for l in self.ax.collections]
            
        if self.norm_checkbox.value:

            x, y, yerr = self.results[(change.new, self.victim_slider.value)]
            self.ax.errorbar(x, y/y[-1], yerr=yerr/(np.sqrt(18)*y[-1]), c='blue', marker='o')
            self.ax.set_title('Aggressor Amp{0}, Victim Amp{1}'.format(change.new, self.victim_slider.value),
                              fontsize=12)
            self.ax.relim()
            self.ax.set_ylabel('Normalized Crosstalk Coefficient', fontsize=12)
            self.ax.auto_rescale()
            
        else:

            x, y, yerr = self.results[(change.new, self.victim_slider.value)]
            self.ax.errorbar(x, y, yerr=yerr/np.sqrt(18), c='blue', marker='o')
            self.ax.set_title('Aggressor Amp{0}, Victim Amp{1}'.format(change.new, self.victim_slider.value),
                              fontsize=12)
            self.ax.relim()
            self.ax.set_ylabel('Crosstalk Coefficient', fontsize=12)
            self.ax.auto_rescale()
            
        
    def update_vic(self, change):
        """Remove old lines from plot and plot new one"""
        [l.remove() for l in self.ax.lines]
        [l.remove() for l in self.ax.collections]
            
        if self.norm_checkbox.value:

            x, y, yerr = self.results[(self.aggressor_slider.value, change.new)]
            self.ax.errorbar(x, y/y[-1], yerr=yerr/(np.sqrt(18)*y[-1]), c='blue', marker='o')
            self.ax.set_title('Aggressor Amp{0}, Victim Amp{1}'.format(self.aggressor_slider.value, change.new),
                              fontsize=12)
            self.ax.relim()
            self.ax.set_ylabel('Normalized Crosstalk Coefficient', fontsize=12)
            self.ax.auto_rescale()
            
        else:

            x, y, yerr = self.results[(self.aggressor_slider.value, change.new)]
            self.ax.errorbar(x, y, yerr=yerr/np.sqrt(18), c='blue', marker='o')
            
            self.ax.set_title('Aggressor Amp{0}, Victim Amp{1}'.format(self.aggressor_slider.value, change.new),
                              fontsize=12)
            self.ax.relim()
            self.ax.set_ylabel('Crosstalk Coefficient', fontsize=12)
            self.ax.auto_rescale()

    def toggle_norm(self, change):
                
        [l.remove() for l in self.ax.lines]
        [l.remove() for l in self.ax.collections]
            
        if change.new:

            x, y, yerr = self.results[(self.aggressor_slider.value, self.victim_slider.value)]
            self.ax.errorbar(x, y/y[-1], yerr=yerr/(np.sqrt(18)*y[-1]), c='blue', marker='o')
            self.ax.relim()
            self.ax.set_ylabel('Normalized Crosstalk Coefficient', fontsize=12)
            self.ax.auto_rescale()
            
        else:

            x, y, yerr = self.results[(self.aggressor_slider.value, self.victim_slider.value)]
            self.ax.errorbar(x, y, yerr=yerr/np.sqrt(18), c='blue', marker='o')
            self.ax.relim()
            self.ax.set_ylabel('Crosstalk Coefficient', fontsize=12)
            self.ax.auto_rescale()

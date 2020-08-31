#!/usr/bin/env python
import argparse
import glob
import numpy as np
from astropy.io import fits
from os.path import join

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

def main(sensor_id, infile, binned_cmap=False, output_dir='./'):

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ## Binned colormap
    if binned_cmap:
        cmap = plt.cm.seismic
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        cmap.set_bad(color='black')

        bounds = np.asarray([-1.0E-2, -1.0E-3,-1.0E-4,-1.0E-5,-1.0E-6, 
                              1.0E-6, 1.0E-5, 1.0E-4, 1.0E-3, 1.0E-2])
        norm = colors.BoundaryNorm(bounds, cmap.N)

        cbar_ticks = [-5.0E-3,-5.0E-4,-5.0E-5,-5.0E-6, 0., 
                       5.0E-6, 5.0E-5, 5.0E-4, 5.0E-3]
        cbar_ticklabels = [r'$-10^{-3}$', r'$-10^{-4}$', r'$-10^{-5}$', 
                           r'$-10^{-6}$', '0', r'$+10^{-6}$', 
                           r'$+10^{-5}$', r'$+10^{-4}$', r'$+10^{-3}$']

    else:
        cmap = plt.cm.seismic
        cmap.set_bad(color='black')
        norm = colors.SymLogNorm(1E-6, vmin=-1.E-2, vmax=1E-2)
        cbar_ticks = [-1.0E-2, -1.0E-3,-1.0E-4,-1.0E-5,-1.0E-6, 
                       1.0E-6, 1.0E-5, 1.0E-4, 1.0E-3, 1.0E-2]
        cbar_ticklabels = [r'$-10^{-2}$', r'$-10^{-3}$', r'$-10^{-4}$', 
                           r'$-10^{-5}$', r'$-10^{-6}$', r'$+10^{-6}$', 
                           r'$+10^{-5}$', r'$+10^{-4}$', r'$+10^{-3}$', 
                           r'$+10^{-2}$']

    with fits.open(infile) as hdulist:
        xtalk = hdulist[1].data
        agg_amp = hdulist[0].header['AGGRESSOR']
    np.fill_diagonal(xtalk, np.nan)

    im = ax.imshow(xtalk, norm=norm, cmap=cmap, interpolation='none', 
                   extent=(0.5, 16.5, 16.5, 0.5))

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Aggressor Amplifier', fontsize=20)
    ax.set_xlabel('Victim Amplifier', fontsize=20)
    ax.set_title('{0}'.format(agg_amp), fontsize=22)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', 
                        ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticklabels)
    cbar.set_label("Crosstalk ", size=18)
    cbar.ax.tick_params(labelsize=14)

    outfile = join(output_dir,
                   '{0}_crosstalk_coefficients.png'.format(sensor_id))
    plt.savefig(outfile)
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Plot crosstalk results for a sensor.")
    parser.add_argument('sensor_id', type=str,
                        help='CCD identifier (e.g. R22_S11)')
    parser.add_argument('infile', type=str,
                        help='Crosstalk matrix results file.')
    parser.add_argument('--binned_cmap', action='store_true',
                        help='Use logarithmically binned color map.')
    parser.add_argument('--output_dir', '-o', type=str, default='./',
                        help='Output directory for analysis products.')
    args = parser.parse_args()

    main(args.sensor_id, args.infile, binned_cmap=args.binned_cmap,
         output_dir=args.output_dir)

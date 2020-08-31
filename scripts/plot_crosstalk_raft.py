#!/usr/bin/env python
import argparse
import glob
import numpy as np
from astropy.io import fits
from os.path import join

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

def main(raft_id, results_dir, binned_cmap=False, full_matrix=True, output_dir='./'):

    sensor_list = ['S00', 'S01', 'S02', 
                   'S10', 'S11', 'S12', 
                   'S20', 'S21', 'S22']

    ## Binned colormap
    if binned_cmap:
        cmap = plt.cm.seismic
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        cmap.set_bad(color='black')

        bounds = np.asarray([-1.0E-2, -1.0E-3,-1.0E-4,
                              -1.0E-5,-1.0E-6, 1.0E-6, 
                              1.0E-5, 1.0E-4, 1.0E-3, 1.0E-2])
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

    ## Plot 
    if full_matrix:
        fig, ax = plt.subplots(1,1, figsize=(14, 12))

        raft_xtalk = np.full((9*16, 9*16), 0.0)
        for i, sensor_id in enumerate(sensor_list):

            for j, sensor_id2 in enumerate(sensor_list):

                infiles = glob.glob(join(results_dir, 
                                         '{0}_{1}_{0}_{2}*.fits'.format(raft_id, 
                                                                        sensor_id, 
                                                                        sensor_id2)))
                                   
                try:
                    with fits.open(infiles[0]) as hdulist:
                        xtalk = hdulist[1].data
                except:
                    continue

                raft_xtalk[i*16:(i+1)*16, j*16:(j+1)*16] = xtalk

        np.fill_diagonal(raft_xtalk, np.nan)

        im1 = ax.imshow(raft_xtalk, norm=norm, cmap=cmap, interpolation='none')

        ## Add lines to differentiate CCD regions
        for i in range(8):
            ax.axvline(x=15.5+16*i, color='dimgray')
        for i in range(8):
            ax.axhline(y=15.5+16*i, color='dimgray')

        ## Set up binned colorbar
        cbar = fig.colorbar(im1,  ax=ax, orientation='vertical', 
                            ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticklabels)
        cbar.set_label("Cross Talk", size=18)
        cbar.ax.tick_params(labelsize=14)

        ## Label by CCD
        labels = ('S00', 'S01', 'S02', 
                  'S10', 'S11', 'S12', 
                  'S20', 'S21', 'S22')
        ax.set_xticks([7.5+16*i for i in range(9)])
        ax.set_yticks([7.5+16*i for i in range(9)])
        ax.set_xticklabels(labels, {'fontsize':14})
        ax.set_yticklabels(labels, {'fontsize':14})

        ax.set_ylabel('Aggressor Amplifier', fontsize=18)
        ax.set_xlabel('Victim Amplifier', fontsize=18)
        ax.set_title('{0} Crosstalk Matrix'.format(raft_id), fontsize=22)

    else:

        fig, axes = plt.subplots(3, 3, figsize=(14, 14), sharey=True, sharex=True)
        fig.subplots_adjust(hspace=0.15, wspace=-.2)
        axes = axes.flatten()

        for i, sensor_id in enumerate(sensor_list):

            infiles = glob.glob(join(results_dir, 
                                     '{0}_{1}_{0}_{1}*.fits'.format(raft_id, 
                                                                    sensor_id)))
            print(sensor_id, infiles)
            try:
                with fits.open(infiles[0]) as hdulist:
                    xtalk = hdulist[1].data
            except:
                print("Not found")
                continue

            np.fill_diagonal(xtalk, np.nan)

            im1 = axes[i].imshow(xtalk, norm=norm, cmap=cmap, interpolation='none', 
                                extent=(0.5, 16.5, 16.5, 0.5))
            axes[i].set_title('{0}'.format(sensor_id), fontsize=16)

        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical', 
                            ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticklabels)
        cbar.set_label("Crosstalk ", size=18)
        cbar.ax.tick_params(labelsize=14)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, 
                        left=False, right=False)
        plt.ylabel('Agressor Amplifier', fontsize=18, labelpad=-30)
        plt.xlabel('Victim Amplifier', fontsize=18)

    outfile = join(output_dir, 
                   '{0}_crosstalk_coefficients.png'.format(raft_id))
    plt.savefig(outfile)
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Plot crosstalk results for a raft.")
    parser.add_argument('raft_id', type=str,
                        help='Raft identifier (e.g. R22)')
    parser.add_argument('results_dir', type=str,
                        help='Directory containing crosstalk matrix results.')
    parser.add_argument('--binned_cmap', action='store_true',
                        help='Flag to use logarithmically binned color map.')
    parser.add_argument('--full_matrix', action='store_true',
                        help='Flag to plot full raft crosstalk matrix.')
    parser.add_argument('--output_dir', '-o', type=str, default='./',
                        help='Output directory for analysis products.')
    args = parser.parse_args()
    print(args)

    main(args.raft_id, args.results_dir, binned_cmap=args.binned_cmap,
         full_matrix=args.full_matrix, output_dir=args.output_dir)

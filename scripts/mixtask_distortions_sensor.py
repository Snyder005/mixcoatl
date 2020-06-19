#!/usr/bin/env python
import argparse
import os
import numpy as np

from mixcoatl.sourcegrid import DistortedGrid, BaseGrid

def main(infiles, output_dir='./', ncols=49, nrows=49):

    xstep_all = np.zeros(len(infiles))
    ystep_all = np.zeros(len(infiles))
    theta_all = np.zeros(len(infiles))
    norm_dx_all = np.zeros((len(infiles), ncols*nrows))
    norm_dy_all = np.zeros((len(infiles), ncols*nrows))
    dxx_all = np.zeros((len(infiles), ncols*nrows))
    dyy_all = np.zeros((len(infiles), ncols*nrows))
    dflux_all = np.zeros((len(infiles), ncols*nrows))

    for i, infile in enumerate(infiles):

        grid = DistortedGrid.from_fits(infile)

        ## Normalize source astrometric shifts
        rotated_dx = (np.cos(-1*grid.theta)*grid['DX'] - np.sin(-1*grid.theta)*grid['DY'])
        rotated_dy = (np.cos(-1*grid.theta)*grid['DY'] - np.sin(-1*grid.theta)*grid['DX'])

        norm_dx_all[i, :] = rotated_dx/grid.xstep
        norm_dy_all[i, :] = rotated_dy/grid.ystep
        dxx_all[i, :] = grid['DXX']
        dyy_all[i, :] = grid['DYY']
        dflux_all[i, :] = grid['DFLUX']

        xstep_all[i] = grid.xstep
        ystep_all[i] = grid.ystep
        theta_all[i] = grid.theta

    mean_xstep = np.mean(xstep_all)
    mean_ystep = np.mean(ystep_all)
    mean_theta = np.mean(theta_all)
    
    mean_dxx = np.nanmean(dxx_all, axis=0)
    mean_dyy = np.nanmean(dyy_all, axis=0)
    mean_dflux = np.nanmean(dflux_all, axis=0)

    ## Mean dx/dy involves more careful masking (test without and compare)
    mean_norm_dx = np.zeros(ncols*nrows)
    mean_norm_dy = np.zeros(ncols*nrows)

    for n in range(ncols*nrows):

        norm_dx = norm_dx_all[:, n]
        norm_dy = norm_dy_all[:, n]

        if np.count_nonzero(~np.isnan(norm_dx)) > 0:
            
            meanx = np.nanmean(norm_dx)
            stdevx = np.nanstd(norm_dx)
            meany = np.nanmean(norm_dy)
            stdevy = np.nanstd(norm_dy)

            if stdevx>0.0015:
                x_bools = np.logical_and(norm_dx>(meanx-stdevx), 
                                         norm_dx<(meanx+stdevx))
            else:
                x_bools = np.logical_and(norm_dx>(meanx-(2.5*stdevx)), 
                                         norm_dx<(meanx+(2.5*stdevx)))
            if stdevy>.0015:
                y_bools = np.logical_and(norm_dy>(meany-stdevy), 
                                         norm_dy<(meany+stdevy))
            else:
                y_bools = np.logical_and(norm_dy>(meany-(2.5*stdevy)), 
                                         norm_dy<(meany+(2.5*stdevy)))

            indices = np.where(np.logical_and(x_bools, y_bools))[0]

            if (norm_dx[indices].shape[0]>20.)*(norm_dy[indices].shape[0]>20.):
                mean_norm_dx[n] = np.nanmean(norm_dx[indices])*mean_xstep
                mean_norm_dy[n] = np.nanmean(norm_dy[indices])*mean_ystep
            
    ## Create and save optical distortions grid
    base_grid = BaseGrid(mean_ystep, mean_xstep, mean_theta, 0, 0, ncols, nrows)
    gY, gX = base_grid.make_ideal_grid()

    data = {}
    data['X'] = gX 
    data['Y'] = gY
    data['DX'] = np.cos(mean_theta)*mean_norm_dx - np.sin(mean_theta)*mean_norm_dy
    data['DY'] = np.cos(mean_theta)*mean_norm_dy + np.sin(mean_theta)*mean_norm_dx
    data['XX'] = np.zeros(ncols*nrows)
    data['YY'] = np.zeros(ncols*nrows)
    data['DXX'] = mean_dxx
    data['DYY'] = mean_dyy
    data['FLUX'] = np.zeros(ncols*nrows)
    data['DFLUX'] = mean_dflux

    print(mean_ystep, mean_xstep, mean_theta)

    optics_grid = DistortedGrid(mean_ystep, mean_xstep, mean_theta,
                                0., 0., ncols, nrows, data)
    optics_grid.write_fits(os.path.join(output_dir, 'optical_distortion_grid.fits'), 
                           overwrite=True)

    ## Subtract mean dy/dx from original grids
    for i, infile, in enumerate(infiles):

        grid = DistortedGrid.from_fits(infile)

        mean_dx = np.cos(grid.theta)*mean_norm_dx - np.sin(grid.theta)*mean_norm_dy
        mean_dy = np.cos(grid.theta)*mean_norm_dy + np.sin(grid.theta)*mean_norm_dx
                                                          
        grid['X'] += mean_dx
        grid['Y'] += mean_dy
        grid['DX'] -= mean_dx
        grid['DY'] -= mean_dy
        grid['XX'] = mean_dxx
        grid['YY'] = mean_dyy
        grid['DXX'] -= mean_dxx
        grid['DYY'] -= mean_dyy
        grid['FLUX'] = mean_dflux
        grid['DFLUX'] -= mean_dflux

        basename = os.path.basename(infile)
        outfile = os.path.join(output_dir, basename.replace('distorted_grid', 
                                                            'corrected_distorted_grid'))
        grid.write_fits(outfile, overwrite=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    main(args.infiles, output_dir=args.output_dir)

import os
import numpy as np
import argparse
from astropy.io import fits
import copy
from os.path import join

def main(infiles, output_dir='./'):

    grid_spacings = np.zeros((len(infiles), 2))
    rotated_norm_disp = np.zeros((len(infiles), 49*49, 2))
    xx = np.zeros((len(infiles), 49*49))
    yy = np.zeros((len(infiles), 49*49))

    for i, infile in enumerate(infiles):

        with fits.open(infile) as hdul:
            ## Read input file
            hdul = fits.open(infile)
            grid_dy = hdul[1].header['DY']
            grid_dx = hdul[1].header['DX']
            grid_theta = hdul[1].header['THETA']
            dx = hdul[1].data['DATA_DX']
            dy = hdul[1].data['DATA_DY']
            xx[i, :] = hdul[1].data['DATA_XX']
            yy[i, :] = hdul[1].data['DATA_YY']

            ## Rotate to grid coordinates
            rotated_dx = (np.cos(-1*grid_theta)*dx - np.sin(-1*grid_theta)*dy)
            rotated_dy = (np.cos(-1*grid_theta)*dy + np.sin(-1*grid_theta)*dx)

            ## Collect results
            rotated_norm_disp[i, :, :] = np.vstack((rotated_dx/grid_dx, 
                                               rotated_dy/grid_dy)).T
            grid_spacings[i, :] = np.asarray([grid_dx, grid_dy])

    mean_grid_dx = np.mean(grid_spacings[:,0])
    mean_grid_dy = np.mean(grid_spacings[:,1])
    mean_xx = np.nanmean(xx, axis=0)
    mean_yy = np.nanmean(yy, axis=0)
    print(mean_xx.shape)

    ## Calculate mean displacement for all sources
    mean_disp = np.zeros((49*49, 2))
    for n in range(49*49):

        elem = rotated_norm_disp[:, n, :]
        if np.count_nonzero(~np.isnan(elem[:, 0])) > 0: # check there are unmasked values

            ## Mask outliers (is this needed?)
            # here calculate mean along all dithers
            meanx = np.nanmean(elem[:,0])
            stdevx = np.nanstd(elem[:,0])
            meany = np.nanmean(elem[:,1])
            stdevy = np.nanstd(elem[:,1])

            # Masking by standard deviation?
            if stdevx>0.0015:
                x_bools = np.logical_and(elem[:,0]>(meanx-stdevx), elem[:,0]<(meanx+stdevx))
            else:
                x_bools = np.logical_and(elem[:,0]>(meanx-(2.5*stdevx)), elem[:,0]<(meanx+(2.5*stdevx)))
            if stdevy>.0015:
                y_bools = np.logical_and(elem[:,1]>(meany-stdevy), elem[:,1]<(meany+stdevy))
            else:
                y_bools = np.logical_and(elem[:,1]>(meany-(2.5*stdevy)), elem[:,1]<(meany+(2.5*stdevy)))

            ind_arr = np.where(np.logical_and(x_bools,y_bools))[0]
            new_el = elem[ind_arr,:]

            ## If more than 20 unmasked values, then do final mean calculation
            if new_el.shape[0]>20:
                avx = np.nanmean(new_el[:,0])*mean_grid_dx
                avy = np.nanmean(new_el[:,1])*mean_grid_dy
                mean_disp[n, :] = np.array([avx, avy])            

    ## Save the mean displacement vectors (rewrite to different format later)
    hdu = fits.PrimaryHDU(mean_disp)
    hdu.writeto(os.path.join(output_dir, "corrected_grid_pt_average_displacement.fits"), overwrite=True)

    ## Subtract the mean displacement vectors from original displacments
    for i, infile in enumerate(infiles):

        with fits.open(infile) as hdul:
            
            ## Get existing raw data
            grid_theta = hdul[1].header['THETA']
            dx = hdul[1].data['DATA_DX']
            dy = hdul[1].data['DATA_DY']
            model_x = hdul[1].data['MODEL_X']
            model_y = hdul[1].data['MODEL_Y']
            xx_array = hdul[1].data['DATA_XX']
            yy_array = hdul[1].data['DATA_YY']

            ## Rotate mean displacements to pixel coordinates
            mean_dx = np.cos(grid_theta)*mean_disp[:, 0] - np.sin(grid_theta)*mean_disp[:, 1]
            mean_dy = np.cos(grid_theta)*mean_disp[:, 1] + np.sin(grid_theta)*mean_disp[:, 0]

            ## Correct displacements
            corrected_dx = dx - mean_dx
            corrected_dy = dy - mean_dy
            corrected_model_x = model_x + mean_dx
            corrected_model_y = model_y + mean_dy

            ## Construct output hdulist
            hdr = copy.deepcopy(hdul[1].header)
            columns = [fits.Column('DATA_DX', array=corrected_dx, format='D'),
                       fits.Column('DATA_DY', array=corrected_dy, format='D'),
                       fits.Column('MODEL_X', array=corrected_model_x, format='D'),
                       fits.Column('MODEL_Y', array=corrected_model_y, format='D'),
                       fits.Column('DATA_XX', array=xx_array, format='D'),
                       fits.Column('DATA_YY', array=yy_array, format='D'),
                       fits.Column('DATA_DXX', array=xx_array-mean_xx, format='D'),
                       fits.Column('DATA_DYY', array=yy_array-mean_yy, format='D')]

            new_hdu = fits.BinTableHDU.from_columns(columns, header=hdr)

            basename = os.path.basename(infile)
            
            outfile = join(output_dir, 
                           basename.replace('raw_displacement_data', 'corrected_displacement_data'))
            new_hdu.writeto(outfile, overwrite=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    infiles = args.infiles
    output_dir = args.output_dir

    main(infiles, output_dir=output_dir)




import argparse
import numpy as np
from astropy.io import fits
import os
from scipy.interpolate import griddata
import scipy.ndimage as ndimage

def main(sensor_id, infiles, output_dir='./', do_smoothing=False):

    gX_results = np.zeros(49*49*len(infiles))
    gY_results = np.zeros(49*49*len(infiles))
    dxx_results = np.zeros(49*49*len(infiles))
    dyy_results = np.zeros(49*49*len(infiles))

    dx_results = np.zeros(49*49*len(infiles))
    dy_results = np.zeros(49*49*len(infiles))

    ## Add displacement results to bins
    for i, result in enumerate(infiles):

        with fits.open(result) as hdulist:

            ## Filter out vectors greater than threshold length
            dx = hdulist[1].data['DATA_DX']
            dy = hdulist[1].data['DATA_DY']

            dxx = hdulist[1].data['DATA_DXX']
            dyy = hdulist[1].data['DATA_DYY']

            dx_results[i*49*49:(i+1)*49*49] = dx
            dy_results[i*49*49:(i+1)*49*49] = dy
            dxx_results[i*49*49:(i+1)*49*49] = dxx
            dyy_results[i*49*49:(i+1)*49*49] = dyy

            gX = hdulist[1].data['MODEL_X']
            gY = hdulist[1].data['MODEL_Y']
            gX_results[i*49*49:(i+1)*49*49] = gX
            gY_results[i*49*49:(i+1)*49*49] = gY

    pixmin = 0
    pixmax = 4000
    xi = yi = np.arange(pixmin,pixmax,1.0)
    xi,yi = np.meshgrid(xi,yi)

    # Interpolate
    DX_im = griddata((gX_results, gY_results),dx_results,(xi,yi),method='linear')
    DY_im = griddata((gX_results, gY_results),dy_results,(xi,yi),method='linear')
    DXX_im = griddata((gX_results, gY_results), dxx_results, (xi,yi),method='linear')
    DYY_im = griddata((gX_results, gY_results), dyy_results, (xi,yi),method='linear')

    if do_smoothing:

        DY_im = ndimage.gaussian_filter(DY_im, sigma=(2, 2), order=0)
        DX_im = ndimage.gaussian_filter(DX_im, sigma=(2, 2), order=0)
        DYY_im = ndimage.gaussian_filter(DYY_im, sigma=(2, 2), order=0)
        DXX_im = ndimage.gaussian_filter(DXX_im, sigma=(2, 2), order=0)

    prihdu = fits.PrimaryHDU()

    DX_hdu = fits.ImageHDU(data=DX_im)
    DY_hdu = fits.ImageHDU(data=DY_im)
    DXX_hdu = fits.ImageHDU(data=DXX_im)
    DYY_hdu = fits.ImageHDU(data=DYY_im)

    hdulist = fits.HDUList([prihdu, DX_hdu, DY_hdu, DXX_hdu, DYY_hdu])
    hdulist.writeto(os.path.join(output_dir, '{0}_astrometric_shifts.fits'.format(sensor_id)))        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str)
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('-o', '--output_dir', type=str, default='./')
    parser.add_argument('--do_smoothing', action='store_true')
    args = parser.parse_args()

    sensor_id = args.sensor_id
    infiles = args.infiles
    output_dir = args.output_dir
    do_smoothing = args.do_smoothing

    main(sensor_id, infiles, output_dir=output_dir, do_smoothing=do_smoothing)

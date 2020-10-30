#!/usr/bin/env python
import argparse
import os

from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

from mixcoatl.gridFitTask import GridFitTask

def main(sensor_id, infile, brute_search=False, ccd_type=None, dx0=0., dy0=0.,
         optics_grid_file=None, output_dir='./', vary_theta=False):

    basename = os.path.basename(infile)
    root = os.path.splitext(basename)[0]
    outfile = os.path.join(output_dir, '{0}_gridfit.cat'.format(root))

    ## Make initial grid center guess
    with fits.open(infile) as src:

        all_srcY = src[1].data['base_SdssCentroid_Y']
        all_srcX = src[1].data['base_SdssCentroid_X']

        mask = (src[1].data['base_SdssShape_XX'] > 4.5) \
            *(src[1].data['base_SdssShape_XX'] < 7.) \
            *(src[1].data['base_SdssShape_YY'] > 4.5) \
            *(src[1].data['base_SdssShape_YY'] < 7.)

        y0_guess = np.nanmedian(all_srcY[mask])
        x0_guess = np.nanmedian(all_srcX[mask])

    ## Configure and run task
    gridfit_task = GridFitTask()
    gridfit_task.config.brute_search = brute_search
    gridfit_task.config.vary_theta = vary_theta
    gridfit_task.config.outfile = outfile
    
    grid, result = gridfit_task.run(infile, (y0_guess, x0_guess),
                                    optics_grid_file=optics_grid_file,
                                    ccd_type=ccd_type)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run GridFitTask on single file.')
    parser.add_argument('sensor_id', type=str, 
                        help='CCD identifier (e.g. R22_S11).')
    parser.add_argument('infile', type=str, 
                        help='Input catalog file to process.')
    parser.add_argument('--brute', action='store_true',
                        help='Flag to enable intial brute search.')
    parser.add_argument('--ccd_type', type=str, default=None,
                        help='CCD manufacturer type (ITL or E2V).')
    parser.add_argument('--dx0', type=float, default=0.,
                        help='Shift in x0 for grid center guess.')
    parser.add_argument('--dy0', type=float, default=0.,
                        help='Shift in y0 for grid center guess.')
    parser.add_argument('--optics_grid_file', type=str, default=None,
                        help='FITS or CAT file with optic shifts.')
    parser.add_argument('--output_dir', '-o', type=str, default='./',
                        help='Output directory for analysis products.')
    parser.add_argument('--vary_theta', action='store_true',
                        help='Flag to enable theta variation during fit.')
    args = parser.parse_args()

    main(args.sensor_id, args.infile, brute_search=args.brute,
         ccd_type=args.ccd_type, dx0=args.dx0, dy0=args.dy0,
         optics_grid_file=args.optics_grid_file,
         output_dir=args.output_dir, vary_theta=args.vary_theta)

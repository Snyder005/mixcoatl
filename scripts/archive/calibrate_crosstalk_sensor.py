#!/usr/env/bin python
import argparse
import glob
import os
import pickle
from os.path import join, basename, isdir

from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

from mixcoatl.utils import calibrated_stack

def main(sensor_id, main_dir, calib_dir, output_dir='./'):

    ## Get calibration products
    bias_frame = join(calib_dir, '{0}_superbias.fits'.format(sensor_id))
    dark_frame = join(calib_dir, '{0}_superdark.fits'.format(sensor_id))

    ## Get projector positions and exptimes
    position_set = set()
    exptime_set = set()
    camera = camMapper._makeCamera()
    lct = LsstCameraTransforms(camera)

    subdir_list = [x.path for x in os.scandir(main_dir) if isdir(x.path)]
    for subdir in subdir_list:
        base = basename(subdir)
        if "xtalk" not in base: continue
        xpos, ypos, exptime = base.split('_')[-4:-1]
        central_ccd, ccdX, ccdY = lct.focalMmToCcdPixel(float(ypos), float(xpos))
        if central_ccd == sensor_id:
            position_set.add((xpos, ypos))
            exptime_set.add(exptime)

    ## For each exptime construct calibrated crosstalk image
    for exptime in exptime_set:
        for i, pos in enumerate(position_set):
            xpos, ypos = pos
            infiles = glob.glob(join(main_dir, 
                                     '*{0}_{1}_{2}*'.format(xpos, ypos, exptime),
                                     '*{0}.fits'.format(sensor_id)))

            pos_output_dir = join(output_dir, 
                                  'xtalk_{0}_{1}_{2}_calibrated'.format(xpos, ypos,
                                                                        exptime))
            os.mkdir(pos_output_dir) # add check if already existing
            base = basename(infiles[0])
            outfile = join(pos_output_dir,
                           base.replace(sensor_id, 
                                        '{0}_calibrated'.format(sensor_id)))

            calibrated_stack(infiles, outfile, bias_frame=bias_frame, 
                             dark_frame=dark_frame)

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Make calibrated crosstalk images for a sensor.")
    parser.add_argument('sensor_id', type=str,
                        help='CCD identifier (e.g. R22_S11)')
    parser.add_argument('main_dir', type=str,
                        help='Directory containing acquisition subdirectories.')
    parser.add_argument('calib_dir', type=str,
                        help='Directory containing calibration products.')
    parser.add_argument('--output_dir', '-o', type=str, default='./',
                        help='Output directory for analysis products.')
    args = parser.parse_args()
    
    main(args.sensor_id, args.main_dir, args.calib_dir,
         output_dir=args.output_dir)


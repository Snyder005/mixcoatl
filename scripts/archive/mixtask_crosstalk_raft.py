#!/usr/env/bin python
import argparse
import glob
import os
import pickle
from os.path import join, basename, isdir

from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

from mixcoatl.crosstalkTask import CrosstalkSpotTask

def main(raft_id, database, main_dir, calib_dir, output_dir='./'):

    sensor_list = ['S00', 'S01', 'S02',
                   'S10', 'S11', 'S12',
                   'S20', 'S21', 'S22']
    sensor_ids = ['{0}_{1}'.format(raft_id, sensor_id) for sensor_id in sensor_list]

    ## Get bias frame
    bias_frames = {sensor_id : os.path.join(calib_dir, 
                                            '{0}_superbias.fits'.format(sensor_id)) for sensor_id in sensor_ids}

    ## Get projector positions and exptimes
    camera = camMapper._makeCamera()
    lct = LsstCameraTransforms(camera)

    subdir_list = [x.path for x in os.scandir(main_dir) if isdir(x.path)]
    for subdir in sorted(subdir_list):

        base = basename(subdir)
        if "xtalk" not in base: continue
        xpos, ypos, exptime = base.split('_')[-4:-1]
        central_sensor, ccdX, ccdY = lct.focalMmToCcdPixel(float(ypos), float(xpos))
        infiles = glob.glob(join(subdir, '*{0}.fits'.format(central_sensor)))

        ## Run crosstalk task
        for infile in infiles:
            crosstalk_task = CrosstalkSpotTask()
            crosstalk_task.config.database = database
            crosstalk_task.run(central_sensor, infile, bias_frame=bias_frame[central_sensor])

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Run CrosstalkTask on single calibrated sensor.")
    parser.add_argument('raft_id', type=str,
                        help='CCD identifier (e.g. R22)')
    parser.add_argument('database', type=str,
                        help='SQLite database.')
    parser.add_argument('main_dir', type=str,
                        help='Directory containing acquisition subdirectories.')
    parser.add_argument('calib_dir', type=str,
                        help='Directory containing calibration products.')
    parser.add_argument('--output_dir', '-o', type=str, default='./',
                        help='Output directory for analysis products.')
    args = parser.parse_args()

    main(args.raft_id, args.database, args.main_dir, args.calib_dir,
         output_dir=args.output_dir)

        

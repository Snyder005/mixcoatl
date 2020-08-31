#!/usr/env/bin python
import argparse
import glob
import os
import pickle
from os.path import join, basename, isdir

from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

from mixcoatl.crosstalkTask import CrosstalkTask

def main(raft_id, main_dir, calib_dir, output_dir='./'):

    sensor_list = ['S00', 'S01', 'S02',
                   'S10', 'S11', 'S12',
                   'S20', 'S21', 'S22']
    sensor_ids = ['{0}_{1}'.format(raft_id, sensor_id) for sensor_id in sensor_list]

    for sensor_id in sensor_ids:
        print("Sensor: {0}".format(sensor_id))

        gain_results = pickle.load(open(join(calib_dir, 
                                             'et_results.pkl'), 'rb'))
        gains = gain_results.get_amp_gains(sensor_id)

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
            central_sensor, ccdX, ccdY = lct.focalMmToCcdPixel(float(ypos), float(xpos))
            if central_sensor == sensor_id:
                position_set.add((xpos, ypos))
                exptime_set.add(exptime)

        ## For each exptime calculate crosstalk
        for exptime in exptime_set:

            outfile = join(output_dir, '{0}_{0}_{1}_crosstalk_matrix.fits'.format(sensor_id,
                                                                                  exptime))

            ## Get calibrated crosstalk infiles
            infiles = []
            for i, pos in enumerate(position_set):
                xpos, ypos = pos
                infile = glob.glob(join(main_dir, 
                                        '*{0}_{1}_{2}*'.format(xpos, ypos, exptime),
                                        '*{0}_calibrated.fits'.format(sensor_id)))[0]
                infiles.append(infile)

            ## Run crosstalk task
            crosstalk_task = CrosstalkTask()
            crosstalk_task.config.outfile = outfile
            crosstalk_task.run(sensor_id, infiles, gains)

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Run CrosstalkTask on single calibrated sensor.")
    parser.add_argument('raft_id', type=str,
                        help='CCD identifier (e.g. R22)')
    parser.add_argument('main_dir', type=str,
                        help='Directory containing acquisition subdirectories.')
    parser.add_argument('calib_dir', type=str,
                        help='Directory containing calibration products.')
    parser.add_argument('--output_dir', '-o', type=str, default='./',
                        help='Output directory for analysis products.')
    args = parser.parse_args()

    main(args.raft_id, args.main_dir, args.calib_dir,
         output_dir=args.output_dir)

        

import glob
import os
from astropy.io import fits
import argparse
import pickle
import errno

from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

import lsst.eotest.sensor as sensorTest

def main(main_dir, output_dir='./', gain_file=None):

    camera = camMapper._makeCamera()
    lct = LsstCameraTransforms(camera)

    all_acq = glob.glob(os.path.join(main_dir, 'BOT_acq', 'v0', '*', '*'))

    for acq in all_acq:
        
        if not 'spot_flat' in acq:
            continue

        split = acq.split('_')
        y = float(split[-7])
        x = float(split[-6])
        num = int(split[-1])

        try:
            ccd1 = lct.focalMmToCcdPixel(x, y-2.)
            ccd2 = lct.focalMmToCcdPixel(x, y+2.)
        except TypeError:
            try:
                ccd1 = lct.focalMmToCcdPixel(x-2., y)
                ccd2 = lct.focalMmToCcdPixel(x+2., y)
            except TypeError:
                print("Error with {0}".format(acq))
                continue

        print(acq, ccd1[0], ccd2[0]) # debug

        pair_output_dir = os.path.join(output_dir, 'spacing_{0:.1f}_{1:.1f}_{2:03d}'.format(y, x, num))
        try:
            os.makedirs(pair_output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        ## First CCD
        try:
            bias_frame1 = glob.glob(os.path.join(main_dir, 'BOT_acq', 'v0', 
                                                 '*', 'bias_bias_000', 
                                                 '*_{0}.fits'.format(ccd1[0])))[0]
            infile1 = glob.glob(os.path.join(acq, '*_{0}.fits'.format(ccd1[0])))[0]

            ## Sensor gains
            if gain_file is not None:
                gain_results = pickle.load(open(gain_file, 'rb'))
                gains1 = gain_results.get_amp_gains(ccd1[0])
            else:
                gains1 = {i : 1.0 for i in range(1, 17)}

            spot_task = sensorTest.SpotTask()
            spot_task.config.output_dir = pair_output_dir
            spot_task.run(ccd1[0], infile1, gains1, bias_frame=bias_frame1)
        except IndexError:
            print("Could not find files for {0} for {1}".format(ccd1[0], acq))

        ## Second CCD
        try:
            bias_frame2 = glob.glob(os.path.join(main_dir, 'BOT_acq', 'v0', 
                                                 '*', 'bias_bias_000',
                                                 '*_{0}.fits'.format(ccd2[0])))[0]
            infile2 = glob.glob(os.path.join(acq, '*_{0}.fits'.format(ccd2[0])))[0]

            ## Sensor gains
            if gain_file is not None:
                gain_results = pickle.load(open(gain_file, 'rb'))
                gains2 = gain_results.get_amp_gains(ccd2[0])
            else:
                gains2 = {i : 1.0 for i in range(1, 17)}

            spot_task = sensorTest.SpotTask()
            spot_task.config.output_dir = pair_output_dir
            spot_task.run(ccd2[0], infile2, gains2, bias_frame=bias_frame2)
        except IndexError:
            print("Could not find files for {0} for {1}".format(ccd2[0], acq))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('main_dir', type=str, 
                        help='Main eotest job harness directory.')
    parser.add_argument('--output_dir', '-o', type=str, default='./',
                        help='Output directory for results files.')
    parser.add_argument('--gains', '-g', type=str, default=None,
                        help='Gains pickle file.')
    args = parser.parse_args()

    main(args.main_dir, output_dir=args.output_dir, gain_file=args.gains)
        

        

#!/usr/bin/env python
import argparse
import os
import pickle

from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

from mixcoatl.crosstalkTask import CrosstalkTask

def main(sensor_id, infiles, gain_file=None, bias_frame=None, dark_frame=None, 
         output_dir='./'):

    gain_results = pickle.load(gain_file, 'rb')
    gains = gain_results.get_amp_gains(sensor_id)

    outfile = os.path.join(output_dir,
                           '{0}_{0}_crosstalk_matrix.fits'.format(sensor_id))

    crosstalk_task = CrosstalkTask()
    crosstalk_task.config.outfile = outfile
    crosstalk_task.run(sensor_id, infiles, gains, bias_frame=bias_frame,
                       dark_frame=dark_frame)

if __name_ == '__main__':

    parser = argparse.ArgumentParser("Run basic CrosstalkTask on a single sensor.")
    parser.add_argument('sensor_id', type=str,
                        help='CCD identifier (e.g. R22_S11)')
    parser.add_argument('infiles', nargs='+',
                        help='Input crosstalk image files.')
    parser.add_argument('--gain_file', '-g', type=str, default=None,
                        help='PKL file with CCD gain results.')
    parser.add_argument('--bias_frame', '-b', type=str, default=None,
                        help='FITS image of bias frame.')
    parser.add_argument('--dark_frame', '-d', type=str, default=None,
                        help='FITS image of dark frame.')
    parser.add_argument('--output_dir', '-o', type=str, default='./',
                        help='Output directory for analysis products.')
    args = parser.parse_args()

    main(args.sensor_id, args.infiles, gain_file=args.gain_file,
         bias_frame=args.bias_frame, dark_frame=args.dark_frame,
         output_dir=args.output_dir)
                        


            
    

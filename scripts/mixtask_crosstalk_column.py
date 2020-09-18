#!/usr/bin/env python
import argparse
from mixcoatl.crosstalkTask import CrosstalkColumnTask as CrosstalkTask

def main(sensor_name, infiles, database, bias_frame=None, dark_frame=None):
    
    crosstalk_task = CrosstalkTask()
    crosstalk_task.config.database = database
    crosstalk_task.run(sensor_name, infiles, bias_frame=bias_frame, dark_frame=dark_frame)

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Run CrosstalkColumnTask on image files.")
    parser.add_argument('sensor_name', type=str, 
                        help="CCD name (e.g. R22/S11")
    parser.add_argument('database', type=str,
                        help="SQL database DB file for analysis output products.")
    parser.add_argument('aggressor_amp', type=int,
                        help="Amplifier number for aggressor amplifier.")
    parser.add_argument('col', type=int,
                        help="Column number for aggressor signal.")
    parser.add_argument('infiles', type=str, nargs='+',
                        help="Input image FITS files.")
    parser.add_argument('--bias_frame', '-b', type=str, default=None,
                        help="Bias image FITS file for calibration.")
    parser.add_argument('--dark_frame', '-d', type=str, default=None,
                        help="Dark image FITS file for calibration")
    args = parser.parse_args()

    main(args.sensor_name, args.infiles, args.database, args.aggressor_amp, args.col,
         bias_frame=args.bias_frame, dark_frame=args.dark_frame)

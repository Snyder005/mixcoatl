#!/usr/bin/env python
import argparse
import logging
from datetime import datetime
from mixcoatl.crosstalkTask import CrosstalkSpotTask

def main(sensor_name, infiles, database, bias_frame=None, dark_frame=None, logfile=None):

    if logfile is None:
        logfile = database.replace('.db', '.log')

    logging.basicConfig(filename=logfile, level=logging.INFO)
    logging.info("{0}  Running mixtask_crosstalk_spot.py".format(datetime.now()))
    crosstalk_task = CrosstalkSpotTask
    crosstalk_task.config.database = database
    crosstalk_task.run(sensor_name, infiles, bias_frame=bias_frame, dark_frame=dark_frame)
    logging.info("{0}  Script completed successfully".format(datetime.now()))

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Run CrosstalkSpotTask on image files.")
    parser.add_argument('sensor_name', type=str, 
                        help="CCD name (e.g. R22/S11")
    parser.add_argument('database', type=str,
                        help="SQL database DB file for analysis output products.")
    parser.add_argument('infiles', type=str, nargs='+',
                        help="Input image FITS files.")
    parser.add_argument('--bias_frame', '-b', type=str, default=None,
                        help="Bias image FITS file for calibration.")
    parser.add_argument('--dark_frame', '-d', type=str, default=None,
                        help="Dark image FITS file for calibration")
    parser.add_argument('--log', '-l', type=str, default=None,
                        help="Optional log file to record script information.")
    args = parser.parse_args()

    main(args.sensor_name, args.infiles, args.database, bias_frame=args.bias_frame, 
         dark_frame=args.dark_frame, logfile=args.log)

import argparse
import subprocess

from os.path import join

def main(sensor_id, infiles, brute_search=False, ccd_type=None, dx0=None, dy0=None,
         log_dir='./', optics_grid_file=None, output_dir=None, vary_theta=False):

    for i, infile in enumerate(infiles):

        ## Construct base command
        logfile = join(log_dir, 'logfile_mixtask_gridfit_sensor_{0:03d}.log'.format(i))
        command = ['bsub', '-W', '1:00', '-o', logfile, 'python',
                   'mixtask_gridfit_sensor.py', sensor_id, infile]

        ## Append optional arguments
        if ccd_type is not None:
            command.extend(['--ccd_type', ccd_type])
        if dx0 is not None:
            command.extend(['--dx0', str(dx0)])
        if dy0 is not None:
            command.extend(['--dy0', str(dy0)])
        if optics_grid_file is not None:
            command.extend(['--optics_grid_file', optics_grid_file])
        if output_dir is not None:
            command.extend(['--output_dir', output_dir])
        if brute_search:
            command.append('--brute')
        if vary_theta:
            command.append('--vary_theta')

        ## Run command to submit job
        subprocess.check_output(command)
        print("Processing {0}, submitted to batch farm.".format(infile))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Submit multiple GridFitTask jobs.')
    parser.add_argument('sensor_id', type=str, 
                        help='CCD identifier (e.g. R22_S11).')
    parser.add_argument('infiles', type=str, nargs='+',
                        help='Input catalog files to process.')
    parser.add_argument('--brute', action='store_true',
                        help='Flag to enable intial brute search.')
    parser.add_argument('--ccd_type', type=str, default=None,
                        help='CCD manufacturer type (ITL or E2V).')
    parser.add_argument('--dx0', type=float, default=None,
                        help='Shift in x0 for grid center guess.')
    parser.add_argument('--dy0', type=float, default=None,
                        help='Shift in y0 for grid center guess.')
    parser.add_argument('--log_dir', '-l', type=str, default='./',
                        help='Output directory for log files.')
    parser.add_argument('--optics_grid_file', type=str, default=None,
                        help='FITS or CAT file with optic shifts.')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                        help='Output directory for data products.')
    parser.add_argument('--vary_theta', action='store_true',
                        help='Flag to enable theta variation during fit.')
    args = parser.parse_args()
    print(args)

    main(args.sensor_id, args.infiles, brute_search=args.brute,
         ccd_type=args.ccd_type, dx0=args.dx0, dy0=args.dy0,
         optics_grid_file=args.optics_grid_file,
         output_dir=args.output_dir, vary_theta=args.vary_theta)

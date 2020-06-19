#!/usr/bin/env python
import argparse
import os
import time

from mixcoatl.sourceGridTask import SourceGridTask

def main(infiles, output_dir='./'):

    for infile in infiles:
        
        basename = os.path.basename(infile)
        root = os.path.splitext(basename)[0]
        outfile = os.path.join(output_dir, '{0}_distorted_grid.fits'.format(root))

        sourcegridtask = SourceGridTask()
        sourcegridtask.config.outfile = outfile
        sourcegridtask.run(infile)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    main(args.infiles, output_dir=args.output_dir)

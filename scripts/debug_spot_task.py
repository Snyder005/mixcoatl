import argparse

from mixcoatl.sourceGridTask import SourceGridTask

def main(infile, outfile='test.fits'):

    sourcegridtask = SourceGridTask()
    sourcegridtask.config.outfile = outfile
    sourcegridtask.run(infile)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str)
    args = parser.parse_args()

    main(args.infile)

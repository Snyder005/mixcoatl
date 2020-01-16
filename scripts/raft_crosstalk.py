import subprocess
import argparse

def main(raft_name, main_dir, calib_dir, output_dir='./'):

    sensor_list = ['S00', 'S01', 'S02', 
                   'S10', 'S11', 'S12', 
                   'S20', 'S21', 'S22']

    sensor_id_list = ['{0}_{1}'.format(raft_name, sensor_name) for sensor_name in sensor_list]

    ## for now only do one sensor
    for aggressor_id in sensor_id_list:

        print("Submitting crosstalk for {0}".format(aggressor_id))
        command = ['bsub', '-W', '8:00', '-R', 'bullet', '-o', 
                   '/nfs/slac/g/ki/ki19/lsst/snyder18/log/xtalklog_{0}_{0}.log'.format(aggressor_id), 
                   'python', 'crosstalk.py', aggressor_id, main_dir, calib_dir, 
                   '-o', output_dir]

        subprocess.check_output(command)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('raft_name', type=str)
    parser.add_argument('main_dir', type=str)
    parser.add_argument('calib_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str, default='./')
    args = parser.parse_args()

    raft_name = args.raft_name
    main_dir = args.main_dir
    calib_dir = args.calib_dir
    output_dir = args.output_dir

    main(raft_name, main_dir, calib_dir
         , output_dir=output_dir)
    
    

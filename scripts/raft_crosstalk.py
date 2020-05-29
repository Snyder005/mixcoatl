import subprocess
import argparse

def main(raft_id, main_dir, calib_dir, output_dir='./'):

    camera = camMapper._makeCamera()
    lct = LsstCameraTransforms(camera)
    gain_results = pickle.load(open(os.path.join(calib_dir, 
                                                 'et_results.pkl'), 'rb'))
    subdir_list = [x.path for x in os.scandir(main_dir) if os.path.isdir(x.path)]

    sensor_list = ['S00', 'S01', 'S02', 
                   'S10', 'S11', 'S12', 
                   'S20', 'S21', 'S22']

    ## Add loop over sensors below
    sensor_id_list = ['{0}_{1}'.format(raft_name, sensor_name) for sensor_name in sensor_list]

    ## for now just try with two different sensors
    sensor_id1 = sensor_id_list[0]
    sensor_id2 = sensor_id_list[1]

    bias_frame1 = os.path.join(calib_dir, '{0}_superbias.fits'.format(sensor_id1))
    dark_frame1 = os.path.join(calib_dir, '{0}_superdark.fits'.format(sensor_id1))
    gains1 = gain_results.get_amp_gains(sensor_id1)

    bias_frame2 = os.path.join(calib_dir, '{0}_superbias.fits'.format(sensor_id2))
    dark_frame2 = os.path.join(calib_dir, '{0}_superdark.fits'.format(sensor_id2))
    gains2 = gain_results.get_amp_gains(sensor_id2)

    position_set = set()
    for subdir in subdir_list:
        basename = os.path.basename(subdir)
        if "xtalk" not in basename:
            continue
        xpos, ypos = basename.split('_')[-4:-2]
        central_sensor, ccdX, ccdY = lct.focalMmToCcdPixel(float(ypos), float(xpos))
        if central_sensor == sensor_id:
            position_set.add((xpos, ypos))

    for i, pos in enumerate(position_set):
        xpos, ypos = pos
        infiles1 = glob.glob(os.path.join(main_dir, '*{0}_{1}*'.format(xpos, ypos),
                                         '*{0}.fits'.format(sensor_id1)))
        infiles2 = glob.glob(os.path.join(main_dir, '*{0}_{1}*'.format(xpos, ypos),
                                         '*{0}.fits'.format(sensor_id2)))

        if i == 0:
            output_file = os.path.join(output_dir, 
                                       '{0}_{0}_crosstalk_matrix.fits'.format(sensor_id1))
            crosstalk_task = CrosstalkTask()
            crosstalk_task.config.output_file = output_file
            crosstalk_task.run(sensor_id1, infiles1, gains1, 
                               bias_frame=bias_frame1, dark_frame=dark_frame1,
                               sensor_id2=sensor_id2, infiles2=infiles2, 
                               gains2=gains2, bias_frame2=bias_frame2, 
                               dark_frame2=dark_frame2)
        else:
            crosstalk_task = CrosstalkTask()
            crosstalk_task.run(sensor_id1, infiles1, gains1, 
                               bias_frame=bias_frame1, dark_frame=dark_frame1,
                               seensor_id2=sensor_id2, infiles2=infiles2,
                               gains2=gains2, bias_frame2=bias_frame2,
                               dark_frame2=dark_frame2,
                               crosstalk_matrix_file=output_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('raft_id', type=str)
    parser.add_argument('main_dir', type=str)
    parser.add_argument('calib_dir', type=str)
    args = parser.parse_args()

    main(args.sensor_id, args.main_dir, args.calib_dir)
    
    

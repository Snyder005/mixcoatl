import os # maybe just import join?
import argparse
import pickle
import glob

from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms


from mixcoatl.crosstalkTask import CrosstalkTask

def main(sensor_id1, sensor_id2, main_dir, calib_dir, output_dir='./'):

    camera = camMapper._makeCamera()
    lct = LsstCameraTransforms(camera)
    gain_results = pickle.load(open(os.path.join(calib_dir, 
                                                 'et_results.pkl'), 'rb'))
    subdir_list = [x.path for x in os.scandir(main_dir) if os.path.isdir(x.path)]

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
        if central_sensor == sensor_id1:
            position_set.add((xpos, ypos))

    for i, pos in enumerate(position_set):
        xpos, ypos = pos
        infiles1 = glob.glob(os.path.join(main_dir, '*{0}_{1}*'.format(xpos, ypos),
                                         '*{0}.fits'.format(sensor_id1)))
        infiles2 = glob.glob(os.path.join(main_dir, '*{0}_{1}*'.format(xpos, ypos),
                                         '*{0}.fits'.format(sensor_id2)))

        if i == 0:
            output_file = os.path.join(output_dir, 
                                       '{0}_{1}_crosstalk_matrix.fits'.format(sensor_id1,
                                                                              sensor_id2))
            crosstalk_task = CrosstalkTask()
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
    parser.add_argument('sensor_id1', type=str)
    parser.add_argument('sensor_id2', type=str)
    parser.add_argument('main_dir', type=str)
    parser.add_argument('calib_dir', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    args = parser.parse_args()

    main(args.sensor_id, args.main_dir, args.calib_dir,
         output_dir=args.output_dir)
        

    

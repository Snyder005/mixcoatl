import os # maybe just import join?
import argparse
import pickle
import glob

from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

from mixcoatl.crosstalkTask import CrosstalkTask

def main(sensor_id, main_dir, calib_dir, output_dir='./':

    camera = camMapper._makeCamera()
    lct = LsstCameraTransforms(camera)

    bias_frame = os.path.join(calib_dir, '{0}_superbias.fits'.format(sensor_id))
    dark_frame = os.path.join(calib_dir, '{0}_superdark.fits'.format(sensor_id))
    gain_results = pickle.load(open(os.path.join(calib_dir, 'et_results.pkl'), 'rb'))
    gains = gain_results.get_amp_gains(sensor_id)

    subdir_list = [x.path for x in os.scandir(main_dir) if os.path.isdir(x.path)]

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
        infiles = glob.glob(os.path.join(main_dir, '*{0}_{1}*'.format(xpos, ypos),
                                         '*{0}.fits'.format(sensor_id)))

        if i == 0:
            output_file = os.path.join(output_dir, 
                                       '{0}_{0}_crosstalk_matrix.fits'.format(sensor_id))
            crosstalk_task = CrosstalkTask()
            crosstalk_task.config.output_file = output_file
            crosstalk_task.run(sensor_id, infiles, gains, bias_frame=bias_frame,
                               dark_frame=dark_frame)
        else:
            crosstalk_task = CrosstalkTask()
            crosstalk_task.run(sensor_id, infiles, gains, bias_frame=bias_frame,
                               dark_frame=dark_frame, 
                               crosstalk_matrix_file=output_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str)
    parser.add_argument('main_dir', type=str)
    parser.add_argument('calib_dir', type=str)
    args = parser.parse_args()

    main(args.sensor_id, args.main_dir, args.calib_dir)
        

    

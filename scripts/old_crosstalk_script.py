def main(main_dir, calib_dir, aggressor_id, victim_id=None, output_dir='./', 
         use_multiprocessing=False):

    ## Task configurables
    nsig = 2.0
    num_iter = 10
    threshold = 40000.0

    camera = camMapper._makeCamera()
    lct = LsstCameraTransforms(camera)

    ## Make bias dictionary
    raft_list  = ['R01', 'R02', 
                  'R10', 'R11', 'R12', 
                  'R20', 'R21', 'R22',
                  'R30']
    sensor_list = ['S00', 'S01', 'S02', 'S10', 'S11', 'S12', 'S20', 'S21', 'S22']
    bias_dict = {}
    dark_dict = {}
    gains_dict = {}
    gain_results = pickle.load(open(os.path.join(calib_dir, 'et_results.pkl'), 'rb'))
    for raft in raft_list:
        for sensor in sensor_list:
            sensor_id = '{0}_{1}'.format(raft, sensor)
            try:
                bias_frame = glob.glob(os.path.join(calib_dir, '{0}_superbias.fits'.format(sensor_id)))[0]
            except IndexError:
                bias_frame = None
            try:
                dark_frame = glob.glob(os.path.join(calib_dir, '{0}_superdark.fits'.format(sensor_id)))[0]
            except IndexError:
                dark_frame = None
            gains = gain_results.get_amp_gains(sensor_id)
            gains_dict[sensor_id] = gains
            bias_dict[sensor_id] = bias_frame
            dark_dict[sensor_id] = dark_frame

    ## Eventual command line args
    aggressor_gains = gains_dict[aggressor_id]
    aggressor_bias = bias_dict[aggressor_id]
    aggressor_dark = MaskedCCD(dark_dict[aggressor_id], bias_frame=aggressor_bias)
       
    if victim_id is not None:
        victim_bias = bias_dict[victim_id]
        victim_dark = MaskedCCD(dark_dict[victim_id], bias_frame=victim_bias)
        victim_gains = gains_dict[victim_id]
    else:
        victim_id = aggressor_id
        victim_bias = aggressor_bias
        victim_gains = aggressor_gains
        victim_dark = aggressor_dark

    victim_noise = calculate_noise(victim_bias)

    ## Sort directories by central CCD
    directory_list = [x.path for x in os.scandir(main_dir) if os.path.isdir(x.path)]
    xtalk_dict = {}
    for acquisition_dir in directory_list:
        basename = os.path.basename(acquisition_dir)
        if "xtalk" not in basename:
            continue
        xpos, ypos = basename.split('_')[-4:-2]    
        central_sensor, ccdX, ccdY = lct.focalMmToCcdPixel(float(ypos), float(xpos))
        if central_sensor in xtalk_dict:
            xtalk_dict[central_sensor].add((xpos, ypos))
        else:
            xtalk_dict[central_sensor] = {(xpos, ypos)}
    
    outfile = os.path.join(output_dir, '{0}_{1}_crosstalk_results.fits'.format(aggressor_id, victim_id))

    ## For given aggressor get infiles per position
    for i, pos in enumerate(xtalk_dict[aggressor_id]):
        xpos, ypos = pos

        ## Aggressor ccd files
        aggressor_infiles = glob.glob(os.path.join(main_dir, '*{0}_{1}*'.format(xpos, ypos), 
                                                   '*{0}.fits'.format(aggressor_id)))
        aggressor_ccds = [MaskedCCD(infile, bias_frame=aggressor_bias) for infile in aggressor_infiles]

        ## Victim ccd files
        victim_infiles = glob.glob(os.path.join(main_dir, '*{0}_{1}*'.format(xpos, ypos), 
                            '*{0}.fits'.format(victim_id)))
        victim_ccds = [MaskedCCD(infile, bias_frame=victim_bias) for infile in victim_infiles]

        if i == 0:
            crosstalk_matrix = CrosstalkMatrix(aggressor_id, victim_id=victim_id)
        else:
            crosstalk_matrix = CrosstalkMatrix(aggressor_id, victim_id=victim_id, filename=outfile)

        num_aggressors = 0
        for aggressor_amp in range(1, 17):

            aggressor_imarr = calibrated_stack(aggressor_ccds, aggressor_amp, dark_ccd=aggressor_dark)

            ## smooth and find largest peak
            gf_sigma = 20
            smoothed = gaussian_filter(aggressor_imarr, gf_sigma)
            y, x = np.unravel_index(smoothed.argmax(), smoothed.shape)

            ## check that circle centered on peak above threshold
            r = 20
            Y, X = np.ogrid[-y:smoothed.shape[0]-y, -x:smoothed.shape[1]-x]
            mask = X*X + Y*Y >= r*r
            test = np.ma.MaskedArray(aggressor_imarr, mask)
            if np.mean(test) > threshold:
                num_aggressors += 1

                aggressor_stamp = make_stamp(aggressor_imarr, y, x)*aggressor_gains[aggressor_amp]

                ## Optionally use multiprocessing
                if use_multiprocessing:
                    manager = mp.Manager()
                    row = manager.dict()
                    job = [mp.Process(target=calculate_crosstalk, args=(row, i, victim_ccds, 
                                                                        victim_gains, victim_noise, 
                                                                        aggressor_stamp, y, x, 
                                                                        victim_dark)) for i in range(1, 17)]
                    _ = [p.start() for p in job]
                    _ = [p.join() for p in job]
                else:
                    row = {}
                    for i in range(1, 17):
                        calculate_crosstalk(row, i, victim_ccds, victim_gains, victim_noise, 
                                            aggressor_stamp, y, x, victim_dark)
                crosstalk_matrix.set_row(aggressor_amp, row)

                if num_aggressors == 4: break

        crosstalk_matrix.write_fits(outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('aggressor_id', type=str)
    parser.add_argument('main_dir', type=str)
    parser.add_argument('calib_dir', type=str)
    parser.add_argument('-v', '--victim_id', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default='./')
    args = parser.parse_args()

    aggressor_id = args.aggressor_id
    main_dir = args.main_dir
    calib_dir = args.calib_dir
    victim_id = args.victim_id
    output_dir = args.output_dir

    main(main_dir, calib_dir, aggressor_id,
         victim_id=victim_id, output_dir=output_dir)

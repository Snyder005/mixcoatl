import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
from scipy import stats
from os.path import join
import os
import glob
import errno
import sys
import time
from subprocess import *
import argparse
import pickle as pkl

from lsst.daf.persistence import Butler
from lsst.cp.pipe.makeBrighterFatterKernel import MakeBrighterFatterKernelTask
from lsst.ip.isr.isrTask import IsrTask, IsrTaskConfig
from lsst.ip.isr.isrFunctions import brighterFatterCorrection
from lsst.meas.algorithms import SourceDetectionTask
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig

DETECTOR_NUM = 4 ## number for RTM-002, S11 (ETU1)

def main(sensor_id, spots_butler_dir, eotest_dir=None, output_dir='./', 
         spots_already_done=False):

    if eotest_dir is not None:
        
        ## Create directory for flats results
        flats_output_dir = os.path.join(output_dir, 'flat_pairs_results')
        try:
            os.makedirs(flats_output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        ## Link TS8Mapper
        link = Popen('echo "lsst.obs.lsst.ts8.Ts8Mapper" > {0}/_mapper'.format(flats_output_dir), shell=True)
        Popen.wait(link)

        flat_files = sorted(glob.glob(join(eotest_dir, 'flat_pair_raft_acq', 
                                           'v0', '*', sensor_id, '*flat_*_flat?_*.fits')))
    
        ## Perform flat image ingest
        ingest_args = flats_output_dir
        for flat_file in flat_files:
            ingest_args += " {0}".format(flat_file)
        
        ingest = Popen('ingestImages.py ' + ingest_args, shell=True)
        Popen.wait(ingest)

        flats_butler = Butler(flats_output_dir)
        visits = []
        my_metaData = flats_butler.queryMetadata('raw', ['visit', 'dateObs'])
        for item in my_metaData:
            visits.append(item[0])
        pairs = []

        for fitsFile_1 in flat_files:
            expNum_1 = int(fitsFile_1[-26:-25])
            expTime_1 = int(float(fitsFile_1[-39:-32]) * 1000)
            if expTime_1 <= 35.:
                continue
            fitsVisitNum_1 = int(fitsFile_1[-19:-5])
            if expNum_1 == 1:
                for visit_1 in visits:
                    visitNum_1 = int(visit_1/10)
                    if visitNum_1 != fitsVisitNum_1:
                        continue
                    else:
                        exposure_1 = flats_butler.get('raw', dataId={'visit': visit_1, 'detector': DETECTOR_NUM})                    
                        for fitsFile_2 in flat_files:
                            expNum_2 = int(fitsFile_2[-26:-25])
                            expTime_2 = int(float(fitsFile_2[-39:-32]) * 1000) # expTime in msec
                            fitsVisitNum_2 = int(fitsFile_2[-19:-5])
                            if expNum_2 == 2 and expTime_2 == expTime_1:
                                break
                        for visit_2 in visits:
                            visitNum_2 = int(visit_2/10)
                            if visitNum_2 != fitsVisitNum_2:
                                continue
                            else:
                                exposure_2 = flats_butler.get('raw', dataId={'visit': visit_2, 'detector': DETECTOR_NUM})                                                
                                pairs.append('%s,%s'%(str(visit_1),str(visit_2)))
                                print(expNum_1, expTime_1, expNum_2, expTime_2)
                                ccd = exposure_1.getDetector()
                                for amp in ccd:
                                    img_1 = exposure_1.image
                                    img_2 = exposure_2.image                                
                                    arr_1 = img_1.Factory(img_1, amp.getBBox()).getArray()
                                    arr_2 = img_2.Factory(img_2, amp.getBBox()).getArray()                                
                                    print(amp.getName(), arr_1.mean(), arr_2.mean())
                                break
                    break

        bf_args = [flats_output_dir, '--rerun', 'test','--id', 'detector={0}'.format(DETECTOR_NUM),'--visit-pairs']
        for pair in pairs:
            bf_args.append(str(pair))

        bf_args = bf_args + ['-c','xcorrCheckRejectLevel=2', 'doCalcGains=True', 'level="AMP"', 'biasCorr=1.0',
                             '--clobber-config', '--clobber-versions']
        command_line = 'makeBrighterFatterKernel.py ' + ' '.join(bf_args)
        corr_struct = MakeBrighterFatterKernelTask.parseAndRun(args=bf_args)
    else:
        flats_output_dir = os.path.join(output_dir, 'flat_pairs_results')

    return

    flats_butler = Butler(os.path.join(flats_output_dir, 'rerun', 'test'))
    bf_kernel = flats_butler.get('brighterFatterKernel', dataId={'raftName': 'RTM-002', 'detectorName': sensor_id, 'detector': DETECTOR_NUM})
    gain_data = flats_butler.get('brighterFatterGain', dataId={'raftName': 'RTM-002', 'detectorName': sensor_id, 'detector': DETECTOR_NUM})

    # Now we shift to the spots data
    # These setup the image characterization and ISR
    isrConfig = IsrTaskConfig()
    isrConfig.doBias = False
    isrConfig.doDark = False
    isrConfig.doFlat = False
    isrConfig.doFringe = False
    isrConfig.doDefect = False
    isrConfig.doAddDistortionModel = False
    isrConfig.doWrite = True
    isrConfig.doAssembleCcd = True
    isrConfig.expectWcs = False
    isrConfig.doLinearize = False

    charConfig = CharacterizeImageConfig()
#    charConfig.installSimplePsf.fwhm = 0.05
    charConfig.doMeasurePsf = False
    charConfig.doApCorr = False
    charConfig.doDeblend = False
    charConfig.repair.doCosmicRay = False
    charConfig.detection.background.binSize = 10
    charConfig.detection.minPixels = 10
    charConfig.detection.thresholdType = "stdev"
    charConfig.detection.thresholdValue = 5

    if not spots_already_done:

        spots_butler = Butler(spots_butler_dir)

        visits = []
        my_metaData = spots_butler.queryMetadata('raw', ['visit', 'dateObs'])
        test_metaData = spots_butler.queryMetadata('raw', ['detector'])
        detectors = [item for item in test_metaData]

        for item in my_metaData:
            if item[0] > 2019060800684:
                visits.append(item[0])

        byamp_results = []
        byamp_corrected_results = []
        for visit in visits[70:75]:
            print("Getting exposure # %d"%visit)
            sys.stdout.flush()
            exposure = spots_butler.get('raw', dataId={'visit': visit, 'detector': 94})
            # Perform the instrument signature removal (mainly assembling the CCD)
            isrTask = IsrTask(config=isrConfig)
            exposure_isr = isrTask.run(exposure).exposure
            # For now, we're applying the gain manually
            ccd = exposure_isr.getDetector()
            for do_bf_corr in [False, True]:
                exposure_copy=exposure_isr.clone()            
                for amp in ccd:
                    gain = gain_data[amp.getName()]
                    img = exposure_copy.image
                    sim = img.Factory(img, amp.getBBox())
#                    sim *= gain
                    print(amp.getName(), gain, amp.getBBox())
                    sys.stdout.flush()                    
                    if do_bf_corr:
                        brighterFatterCorrection(exposure_copy[amp.getBBox()],bf_kernel.kernel[amp.getName()],20,10,False)

                # Now find and characterize the spots
                charTask = CharacterizeImageTask(config=charConfig)
                tstart=time.time()
                charResult = charTask.run(exposure_copy)
                spotCatalog = charResult.sourceCat
                print("%s, Correction = %r, Characterization took "%(amp.getName(),do_bf_corr),str(time.time()-tstart)[:4]," seconds")
                sys.stdout.flush()
                select = ((spotCatalog['base_SdssShape_xx'] >= 1.0) & (spotCatalog['base_SdssShape_xx'] <= 10.0) & 
                          (spotCatalog['base_SdssShape_yy'] >= 1.0) & (spotCatalog['base_SdssShape_yy'] <= 10.0))
                spotCatalog  = spotCatalog.subset(select)                                

                x2 = spotCatalog['base_SdssShape_xx']
                y2 = spotCatalog['base_SdssShape_yy']
                flux = spotCatalog['base_SdssShape_instFlux']
                numspots = len(flux)
                print("Detected ",len(spotCatalog)," objects, Flux = %f, X2 = %f, Y2 = %f"%(np.nanmean(flux),np.nanmean(x2),np.nanmean(y2)))
                sys.stdout.flush()                                
                if do_bf_corr:
                    byamp_corrected_results.append([numspots, np.nanmean(flux), np.nanstd(flux), np.nanmean(x2), np.nanstd(x2),
                                       np.nanmean(y2), np.nanstd(y2)])
                else:
                    byamp_results.append([numspots, np.nanmean(flux), np.nanstd(flux), np.nanmean(x2), np.nanstd(x2),
                                       np.nanmean(y2), np.nanstd(y2)])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_id', type=str, help='Sensor ID (e.g. S00)')
    parser.add_argument('spots_butler_dir', type=str, help='Directory for ingested spots images.')
    parser.add_argument('--eotest_dir', type=str, default=None, help='Path to eotest directory')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    parser.add_argument('--spots_already_done', action='store_true')
    args = parser.parse_args()

    sensor_id = args.sensor_id
    spots_butler_dir = args.spots_butler_dir
    eotest_dir = args.eotest_dir
    output_dir = args.output_dir
    spots_already_done = args.spots_already_done

    main(sensor_id, spots_butler_dir, eotest_dir, 
         output_dir=output_dir, 
         spots_already_done=spots_already_done)

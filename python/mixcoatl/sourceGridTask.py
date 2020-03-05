import glob
import numpy as np
from scipy.spatial import distance
from astropy.io import fits

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.obs.lsst import LsstCamMapper as camMapper
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms


camera = camMapper._makeCamera()
lct = LsstCameraTransforms(camera)

class GridFitConfig(pexConfig.Config):
    """Configuration for GridFitTask."""

    max_displacement = pexConfig.Field("Maximum distance (pixels) between matched sources.",
                                       float, default=10.)
    grid_size = pexConfig.Field("Number of grid columns/rows.",
                                int, default=49)
    output_dir = pexConfig.Field("Output directory", str, default=".")

class GridFitTask(pipeBase.Task):
    """Task to perform source grid fit."""
    ConfigClass = GridFitConfig
    _DefaultName = "GridFitTask"

    @pipeBase.timeMethod
    def run(self, infile):

        ## Below needs to be hand tuned for now!
        basename = os.path.basename(infile)
        projector_y = float(basename.split('_')[-1][:-5]) # these are camera x/y coords
        projector_x = float(basename.split('_')[-2][:-1])

        ccd_name, ccd_x, ccd_y = lct.focalMmToCcdPixel(projector_y, projector_x)

        x_guess = 2*509*4. - ccd_x - 27.0
        y_guess = ccd_y - 67.0
        ## above needs to be hand tuned for now!

        src = fits.getdata(infile)
        model_grid = SourceGrid.from_source_catalog(src, y_guess=y_guess, x_guess=x_guess,
                                                    mean_func = np.nanmedian)
        nrows = ncols = self.config.grid_size
        gY, gX = model_grid.make_grid(nrows=nrows, ncols=ncols)

        indices, distances = coordinate_distances(gY, gX, srcY, srcX)
        nn_indices = indices[:, 0]
        dx_array = srcX[nn_indices]-gX
        dy_array = srcY[nn_indices]-gY
        xx_array = srcXX[nn_indices]
        yy_array = srcYY[nn_indices]
        flux_array = srcF[nn_indices]

        max_displacement = self.config.max_displacement
        bad_indices = np.hypot(dx_array, dy_array) >= self.max_displacement
        dx_array[bad_indices] = np.nan
        dy_array[indices] = np.nan
        xx_array[indices] = np.nan
        yy_array[indices] = np.nan
        flux_array[indices] = np.nan

        ## Output FITs construction
        hdr = fits.Header()
        hdr['X0'] = model_grid.x0
        hdr['Y0'] = model_grid.y0
        hdr['DX'] = model_grid.dx
        hdr['DY'] = model_grid.dy
        hdr['THETA'] = model_grid.theta

        columns = [fits.Column('DATA_DX', array=dx_array, format='D'),
                   fits.Column('DATA_DY', array=dy_array, format='D'),
                   fits.Column('MODEL_X', array=gX, format='D'),
                   fits.Column('MODEL_Y', array=gY, format='D'),
                   fits.Column('DATA_XX', array=xx_array, format='D'),
                   fits.Column('DATA_YY', array=yy_array, format='D'),
                   fits.Column('DATA_FLUX', array=flux_array, format='D')]

        hdu = fits.BinTableHDU.from_columns(columns, header=hdr)
        output_dir = self.config.output_dir
        outfile = os.path.join(output_dir, '{0}_raw_displacement_data.fits'.format(os.path.splitext(basename)[0]))
        hdu.writeto(outfile, overwrite=True)

def main(catalog_list, num_processes, output_dir='./', max_displacement=10.0):

    spot_fitter = GridFitTask(max_displacement=max_displacement, output_dir=output_dir)
    
    num_processes = max(1, min(num_processes, mp.cpu_count()-1))
    pool = mp.Pool(num_processes)
    pool.map(spot_fitter.run, catalog_list)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('catalog_list', nargs='+', 
                        help='Source catalog files to analyze.')
    parser.add_argument('-m', '--max_displacement', type=float, default=10.0, 
                        help='Maximum allowed displacement threshold.')
    parser.add_argument('-n', '--num_processes', type=int, default=1,
                        help='Number of parallel processes to spawn')
    parser.add_argument('--output_dir', '-o', type=str, default='./',
                        help='Output directory to write result files to.')
    args = parser.parse_args()

    catalog_list = args.catalog_list
    output_dir = args.output_dir
    max_displacement = args.max_displacement
    num_processes = args.num_processes

    main(catalog_list, num_processes=num_processes, 
         output_dir=output_dir, max_displacement=max_displacement)

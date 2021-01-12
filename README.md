# MixCOATL

Non-standard analysis tasks and scripts for processing LSST Camera images taken in the Bench for Optical Testing (BOT) during the integration and testing period. The Mixed Calibration Optics Analysis Test Library, or MixCOATL module was developed to perform the following analysis tasks:

* Measurement of electronic crosstalk using a number of methodologies.
* Organization of electronic crosstalk results within an SQL database framework.
* Handling of source catalogs generated for BOT images taken using a "fake star field".

## Dependencies

The MixCOATL module is being developed concurrently to use the latest version of the LSST Science Pipeline DM Stack. In addition, MixCOATL requires the installation of the `eotest` module (<https://github.com/lsst-camera-dh/eotest>), which can be installed using the `release` repository (<https://github.com/lsst-camera-dh/release>). For information on how to install the LSST Science Pipeline, visit: <https://pipelines.lsst.io/>

## Package Description

The package is divided into separate sections, described as follows:

* `mixcoatl/python`: This directory is the location of the Python module files. To use the MixCOATL package, this directory should be added to your Python path (e.g. `PYTHONPATH`).
* `mixcoatl/notebooks`: This directory contains a number of Jupyter notebooks used for demonstrating of the MixCOATL features or new feature development.
* `mixcoatl/scripts`: This directory contains a number of "convenience" Python scripts that can be used to run the MixCOATL analysis tasks, or as reference for writing custom analysis scripts.

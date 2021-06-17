# MixCOATL

Non-standard analysis tasks and scripts for processing LSST Camera images taken in the Bench for Optical Testing (BOT) during the integration and testing period. The Mixed Calibration Optics Analysis Test Library, or MixCOATL module was developed to perform the following analysis tasks:

* Measurement of electronic crosstalk from images of projected objects and hot columns using a best fit model calculation.
* Analysis of BOT images of a projected grid of "star-like" objects to study sensor level distortions of source position, shape, and flux.

## Dependencies

The MixCOATL module is being developed concurrently to use the latest version of the LSST Science Pipeline DM Stack, and should require no additional dependencies or modules outside of those include in `lsst_distrib` selection of packages in the DM Stack setup.  For information on how to install the LSST Science Pipeline, visit: <https://pipelines.lsst.io/>

## Package Description

The package is divided into separate sections, described as follows:

* `mixcoatl/python`: This directory is the location of the Python module files. To use the MixCOATL package, this directory should be added to your Python path (e.g. `PYTHONPATH`).
* `mixcoatl/notebooks`: This directory contains a number of Jupyter notebooks used for demonstrating of the MixCOATL features or new feature development.
* `mixcoatl/pipelines`: This directory contains the YAML files used to specify MixCOATL analyses within the "pipeline task" formulation of the DM Stack that uses the Gen 3 Butler to interface with a repository of datasets.

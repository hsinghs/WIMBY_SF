Metadata
========

Project
-------

Owner: Andrea Hahmann <ahah@dtu.dk>
Project: WIMBY
Project Partner: DTU
Date: 2023-12-11

Dataset
-------

Author: DTU Wind
DOI: n/a
URL: n/a
License: CC-BY
Title: Mean wind speed by height, month, and time of the day
Version: 2.0.0a
Geographic_Area: Europe
Temporal_Coverage: 2013-2022 (ten-year average)



General
=======
This readme.txt file briefly describes the contents of the ERA5 mean 12 x 24 dataset.


Folder Structure
================

* D1.1_wind_resource_data
	* ERA5_month_by_hour


Folder contents
===============
The folder contains two files: README and data file. 


Software version required
=========================
netCDF files can be opened in many applications (MATLAB, Python). The use of Xarray facilitates manipulations. 



Formats
=======
Data is written in netCDF format. 


Abbreviations
=============
ERA5: ECMWF reanalysis version 5
ECMWF: European Centre for Medium-Range Weather Forecasting
netCDF: 


Codebook
=========
The data were created by averaging wind speeds at three heights [50, 100 and 200 m] over the month and hour of the day for the period (2013-2022). We select an area containing Europe from 35N-72N and 15W-35E. The resulting matrix has dimensions of (12,24,2,149,201). The horizontal grid spacing is 0.25 degrees by 0.25 degrees. 

We recommend that this data be biased corrected using the ratio_gwa2_era5.nc dataset for application models. 





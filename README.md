Project: This package is for calculating shadow flicker impacts from wind turbines.

Steps to run the package:

Import shadow_flicker.py and shadow_flicker_wind.py from shdowflicker package.

Download the era5_wind data file "era5_mean_2013-2022_month_by_hour.nc", and the correction file "ratio_gwa2_era5/ratio_gwa2_era5.nc".

The Copernicus DEM GeoTIFFs (for Europe) at 30 m is needed to run the code.

Open a new Python script. Run the example code, with the adjusted input data.

GDAL with Python bindings is required for viewshed generation.
Verify installation with:

python -c "from osgeo import gdal; print(gdal.VersionInfo())"
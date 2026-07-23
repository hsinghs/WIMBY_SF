# -*- coding: utf-8 -*-
"""
Last updated 22/07/2026
Version: 3.0

This is the code for the shadow flicker (SF) calculator of the Horizon Europe
WIMBY project. It calculates SF frequency for one or multiple turbines under a
worst case scenario (turbines that are always operating and a clear atmosphere).
The outputs include i) a 3D xarray array (lat, lon, time) of shadow footprints,
that take into consideration the terrain and 2) a 2D xarray array (lat, lon)
with the cumulated SF and reclisfied to show areas with more than 30h/a and
areas with less than 30h/a. For details about the algorithm please refer to
 WIMBY's deliverable D2.4 "Maps of health and safety impact metrics (b)"
under https://wimby.eu/resources/

@authors: Hsing-Hsuan Chen and Luis Ramirez Camargo
"""

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import pvlib
import rasterio
import rioxarray
from shapely.geometry import Point
from shapely.affinity import scale, rotate, translate
from timezonefinder import TimezoneFinder
from pyproj import Transformer
import pytz
from osgeo import gdal
from joblib import Parallel, delayed
from rasterio.features import rasterize
from shapely.geometry import mapping
import xarray as xr
import os
from functools import lru_cache

###############################################################################
# 1) Create DSM as well as the viewshed for the surrounding area of a turbine
# in target epsg for a particular buffer and resolution.
###############################################################################
def create_base_dsm_and_viewshed(
    lat,
    lon,
    input_dsm_path,
    overground_height_high=150,
    buffer_meters=5000.0,
    resolution_meters=20.0,
    target_epsg="3035",
):
    """
    Create a reprojected Digital Surface Model (DSM) and compute a viewshed
    raster around a wind turbine location using GDAL in-memory processing.

    This function performs the following steps:
    1. Transforms the turbine's WGS84 coordinates (lat, lon) to a target EPSG.
    2. Defines a square buffer (e.g., 5x5 km) around the turbine in projected
    coordinates.
    3. Clips and resamples the input DSM using GDAL Warp to match the specified
    extent and resolution.
    4. Computes the viewshed based on line-of-sight visibility from the turbine
    hub height using GDAL ViewshedGenerate.

    Parameters
    ----------
    lat : float
        Latitude of the wind turbine in EPSG:4326 (WGS84).
    lon : float
        Longitude of the wind turbine in EPSG:4326 (WGS84).
    input_dsm_path : str
        File path to the input DSM (GeoTIFF or similar format) in EPSG:4326
    overground_height_high : float, default=150
        hub height + rotor radius in meters
    buffer_meters : float, default=5000.0
        Radius of the square buffer around the turbine (in meters).
        The output DSM will have size (2 × buffer).
    resolution_meters : float, default=20.0
        Desired resolution (pixel size) for the output DSM in meters.
        The suggested resolution is 20m
    target_epsg : str, default="3035"
        EPSG code of the target projected coordinate system
        The tool has been thorouhgly tested with EPSG:"3035" for LAEA Europe

    Returns
    -------
    base_dsm : osgeo.gdal.Dataset
        In-memory GDAL dataset of the reprojected and clipped DSM.
    viewshed_gdal : osgeo.gdal.Dataset
        In-memory GDAL dataset representing the viewshed raster:
        - 1 = visible
        - 0 = not visible
        - 0.0 = out of range or no-data

    Notes
    -----
    - Viewshed uses Earth's curvature correction (`dfCurvCoeff=0.85714`).
    - The `TARGET_HEIGHT`, the height of the target above the DEM surface in
      the height unit of the DEM, is fixed at 2 meters (resembling the
      elevation of windows in buildings).
    """
    TARGET_HEIGHT = 2
    # Set up transformer: from EPSG:4326 to EPSG:3035
    transformer_4326_to_target_epsg = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{target_epsg}", always_xy=True
    )
    wt_x, wt_y = transformer_4326_to_target_epsg.transform(lon, lat)

    # 5x5 km bounding box around the turbine location
    min_x = wt_x - buffer_meters
    max_x = wt_x + buffer_meters
    min_y = wt_y - buffer_meters
    max_y = wt_y + buffer_meters

    # Clip and resample DSM using GDAL Warp (in memory)
    base_dsm = gdal.Warp(
        "/vsimem/temp.tif",
        input_dsm_path,
        dstSRS=f"EPSG:{target_epsg}",
        outputBounds=[min_x, min_y, max_x, max_y],
        xRes=resolution_meters,
        yRes=resolution_meters,
        resampleAlg="bilinear",
    )

    # Compute viewshed from turbine location
    viewshed_gdal = gdal.ViewshedGenerate(
        srcBand=base_dsm.GetRasterBand(1),
        driverName="MEM",
        targetRasterName="/vsimem/temp_viewshed.tif",
        creationOptions=[],
        observerX=wt_x,
        observerY=wt_y,
        observerHeight=overground_height_high,
        targetHeight=TARGET_HEIGHT,
        maxDistance=buffer_meters * 2,
        dfCurvCoeff=0.85714,
        visibleVal=1,
        invisibleVal=0,
        outOfRangeVal=0.0,
        noDataVal=0.0,
        mode=1,
    )

    return base_dsm, viewshed_gdal


###############################################################################
# 2) ANGLE MAPS GENERATION (ALTITUDE & AZIMUTH) in target EPSG (3035 for Europe)
###############################################################################
def get_wt_elevation_and_location(
    base_dsm, lat, lon, overground_height_high, overground_height_low
):
    """
    Compute the turbine's ground elevation and above-ground total heights
    by extracting values from the `base_dsm` in the target EPSG.

    This function:
    1. Converts the turbine's (lat, lon) coordinates in EPSG:4326 (WGS84)
       into the projected coordinate system used by `base_dsm`.
    2. Extracts the ground elevation at the turbine's projected location
       from the `base_dsm`.
    3. Calculates two turbine heights (to rotor top and rotor
       bottom) by adding hub-based overground offsets to the extracted ground
    elevation.
    4. Returns both the heights and the turbine's location in projected
       coordinates.

    Parameters
    ----------
    base_dsm : osgeo.gdal.Dataset
        A GDAL raster dataset (in-memory) created with the function
        create_base_dsm_and_viewshed
    lat : float
        Latitude of the wind turbine (WGS84, EPSG:4326).
    lon : float
        Longitude of the wind turbine (WGS84, EPSG:4326).
    overground_height_high : float, default=150
        hub height + rotor radius in meters
    overground_height_low : float
        hub height - rotor radius in meters

    Returns
    -------
    wt_ground_elevation : float
        Ground elevation (in meters) at the turbine location, extracted from
        the `base_dsm`.
    total_height_high : float
        Ground elevation plus `overground_height_high`.
    total_height_low : float
        Ground elevation plus `overground_height_low`.
    wt_x : float
        X-coordinate of the turbine in the DSM's projected CRS.
    wt_y : float
        Y-coordinate of the turbine in the DSM's projected CRS.
    """
    # Transform the turbine lat/lon to target_epsg
    base_dsm_proj = base_dsm.GetProjection()
    transformer_4326_to_target_epsg = Transformer.from_crs(
        "EPSG:4326", base_dsm_proj, always_xy=True
    )
    wt_x, wt_y = transformer_4326_to_target_epsg.transform(lon, lat)
    geotransform = base_dsm.GetGeoTransform()
    dsm_data = base_dsm.GetRasterBand(1).ReadAsArray()

    # Get ground elevation at turbine
    col = int((wt_x - geotransform[0]) / geotransform[1])
    row = int((wt_y - geotransform[3]) / geotransform[5])
    wt_ground_elevation = dsm_data[row, col]
    total_height_high = wt_ground_elevation + overground_height_high
    total_height_low = wt_ground_elevation + overground_height_low
    return wt_ground_elevation, total_height_high, total_height_low, wt_x, wt_y


def calculate_angle_maps(
    base_dsm, lat, lon, total_height_high, total_height_low
):
    """
    Calculate maps that are necesary to determine the shadow footprint in a
    complex terrain. This function creates angle altitude maps and azimuth
    maps from the perspective of the turbine on every single pixel in a
    raster of the same extent as the 'base_dsm'.
    It returns both raw NumPy arrays and GDAL in-memory datasets.
    The NumPy arrays are needed to calcultate shadow_ring, the possible SF
    area for complex terrain.

    Parameters
    ----------
    base_dsm : osgeo.gdal.Dataset
        A GDAL raster dataset (in-memory or on disk) representing the DSM in
        EPSG:3035.
    lat : float
        Latitude of the wind turbine (WGS84, EPSG:4326).
    lon : float
        Longitude of the wind turbine (WGS84, EPSG:4326).
    total_height_high : float
        Vertical offset (in meters) to be added to ground elevation to compute
        the "rotor top" point (hub height + rotor radius).
    total_height_low : float
        Vertical offset (in meters) to be added to ground elevation to compute
        the "rotot bottom" point (e.g., hub height - rotor radius).

    Returns
    -------
    azimuths_array : np.ndarray of shape (H, W), Height and Width of the raster
        Azimuth angle in degrees (0–360), computed from each raster cell toward the turbine.
    altitude_angle_high_array : np.ndarray of shape (H, W)
        Altitude angle in degrees from each cell to the turbine's rotor top.
    altitude_angle_low_array : np.ndarray of shape (H, W)
        Altitude angle in degrees from each cell to the turbine's rotor bottom.
    distance_array : np.ndarray of shape (H, W)
        Euclidean distance (in meters) from each raster cell to the turbine.
    altitude_angle_high_map : gdal.Dataset
        GDAL in-memory raster of `altitude_angle_high_array`, useful for debugging or export.
    altitude_angle_low_map : gdal.Dataset
        GDAL in-memory raster of `altitude_angle_low_array`, useful for debugging or export.
    azimuths_map : gdal.Dataset
        GDAL in-memory raster of `azimuths_array`.
    distance_map : gdal.Dataset
        GDAL in-memory raster of `distance_array`.
    geotransform : tuple
        Affine geotransform associated with the `base_dsm`.

    Raises
    ------
    ValueError
        If the turbine's projected location falls outside the raster bounds of the DSM.

    Notes
    -----
    - The returned rasters are aligned spatially with the DSM and maintain the same extent/resolution.
    - Azimuth angles follow the meteorological convention (0° = North, 90° = East).
    """
    # Transform the turbine lat/lon to target_epsg
    base_dsm_proj = base_dsm.GetProjection()
    transformer_4326_to_target_epsg = Transformer.from_crs(
        "EPSG:4326", base_dsm_proj, always_xy=True
    )
    wt_x, wt_y = transformer_4326_to_target_epsg.transform(lon, lat)
    geotransform = base_dsm.GetGeoTransform()
    dsm_data = base_dsm.GetRasterBand(1).ReadAsArray()

    # 1) Build common elements of the azimuth and altitude arrays
    rows, cols = dsm_data.shape
    # Raster coordinate arrays
    j_array, i_array = np.meshgrid(np.arange(cols), np.arange(rows))
    x_array = (
        geotransform[0] + j_array * geotransform[1] + i_array * geotransform[2]
    )
    y_array = (
        geotransform[3] + j_array * geotransform[4] + i_array * geotransform[5]
    )

    dx = wt_x - x_array
    dy = wt_y - y_array
    distance = np.sqrt(dx**2 + dy**2)
    # Avoid divide-by-zero
    distance = np.where(distance == 0, np.finfo(float).eps, distance)

    # 3) Build azimuth angle array
    azimuths_array = np.zeros((rows, cols), dtype=np.float32)
    azimuth_angle = np.degrees(np.arctan2(dx, dy))
    azimuth_angle[azimuth_angle < 0] += 360.0
    azimuths_array = azimuth_angle.astype(np.float32)

    # Create in-memory GDAL azymuth map
    driver = gdal.GetDriverByName("MEM")  # In-memory dataset
    azimuths_map = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
    azimuths_map.SetGeoTransform(geotransform)
    azimuths_map.SetProjection(base_dsm_proj)
    azimuths_map.GetRasterBand(1).WriteArray(azimuths_array)

    # 4) Build altitude angle arrays towards turbine rotor top and bottom
    # Current ground elevations
    dz_high = total_height_high - dsm_data
    dz_low = total_height_low - dsm_data

    altitude_angle_high_array = np.degrees(
        np.arctan2(dz_high, distance)
    ).astype(np.float32)
    altitude_angle_low_array = np.degrees(np.arctan2(dz_low, distance)).astype(
        np.float32
    )

    # Create in-memory GDAL altitude angle high and low maps

    altitude_angle_high_map = driver.Create(
        "", cols, rows, 1, gdal.GDT_Float32
    )
    altitude_angle_high_map.SetGeoTransform(geotransform)
    altitude_angle_high_map.SetProjection(base_dsm_proj)
    altitude_angle_high_map.GetRasterBand(1).WriteArray(
        altitude_angle_high_array
    )

    altitude_angle_low_map = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
    altitude_angle_low_map.SetGeoTransform(geotransform)
    altitude_angle_low_map.SetProjection(base_dsm_proj)
    altitude_angle_low_map.GetRasterBand(1).WriteArray(
        altitude_angle_low_array
    )

    # 5) create the distance array

    distance_array = np.sqrt(
        (x_array - wt_x) ** 2 + (y_array - wt_y) ** 2
    ).astype(np.float32)
    # Create in-memory GDAL distance map
    distance_map = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
    distance_map.SetGeoTransform(geotransform)
    distance_map.SetProjection(base_dsm_proj)
    distance_map.GetRasterBand(1).WriteArray(distance_array)

    return (
        azimuths_array,
        altitude_angle_high_array,
        altitude_angle_low_array,
        distance_array,
        altitude_angle_high_map,
        altitude_angle_low_map,
        azimuths_map,
        distance_map,
        geotransform,
    )


###############################################################################
# 3) SHADOW FOOTPRINT CALCULATION FOR ONE SINGEL TIME STEP
###############################################################################


def create_circle_geodataframe_target_epsg(
    lat, lon, ROTOR_DIAMETER, target_epsg
):
    """
    Create a circular turbine rotor geometry centered at the given WGS84
    location, reprojected to the specified EPSG coordinate system.

    Parameters
    ----------
    lat : float
        Latitude of the turbine (EPSG:4326).
    lon : float
        Longitude of the turbine (EPSG:4326).
    ROTOR_DIAMETER : float
        Rotor diameter in meters.
    target_epsg : str or int
        EPSG code of the target coordinate system, e.g., '3035' for Europe.


    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with a single circular polygon representing the
        turbine rotor in the target projection.
    """
    ROTOR_COEF = 0.9
    # Excluding the rotor tip that does not generate shadows.

    gdf_wgs84 = gpd.GeoDataFrame(
        [{"geometry": Point(lon, lat)}], crs="EPSG:4326"
    )
    gdf_turbine = gdf_wgs84.to_crs(f"EPSG:{target_epsg}")

    circle_polygon_laea = gdf_turbine.geometry.buffer(
        ROTOR_DIAMETER * ROTOR_COEF / 2.0
    )
    gdf_turbine["geometry"] = circle_polygon_laea
    return gdf_turbine



def calculate_solar_positions(
    lat,
    lon,
    start_datetime,
    end_datetime,
    total_height_high,
    days_resolution=1,
    hours_resolution=1,
    minutes_resolution=1,
):
    """
    Calculate solar positions (azimuth and altitude) for a given turbine
    location and time range.

    Parameters
    ----------
    lat : float
        Latitude of the turbine in EPSG:4326.
    lon : float
        Longitude of the turbine in EPSG:4326.
    start_datetime : datetime
        Start of the simulation period.
    end_datetime : datetime
        End of the simulation period.
    total_height_high : float
        Total observer height (ground elevation + hub + rotor tip).
    days_resolution : int, optional
        Step size in days (default is 1).
    hours_resolution : int, optional
        Step size in hours (default is 1).
    minutes_resolution : int, optional
        Step size in minutes (default is 1).

    Returns
    -------
    solar_positions: pandas.DataFrame
        DataFrame indexed by UTC datetime with columns:
        - 'sun_azimuth'
        - 'sun_altitude_top'
    """

    total_days = (end_datetime - start_datetime).days + 1
    days = np.arange(0, total_days, days_resolution)
    hours = np.arange(0, 24, hours_resolution)
    minutes = np.arange(0, 60, minutes_resolution)

    days_grid, hours_grid, minutes_grid = np.meshgrid(
        days, hours, minutes, indexing="ij"
    )
    datetimes = (
        pd.to_datetime(start_datetime)
        + pd.to_timedelta(days_grid.flatten(), unit="D")
        + pd.to_timedelta(hours_grid.flatten(), unit="h")
        + pd.to_timedelta(minutes_grid.flatten(), unit="min")
    )

    datetimes_utc = datetimes.tz_localize("UTC")

    altitude_for_solar = round(total_height_high, 2)
    solar_positions = pvlib.solarposition.pyephem(
        time=datetimes_utc,
        latitude=lat,
        longitude=lon,
        altitude=altitude_for_solar,
        pressure=1013.25,
        temperature=20,
    )
    return solar_positions

def calculate_middle_point_shadow_ring(
    shadow_distance_map,
    shadow_ring,
    geotransform,
    turbine_x,
    turbine_y,
    shadow_direction_deg,
):
    """
    Calculate the midpoint along the shadow direction behind a wind turbine
    for shadow ring estimation.

    This function identifies the farthest aligned point within the shadow
    distance map along the specified shadow direction (in degrees). It then
    traces backward towards the turbine until encountering a zero-value pixel
    in `shadow_ring`, indicating the end of the shadow region. The midpoint
    between this stopping point and the farthest point
    is computed as the penumbra center.

    Parameters:
        shadow_distance_map (np.ndarray): 2D array representing the distance of
        the shadow.
        shadow_ring (np.ndarray): 2D binary map indicating where the shadow
                           features may exist (0 means no-shadow area).
        geotransform (tuple): Affine transformation parameters for spatial
        reference
                              (origin_x, pixel_width, rotation_x, origin_y,
                               rotation_y, pixel_height).
        turbine_x (float): X-coordinate of the turbine location in map
                            coordinates.
        turbine_y (float): Y-coordinate of the turbine location in map
                            coordinates.
        shadow_direction_deg (float): Direction of the shadow (degrees).

    Returns:
        tuple:
            - dist_mid (float): Distance from the turbine to the computed
                                midpoint.
            - (mid_x, mid_y) (tuple of float): Coordinates of the computed
                                                midpoint.
              Returns (None, None) if no valid aligned point is found.
    """

    # Retrieve the shape of the shadow map
    rows, cols = shadow_distance_map.shape

    # Convert shadow direction from degrees to radians and compute direction vector
    dx = math.cos(math.radians(shadow_direction_deg))
    dy = math.sin(math.radians(shadow_direction_deg))

    # Generate grid of map coordinates based on geotransformation
    j_indices, i_indices = np.meshgrid(np.arange(cols), np.arange(rows))
    x_grid = (
        geotransform[0]
        + j_indices * geotransform[1]
        + i_indices * geotransform[2]
    )
    y_grid = (
        geotransform[3]
        + j_indices * geotransform[4]
        + i_indices * geotransform[5]
    )

    # Identify valid cells where the shadow is present
    valid_mask = shadow_distance_map > 0
    if not np.any(valid_mask):
        return None, None  # No valid shadow area found

    valid_x = x_grid[valid_mask]
    valid_y = y_grid[valid_mask]

    # Compute distance from turbine to valid points
    dist = np.sqrt((valid_x - turbine_x) ** 2 + (valid_y - turbine_y) ** 2)

    # Compute normalized vectors from turbine to valid points
    vx = valid_x - turbine_x
    vy = valid_y - turbine_y
    vx_n = vx / dist
    vy_n = vy / dist

    # Compute dot products to find points aligned with the shadow direction
    dot_products = vx_n * dx + vy_n * dy
    ALIGNMENT_ANGLE = 0.01
    aligned_mask = np.abs(dot_products - 1) < ALIGNMENT_ANGLE
    # Threshold for alignment

    if not np.any(aligned_mask):
        return None, None  # No aligned points found

    # Extract aligned points and their distances
    aligned_x = valid_x[aligned_mask]
    aligned_y = valid_y[aligned_mask]
    aligned_dist = dist[aligned_mask]

    # Identify the farthest aligned point along the shadow direction
    max_idx = np.argmax(aligned_dist)
    farthest_x, farthest_y = aligned_x[max_idx], aligned_y[max_idx]

    # Compute the vector from the farthest point back to the turbine
    rx = turbine_x - farthest_x
    ry = turbine_y - farthest_y
    dist_back = math.sqrt(rx**2 + ry**2)

    if dist_back == 0:
        return (
            None,
            None,
        )  # Edge case where the farthest point is the turbine itself

    # Normalize the return vector
    rx_n = rx / dist_back
    ry_n = ry / dist_back

    # Set step size based on half the pixel width and compute the number of steps
    step_size = abs(geotransform[1]) / 2.0
    step_count = int(dist_back / step_size) + 2
    step_offsets = np.arange(step_count) * step_size

    # Generate coordinates along the reverse path from the farthest point back to the turbine
    check_x = farthest_x + step_offsets * rx_n
    check_y = farthest_y + step_offsets * ry_n

    # Convert map coordinates to raster indices
    col_indices = np.round(
        (check_x - geotransform[0]) / geotransform[1]
    ).astype(int)
    row_indices = np.round(
        (check_y - geotransform[3]) / geotransform[5]
    ).astype(int)

    # Filter out indices that fall outside the map boundaries
    in_bounds = (
        (row_indices >= 0)
        & (row_indices < rows)
        & (col_indices >= 0)
        & (col_indices < cols)
    )
    col_indices = col_indices[in_bounds]
    row_indices = row_indices[in_bounds]
    check_x = check_x[in_bounds]
    check_y = check_y[in_bounds]

    # Identify the first occurrence of zero in shadow_ring, indicating the shadow border
    zero_mask = shadow_ring[row_indices, col_indices] == 0
    if np.any(zero_mask):
        first_zero_idx = np.argmax(zero_mask)
        red_x, red_y = check_x[first_zero_idx], check_y[first_zero_idx]
        # red_x and red_y = Coordinates of the first obstacle point (where shadow_ring == 0)
        # on the return path
    else:
        # If no zero is found, fallback to the farthest point
        red_x, red_y = farthest_x, farthest_y

    # Compute the midpoint between the farthest aligned point and the detected boundary
    mid_x = (farthest_x + red_x) / 2.0
    mid_y = (farthest_y + red_y) / 2.0
    dist_mid = math.sqrt((mid_x - turbine_x) ** 2 + (mid_y - turbine_y) ** 2)

    return dist_mid, (mid_x, mid_y)


def calculate_shadow_footprint(
    gdf_turbine,
    max_shadow_length,
    line_length,
    solar_position_row,
    sf_multiplier,
    ROTOR_DIAMETER,
    target_epsg,
    shadow_ring,
    geotransform,
    wt_x,
    wt_y,
    shadow_distance_map,
):
    """
    Project the turbine rotor geometry into a penumbra ellipse representing the shadow
    footprint for a single time step.

    Parameters
    ----------
    gdf_turbine : GeoDataFrame
        Rotor geometry in the target EPSG coordinate system.
    max_shadow_length : float
        Maximum shadow length to consider in meters.
    line_length : float
        Current calculated shadow length based on solar geometry and topography.
    solar_position_row : pandas.Series
        Row containing sun azimuth, altitude, and timestamp.
    sf_multiplier : float
        Weighting factor for shadow flicker based on temporal resolution.
    ROTOR_DIAMETER : float
        Rotor diameter in meters.
    target_epsg : str
        EPSG code of the projected CRS (e.g., '3035').
    shadow_ring : np.ndarray
        Mask identifying the possible shadow area for certain altitude.
    geotransform : tuple
        Raster geotransform (GDAL format).
    wt_x : float
        Turbine X coordinate in target CRS.
    wt_y : float
        Turbine Y coordinate in target CRS.
    shadow_distance_map : np.ndarray
        Distance map of matching shadow regions.

    Returns
    -------
    GeoDataFrame or None
        Shadow footprint as a GeoDataFrame for the time step,
        or None if conditions are invalid.
    """
    if gdf_turbine.crs != f"EPSG:{target_epsg}":
        gdf_turbine = gdf_turbine.to_crs(f"EPSG:{target_epsg}")

    sun_azimuth = solar_position_row["sun_azimuth"]
    sun_elevation = solar_position_row["sun_altitude_top"]
    datetime_utc = solar_position_row.name

    if sun_elevation <= 3:
        return None, None, None

    shadow_direction_deg = (360 - sun_azimuth - 90) % 360

    dist_midline, mid_coords = calculate_middle_point_shadow_ring(
        shadow_distance_map,
        shadow_ring,
        geotransform,
        wt_x,
        wt_y,
        shadow_direction_deg,
    )
    if not mid_coords:
        return None, None, None

    max_distance = 2500
    if dist_midline > max_distance:
        return None, None, None

    if (dist_midline + 0.5 * line_length) > max_distance:
        shadow_length = 2.0 * (max_distance - dist_midline)
    else:
        shadow_length = min(line_length, max_shadow_length)

    penumbra_width = ROTOR_DIAMETER + 2 * shadow_length * np.tan(
        np.radians(0.53)
    )

    xoff = mid_coords[0] - wt_x
    yoff = mid_coords[1] - wt_y

    def project_shadow_geometry(row):
        base_polygon = row["geometry"]
        ellipse_shadow = scale(
            base_polygon,
            xfact=shadow_length / ROTOR_DIAMETER,
            yfact=penumbra_width / ROTOR_DIAMETER,
        )
        rotated_ellipse = rotate(
            ellipse_shadow, angle=shadow_direction_deg, origin="centroid"
        )
        return translate(rotated_ellipse, xoff=xoff, yoff=yoff)

    gdf_turbine["geometry"] = gdf_turbine.apply(
        project_shadow_geometry, axis=1
    )
    if gdf_turbine.empty:
        return None, None, None

    gdf_turbine["SF"] = sf_multiplier
    gdf_turbine["time"] = datetime_utc

    return gdf_turbine


def compute_shadow_for_time_step(
    row,
    gdf_turbine,
    altitude_angle_high_array,
    altitude_angle_low_array,
    azimuths_array,
    distance_array,
    geotransform,
    wt_x,
    wt_y,
    tolerance,
    sf_multiplier,
    max_shadow_length,
    target_epsg,
    ROTOR_DIAMETER,
):
    """
    Compute the shadow footprint for a single time step using sun position
    and turbine geometry.

    Parameters
    ----------
    row : pandas.Series
        Solar position data for the time step.
    gdf_turbine : GeoDataFrame
        Geometry of the wind turbine rotor.
    altitude_angle_high_array : np.ndarray
        Altitude angles from top of rotor to all terrain points.
    altitude_angle_low_array : np.ndarray
        Altitude angles from bottom of rotor to all terrain points.
    azimuths_array : np.ndarray
        Azimuth angles from turbine to all terrain points.
    distance_array : np.ndarray
        Distance from turbine to each point in the terrain raster.
    geotransform : tuple
        GDAL geotransform of the raster.
    wt_x : float
        Turbine X coordinate.
    wt_y : float
        Turbine Y coordinate.
    tolerance : float
        Azimuth tolerance for matching sun direction.
    sf_multiplier : float
        Weighting factor for shadow contribution.
    max_shadow_length : float
        Maximum allowed shadow length (m).
    target_epsg : str
        EPSG code of the projection.
    ROTOR_DIAMETER : float
        Rotor diameter in meters.

    Returns
    -------
    GeoDataFrame or None
        Shadow footprint geometry at this time step, or None if no shadow.
    """
    ANGULAR_DIAMETER_SUN = 0.53  # Average apparent angular diameter of the Sun
    # from Earth.
    sun_azimuth = row["sun_azimuth"]
    sun_altitude_top = row["sun_altitude_top"]
    sun_altitude_bottom = row.get("sun_altitude_bottom", sun_altitude_top)

    sun_altitude_top_min = sun_altitude_top - ANGULAR_DIAMETER_SUN / 2
    sun_altitude_bottom_max = sun_altitude_bottom + ANGULAR_DIAMETER_SUN / 2

    shadow_ring = np.logical_and(
        altitude_angle_high_array >= sun_altitude_top_min,
        altitude_angle_low_array <= sun_altitude_bottom_max,
    ).astype(np.float32)

    shadow_az = ((sun_azimuth + 180) % 360 + 180) % 360
    azimuth_match = (
        np.abs((azimuths_array - shadow_az + 360) % 360) <= tolerance
    )
    shadow_map = np.logical_and(azimuth_match, shadow_ring)

    shadow_distance_map = shadow_map * distance_array

    if np.sum(shadow_map) == 0:
        return None

    line_length = min(np.nanmax(shadow_distance_map), max_shadow_length)
    shadow_gdf = calculate_shadow_footprint(
        gdf_turbine,
        max_shadow_length,
        line_length,
        row,
        sf_multiplier,
        ROTOR_DIAMETER,
        target_epsg,
        shadow_ring,
        geotransform,
        wt_x,
        wt_y,
        shadow_distance_map,
    )
    return shadow_gdf


##############################################################################
# 4) Calculate the shadow footprint for all time steps
##############################################################################
def run_sf_vectors(
    lat,
    lon,
    wt_x,
    wt_y,
    altitude_angle_high_array,
    altitude_angle_low_array,
    azimuths_array,
    start_date,
    end_date,
    distance_array,
    HUB_HEIGHT,
    ROTOR_DIAMETER,
    total_height_high,
    target_epsg,
    geotransform,
    max_shadow_length=2500,
    days_resolution=14,
    hours_resolution=1,
    minutes_resolution=4,
    tolerance=1,
    n_jobs=-1,  # if -1 it automatically use all available CPU cores
    filtered_datetimes_index=None,
):
    """
    Calculates the shadow footprint of one turbine over all time steps and returns a
    combined GeoDataFrame of shadow footprints.

    Parameters
    ----------
    lat : float
        Latitude of the turbine (EPSG:4326).
    lon : float
        Longitude of the turbine (EPSG:4326).
    wt_x, wt_y : float
        Projected X, Y coordinates in target EPSG.
    altitude_angle_high_array : np.ndarray
        Altitude angles from rotor top.
    altitude_angle_low_array : np.ndarray
        Altitude angles from rotor bottom.
    azimuths_array : np.ndarray
        Azimuth angles for all points in raster.
    start_date, end_date : datetime
        Start and end date of the analysis period.
    distance_array : np.ndarray
        Distance from turbine to raster cells.
    HUB_HEIGHT : float
        Hub height of turbine (m).
    ROTOR_DIAMETER : float
        Rotor diameter (m).
    total_height_high : float
        Hub + rotor tip elevation (m).
    target_epsg : str
        EPSG code of projection.
    geotransform : tuple
        Raster GDAL geotransform.
    max_shadow_length : float, optional
        Maximum allowed shadow projection length (default 2500 m).
    days_resolution, hours_resolution, minutes_resolution : int, optional
        Time sampling frequency for simulation.
    tolerance : float, optional
        Angular tolerance for azimuth matching.
    n_jobs : int, optional
        Number of parallel workers.

    Returns
    -------
    GeoDataFrame
        Combined shadow geometries for all valid time steps.
    """
    MIN_IN_HOUR = 60  # to transform values in minutes to hours
    ELEVATION_ANGLE = 3  # Minimum elevation of the sun above the Horizon to
    # see the SF
    sf_multiplier = (
        days_resolution * hours_resolution * minutes_resolution
    ) / MIN_IN_HOUR

    # Create turbine geometry (extract single geometry to avoid passing large objects)
    gdf_turbine = create_circle_geodataframe_target_epsg(
        lat, lon, ROTOR_DIAMETER, target_epsg
    )

    # Compute solar positions
    solar_positions = calculate_solar_positions(
        lat,
        lon,
        start_date,
        end_date,
        total_height_high,
        days_resolution,
        hours_resolution,
        minutes_resolution,
    )

    solar_positions = solar_positions.rename(
        columns={"azimuth": "sun_azimuth", "elevation": "sun_altitude_top"}
    )

    if "sun_altitude_bottom" not in solar_positions.columns:
        solar_positions["sun_altitude_bottom"] = solar_positions[
            "sun_altitude_top"
        ]

    # Filter sun positions (avoid low elevation angles)

    solar_positions = solar_positions[
        solar_positions["sun_altitude_top"] >= ELEVATION_ANGLE
    ]
    
    # Adding operation condition based on wind speed

    if filtered_datetimes_index is not None:
       # Count before intersection
       cnt_before = len(solar_positions)
       # Perform the intersection
       valid_index = solar_positions.index.intersection(filtered_datetimes_index)
       cnt_after  = len(valid_index)
       print(f"[DEBUG] Turbine {lat}, {lon}: wind filter retains {cnt_after}/{cnt_before} timesteps")
       # Apply it
       solar_positions = solar_positions.loc[valid_index]
    # Function to compute a single time step (keeping GeoDataFrame intact)
    def process_time_step(row):
        return compute_shadow_for_time_step(
            row,
            gdf_turbine.copy(),  # Pass full GeoDataFrame instead of extracting individual elements
            altitude_angle_high_array,
            altitude_angle_low_array,
            azimuths_array,
            distance_array,
            geotransform,
            wt_x,
            wt_y,
            tolerance,
            sf_multiplier,
            max_shadow_length,
            target_epsg,
            ROTOR_DIAMETER,
        )

    # Parallel Processing with joblib (keeps GeoDataFrames intact)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_time_step)(row)
        for _, row in solar_positions.iterrows()
    )
    # Filter valid results
    sf_vector = [
        res
        for res in results
        if isinstance(res, gpd.GeoDataFrame) and not res.empty
    ]

    if sf_vector:
        return gpd.GeoDataFrame(pd.concat(sf_vector, ignore_index=True))


##############################################################################
# 5) CALCULATES SHADOW FLICKER FREQUENCY FOR ONE TURBINE
##############################################################################
def sf_vector_to_xr(gdf, output_resolution, viewshed_gdal):
    """
    This function rasterizes shadow polygons at each time step into a 3D xarray array
    (time, y, x) and uses the viewshed map as a mask to determine the SF frequency.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
    Input GeoDataFrame containing shadow footprint polygons for a single
    turbine.
        Must contain the following columns:
            - 'geometry': Polygon geometries representing shadow regions.
            - 'time'    : Datetime indicating time of shadow occurrence.
            - 'SF'      : Shadow flicker weight per time step (SF frequency).
    output_resolution : float
        Output raster resolution in meters (e.g., 30.0 for 30m x 30m pixels).
    viewshed_gdal : osgeo.gdal.Dataset
        GDAL raster dataset representing the visibility (viewshed) around the
        turbine. Pixels with value > 0 are considered visible.

    Returns
    -------
    - sf_xr : xarray.Dataset
         SF frequency map with dimensions (time, y, x).
    """

    # Get bounding box
    minx, miny, maxx, maxy = gdf.total_bounds

    # Define rasterization grid
    width = int((maxx - minx) / output_resolution)
    height = int((maxy - miny) / output_resolution)
    transform = rasterio.transform.from_bounds(
        minx, miny, maxx, maxy, width, height
    )

    # Get unique times
    time_values = sorted(gdf["time"].unique())

    # Create an empty 3D NumPy array (time, y, x)
    raster_stack = np.zeros(
        (len(time_values), height, width), dtype=np.float32
    )

    # Rasterize each polygon per time step
    for i, time_step in enumerate(time_values):
        gdf_time = gdf[gdf["time"] == time_step]
        shapes = [
            (mapping(geom), gdf["SF"].min()) for geom in gdf_time.geometry
        ]  # Assign value=1 to polygons

        # Perform rasterization
        raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,  # Background value
            dtype=np.float32,
        )

        # Flip the Y-axis of the raster to match the coordinate system (the raster origin is top-left)
        raster = np.flipud(raster)

        # Store the rasterized data in the raster stack
        raster_stack[i] = raster

    # Convert NumPy array to xarray Dataset
    sf_xr = xr.DataArray(
        raster_stack,
        coords={
            "time": time_values,
            "y": np.linspace(miny, maxy, height),
            "x": np.linspace(minx, maxx, width),
        },
        dims=("time", "y", "x"),
        name="SF_freq_h",
    ).to_dataset()

    # Assign CRS metadata
    sf_xr.rio.write_crs(gdf.crs, inplace=True)

    ### ---- Convert GDAL Viewshed to xarray ---- ###
    viewshed_array = viewshed_gdal.ReadAsArray()
    viewshed_transform = viewshed_gdal.GetGeoTransform()

    # Create xarray DataArray for the viewshed
    viewshed_xr = xr.DataArray(
        viewshed_array,
        coords={
            "y": np.linspace(
                viewshed_transform[3],
                viewshed_transform[3]
                + viewshed_transform[5] * viewshed_array.shape[0],
                viewshed_array.shape[0],
            ),
            "x": np.linspace(
                viewshed_transform[0],
                viewshed_transform[0]
                + viewshed_transform[1] * viewshed_array.shape[1],
                viewshed_array.shape[1],
            ),
        },
        dims=("y", "x"),
        name="viewshed",
    ).to_dataset()

    viewshed_xr.rio.write_crs(gdf.crs, inplace=True)

    viewshed_xr = viewshed_xr.rio.reproject_match(sf_xr)
    ### ---- Align Viewshed and Dataset ---- ###
    viewshed_xr, sf_xr = xr.align(
        viewshed_xr, sf_xr, join="left", fill_value=0
    )

    ### ---- Apply Mask ---- ###
    sf_xr["SF_freq_h"] = sf_xr["SF_freq_h"].where(viewshed_xr["viewshed"] > 0)

    # Cumulate the SF values in time
    sf_xr_sum = sf_xr.sum(dim="time")

    # Define the conditions
    sf_xr_reclassified = xr.where(
        sf_xr_sum.SF_freq_h > 30,
        30,  # Values > 30 → 30
        xr.where(sf_xr_sum.SF_freq_h > 0, 10, 0),  # Values 1-30 → 1
    )  # Values == 0 → 0

    # Assign a name to the DataArray
    sf_xr_reclassified.name = "SF_freq_h"
    sf_xr_reclassified.rio.write_crs("EPSG:3035", inplace=True)

    # Save to NetCDF (commented out for now)
    # ds.to_netcdf(output_netcdf)

    return sf_xr


def process_turbine(
    idx,
    lat,
    lon,
    base_dsm_path,
    HUB_HEIGHT,
    ROTOR_DIAMETER,
    buffer_meters,
    resolution_meters,
    target_epsg,
    start_date,
    end_date,
    max_shadow_length,
    days_resolution,
    hours_resolution,
    minutes_resolution,
    tolerance,
    n_jobs,
    output_resolution,
    filtered_datetimes_index=None,
    output_dir=None,  
):
    """
    High-level wrapper function to process shadow flicker analysis for a single
    wind turbine.

    This function performs all necessary steps(1-5), including DSM generation,
    viewshed analysis, solar position modeling, shadow footprint projection,
    and rasterization of the final output.

    Parameters
    ----------
    idx : str
        Identifier of the turbine (e.g., "T1").
    lat : float
        Latitude of the turbine (EPSG:4326).
    lon : float
        Longitude of the turbine (EPSG:4326).
    base_dsm_path : str
        Path to the input DSM file (GeoTIFF).
    HUB_HEIGHT : float
        Hub height of the turbine in meters.
    ROTOR_DIAMETER : float
        Rotor diameter of the turbine in meters.
    buffer_meters : float
        Buffer size in meters to clip and resample the DSM.
    resolution_meters : float
        Raster resolution in meters.
    target_epsg : str
        EPSG code of the target projection (e.g., "3035").
    start_date : datetime
        Start date of simulation.
    end_date : datetime
        End date of simulation.
    max_shadow_length : float
        Maximum length in meters for shadow projection.
    days_resolution : int
        Simulation frequency in days.
    hours_resolution : int
        Simulation frequency in hours.
    minutes_resolution : int
        Simulation frequency in minutes.
    tolerance : float
        Azimuthal tolerance for shadow alignment.
    n_jobs : int
        Number of parallel threads to use (for joblib).
    output_resolution:  int
        Pixel resolution of the output xarray.Dataset.

    Returns
    -------
    final_sf_xr: xarray.Dataset of hourly shadow flicker frequency.
    """
    overground_height_low = HUB_HEIGHT - (ROTOR_DIAMETER / 2.0)
    overground_height_high = HUB_HEIGHT + (ROTOR_DIAMETER / 2.0)

    print(f"\n[PROCESS] Turbine {idx}: lat={lat}, lon={lon}")
    

    # 1) Create the bas DSM and the viewshed
    (base_dsm, viewshed_gdal) = create_base_dsm_and_viewshed(
        lat,
        lon,
        base_dsm_path,
        overground_height_high,
        buffer_meters,
        resolution_meters,
        target_epsg,
    )
    # 2) get the elevation and location cariables in the target_epsg
    (
        wt_ground_elevation,
        total_height_high,
        total_height_low,
        wt_x,
        wt_y,
    ) = get_wt_elevation_and_location(
        base_dsm, lat, lon, overground_height_high, overground_height_low
    )

    # 3) Generate the azimuth and altitude angle maps for
    #    rotor-high and rotor-low
    (
        azimuths_array,
        altitude_angle_high_array,
        altitude_angle_low_array,
        distance_array,
        altitude_angle_high_map,
        altitude_angle_low_map,
        azimuths_map,
        distance_map,
        geotransform,
    ) = calculate_angle_maps(
        base_dsm, lat, lon, total_height_high, total_height_low
    )

    # 4) Run the Shadow Analysis (produces a GeoDataFrame + geojson file)
    final_sf_gdf = run_sf_vectors(
        lat,
        lon,
        wt_x,
        wt_y,
        altitude_angle_high_array,
        altitude_angle_low_array,
        azimuths_array,
        start_date,
        end_date,
        distance_array,
        HUB_HEIGHT,
        ROTOR_DIAMETER,
        total_height_high,
        target_epsg,
        geotransform,
        max_shadow_length,
        days_resolution,
        hours_resolution,
        minutes_resolution,
        tolerance,
        n_jobs,
        filtered_datetimes_index, 
    )
    if final_sf_gdf is None or final_sf_gdf.empty:
        print(f"[WARNING] No shadow vectors for turbine {idx}. Skipping rasterization.")
        return None
    # ——— New: save raw vectors as GeoJSON ———
    if output_dir is not None:
        geojson_path = os.path.join(output_dir, f"{idx}_shadow_vectors.geojson")
        final_sf_gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"[INFO] Raw shadow vectors for {idx} written to: {geojson_path}")
    # 5) rasterize the GeoDataFrame into an xarray.Dataset
    final_sf_xr = sf_vector_to_xr(
        final_sf_gdf, output_resolution, viewshed_gdal
    )
    return final_sf_xr


##############################################################################
# 6) CALCULATES SHADOW FLICKER FREQUENCY FOR ONE OR MULTIPLE TURBINES
##############################################################################
def merge_sf_datasets_rioxarray(sf_datasets):
    """
    Merge multiple shadow flicker (SF) xarray datasets into a single
    spatio-temporal dataset.
    This function computes the unioned spatial extent of all input SF datasets,
    reprojects them to a common resolution and grid, aligns them spatially,
    and then sums the values (SF frequency hours) for each time step across all
    turbines.

    Parameters
    ----------
    sf_datasets : list of xarray.Dataset
        List of individual SF raster datasets (outputs from `process_turbine`)
        with spatial dimensions ("x", "y") and time ("time") dimension.
        Each dataset must contain the variable `SF_freq_h`.
    Returns
    -------
    merged_sf: xarray.Dataset
        A merged dataset with:
        - The union spatial extent of all inputs.
        - The sum of `SF_freq_h` values from all datasets.
        - CRS preserved from the reference dataset.
    Notes
    -----
    - Assumes all input datasets have consistent CRS and variable names.
    """

    # Step 1: Determine the common grid (union of extents)
    all_bounds = [ds.rio.bounds() for ds in sf_datasets]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)

    # Step 2: Define a reference grid covering the full extent
    # Use first dataset as reference for resolution
    reference_ds = sf_datasets[0]
    resolution_x, resolution_y = reference_ds.rio.resolution()

    width = int((maxx - minx) / resolution_x)
    height = int((maxy - miny) / resolution_y)

    # Create an empty dataset as the common grid
    common_grid = xr.DataArray(
        data=np.zeros((height, width)),
        coords={
            "y": np.linspace(maxy, miny, height),
            "x": np.linspace(minx, maxx, width),
        },
        dims=("y", "x"),
    ).rio.write_crs(reference_ds.rio.crs)

    # Step 3: Reproject and align all datasets to the common grid
    reprojected_datasets = [
        ds.rio.reproject_match(common_grid, nodata=0) for ds in sf_datasets
    ]

    # Step 4: Align datasets to the common grid
    aligned_datasets = xr.align(
        *reprojected_datasets, join="left", fill_value=0
    )

    # Step 5: Sum overlapping values across datasets
    merged_sf = xr.concat(aligned_datasets, dim="new_dim").sum(
        dim="new_dim", skipna=True
    )

    return merged_sf


def sf_multiple_turbines(turbine_data, target_epsg_merged, filtered_map=None,
                         solar_latitude_override=None, 
                         solar_longitude_override=None,
                         output_dir=None,
                         **kwargs):
    """
    Process shadow flicker (SF) simulation for multiple turbines and merge
    their outputs into a single spatio-temporal dataset. It also calculates the
    total SF frequency for all turbines together and creates a reclassified map
    for easy visualization based on typical SF regulation.

    Parameters
    ----------
    turbine_data : list of tuple
        A list of tuples in the form `(idx, lat, lon)`, where:
        - idx : str
            Unique turbine identifier.
        - lat : float
            Latitude of the turbine (EPSG:4326).
        - lon : float
            Longitude of the turbine (EPSG:4326).
    - kwargs: Additional arguments passed to `WIMBY_SF_clean.process_turbine`.

    Returns:
    -------
    merged_sf_raster : xarray.Dataset
        Merged shadow flicker raster dataset containing the SF frequency
        for each time step.
    sf_xr_reclassified : xarray.DataArray
        A reclassified raster of the total SF frequency with discrete values
        for easier visualization using typical policy thresholds:
        - 0 → no shadow
        - 10 → shadow duration between 0 and 30 hours
        - 30 → shadow duration exceeding 30 hours
    Notes
    -----
    - Selecting a  "turbine_data" with a very large total coverage area could
    lead to errors since the merging requires a considerable volume of RAM that
    increases with the total spatial coverage and the pixel resolution (set in
    parameter output_resolution).
    """
    sf_raster_list = []

    for idx, lat, lon in turbine_data:
        # Extract wind-filtered datetimes for each turbine
        filt = filtered_map.get(idx) if filtered_map else None
        sf_xr = process_turbine(idx, lat, lon, filtered_datetimes_index=filt,
                                output_dir=output_dir,
                                **kwargs)

        if sf_xr is not None:
            sf_raster_list.append(sf_xr)
        else:
            print(f"[INFO] Turbine {idx} returned None, skipping.")
    if not sf_raster_list:
        print("[WARNING] No SF data returned for any turbines.")
        return None, None

    # print(sf_raster_list)
    # Merge all processed datasets using the function
    merged_sf_raster = merge_sf_datasets_rioxarray(sf_raster_list)
    sf_xr_sum = merged_sf_raster.sum(dim="time")

    # Define the conditions
    sf_xr_reclassified = xr.where(
        sf_xr_sum.SF_freq_h > 30,
        30,  # Values > 30 → 30
        xr.where(sf_xr_sum.SF_freq_h > 0, 10, 0),  # Values 1-30 → 1
    )  # Values == 0 → 0
    sf_xr_reclassified.name = "SF_freq_h"
    sf_xr_reclassified.rio.write_crs(
        f"EPSG:{target_epsg_merged}", inplace=True
    )

    return merged_sf_raster, sf_xr_reclassified

##############################################################################
# 7) PROCESS WIND DATA FOR OPERATIONAL TIME
##############################################################################

# These globals are loaded once per worker process via init_worker().
# Each worker keeps its own handle to avoid reopening NetCDF files for every turbine.
_ERA5_DS = None
_BIAS_DS = None

# The main script should set these two module-level variables before spawning workers:
#   WIMBY.ERA5_PATH = "<path to era5 netcdf>"
#   WIMBY.BIAS_PATH = "<path to bias netcdf>"
ERA5_PATH = None
BIAS_PATH = None


def init_worker():
    """
    Worker initializer for ProcessPoolExecutor.
    Opens ERA5 and bias datasets once per worker process to avoid repeated I/O.
    The main script must set ERA5_PATH and BIAS_PATH on this module before using this initializer.
    """
    global _ERA5_DS, _BIAS_DS
    if ERA5_PATH is None or BIAS_PATH is None:
        # Fallback: allow running in single-process mode where paths come from the function call.
        return
    _ERA5_DS = xr.open_dataset(ERA5_PATH)
    _BIAS_DS = xr.open_dataset(BIAS_PATH)



@lru_cache(maxsize=200000)
def allowed_hours_by_month(
    lat_q: float,
    lon_q: float,
    height_meters: float,
    year: int,
    cutin_speed: float,
    cutout_speed: float,
):
    """
    Compute, for a specific ERA5 grid cell (lat_q, lon_q), hub height, and year,
    which HOURS (0–23) in each MONTH (1–12) satisfy the wind-speed thresholds
    after bias correction. The result is cached because many turbines share the
    same nearest-neighbor grid cell.

    Returns
    -------
    dict[int, set[int]]
        Mapping: month -> set of allowed hours for that month.
        Example: {1: {7, 8, 9, 10}, 2: {6, 7}, ...}
    """
    # Select ERA5 wind_speed at the nearest grid and height; then apply bias correction.
    ws = _ERA5_DS["wind_speed"].sel(
        latitude=lat_q, longitude=lon_q, height=height_meters, method="nearest"
    )
    factor = _BIAS_DS["ratio_gwa2_era5_mean_WS"].sel(
        latitude=lat_q, longitude=lon_q, height=height_meters, method="nearest"
    ).values
    ws_corr = ws / factor  # corrected wind speed

    # Expecting ERA5 to have 'month' and 'hour' coordinates (as in your source data).
    df = (
        ws_corr.to_dataframe(name="wind_speed")
        .reset_index()[["month", "hour", "wind_speed"]]
    )

    # Filter by cut-in / cut-out thresholds.
    df_valid = df[
        (df["wind_speed"] >= cutin_speed) & (df["wind_speed"] <= cutout_speed)
    ]
    if df_valid.empty:
        return {}

    # Build a month -> allowed hours set mapping.
    allow_map = {}
    for m, sub in df_valid.groupby("month"):
        allow_map[int(m)] = set(int(h) for h in sub["hour"].unique())
    return allow_map


def expand_minutes(hourly_index: pd.DatetimeIndex, r_minutes: int) -> pd.DatetimeIndex:
    """
    Expand an hourly DatetimeIndex to minute-level by adding fixed minute offsets.
    Uses a fully vectorized approach (no Python loops over timestamps).

    Parameters
    ----------
    hourly_index : DatetimeIndex
        Hours that are already allowed by wind filtering.
    r_minutes : int
        Minute step; if >= 60, no expansion is performed.

    Returns
    -------
    DatetimeIndex
        Expanded and sorted timestamps at minute granularity.
    """
    if r_minutes >= 60:
        return hourly_index

    offsets = pd.to_timedelta(np.arange(0, 60, r_minutes), unit="m")
    base_vals = hourly_index.values
    expanded = np.repeat(base_vals, len(offsets)) + np.tile(offsets.values, len(base_vals))
    return pd.DatetimeIndex(expanded).sort_values()


def process_wind_data(
    era5_wind_path,
    bias_correction_path,
    lat,
    lon,
    height_meters,
    cutin_speed,
    cutout_speed,
    start_datetime,
    end_datetime,
    r_days,
    r_hours,
    r_minutes,
):
    """
    Process wind data for a turbine location, apply wind-speed bias correction,
    and generate operational timestamps based on cut-in and cut-out wind speeds.

    The nearest ERA5 grid cell is selected directly with xarray. The selected
    grid coordinates are then used as stable cache keys so turbines sharing the
    same ERA5 cell can reuse the cached wind-filtering result.

    Parameters
    ----------
    era5_wind_path : str or path-like
        Path to the ERA5 wind-speed dataset.
    bias_correction_path : str or path-like
        Path to the wind-speed bias-correction dataset.
    lat : float
        Turbine latitude.
    lon : float
        Turbine longitude.
    height_meters : float
        Turbine hub height in meters.
    cutin_speed : float
        Minimum operational wind speed.
    cutout_speed : float
        Maximum operational wind speed.
    start_datetime : datetime-like
        Start of the requested time range.
    end_datetime : datetime-like
        End of the requested time range.
    r_days : int
        Day sampling interval. Currently not applied separately because the
        timestamps are generated using the hourly interval.
    r_hours : int
        Hour sampling interval.
    r_minutes : int
        Minute sampling interval within each accepted hour.

    Returns
    -------
    pandas.DatetimeIndex
        UTC timestamps within the requested range that satisfy the wind-speed
        thresholds.
    """
    global _ERA5_DS, _BIAS_DS

    # Support both ProcessPoolExecutor workers and single-process execution.
    if _ERA5_DS is None:
        _ERA5_DS = xr.open_dataset(era5_wind_path)

    if _BIAS_DS is None:
        _BIAS_DS = xr.open_dataset(bias_correction_path)

    # Robustly convert both timezone-naive and timezone-aware inputs to UTC.
    start_ts = pd.Timestamp(start_datetime)
    end_ts = pd.Timestamp(end_datetime)

    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    if end_ts < start_ts:
        raise ValueError("end_datetime must be later than or equal to start_datetime.")

    # Let xarray directly select the nearest spatial grid cell.
    nearest_ws = _ERA5_DS["wind_speed"].sel(
        latitude=float(lat),
        longitude=float(lon),
        method="nearest",
    )

    # Extract the actual ERA5 grid coordinates selected by xarray.
    # These coordinates provide stable cache keys for turbines that fall within
    # the same ERA5 grid cell.
    lat_q = float(nearest_ws["latitude"].item())
    lon_q = float(nearest_ws["longitude"].item())

    allow_map = allowed_hours_by_month(
        lat_q=lat_q,
        lon_q=lon_q,
        height_meters=float(height_meters),
        year=int(start_ts.year),
        cutin_speed=float(cutin_speed),
        cutout_speed=float(cutout_speed),
    )

    if not allow_map:
        return pd.DatetimeIndex([], name="datetime", tz="UTC")

    step_hours = max(int(r_hours), 1)

    hourly = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq=pd.Timedelta(hours=step_hours),
    )

    # Vectorized lookup is preferable to repeatedly checking Python sets inside
    # a generator, especially when the requested time range is long.
    allowed_pairs = {
        (month, hour)
        for month, hours in allow_map.items()
        for hour in hours
    }

    month_hour_pairs = zip(hourly.month, hourly.hour)

    mask = np.fromiter(
        (pair in allowed_pairs for pair in month_hour_pairs),
        dtype=bool,
        count=len(hourly),
    )

    hourly_ok = hourly[mask]

    if hourly_ok.empty:
        return pd.DatetimeIndex([], name="datetime", tz="UTC")

    idx = expand_minutes(
        hourly_index=hourly_ok,
        r_minutes=max(int(r_minutes), 1),
    )

    # Minute expansion can create timestamps beyond end_datetime when the final
    # accepted timestamp lies within the last hour.
    idx = idx[(idx >= start_ts) & (idx <= end_ts)]

    return pd.DatetimeIndex(idx, name="datetime")

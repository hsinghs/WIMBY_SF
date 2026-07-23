# -*- coding: utf-8 -*-
"""
Last updated 22/07/2026

@author: Hsing-Hsuan Chen & Luis Ramirez Camargo
"""

import os
import sys
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path

import wimby_sf as WIMBY


# === CONFIG ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = (PROJECT_ROOT 
            / "data"
            / "turbine_info" 
            / "IT_single_turbine.csv")
base_dsm_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Copernicus_DSM"
    / "Copernicus_DSM_30m_COG.tif"
)
era5_wind_path = (
    PROJECT_ROOT
    / "data"
    / "ws_mean_era5_12_by_24"
    / "era5_mean_2013-2022_month_by_hour.nc"
)

bias_correction_path = (
    PROJECT_ROOT
    / "data"
    / "ws_mean_era5_12_by_24"
    / "ratio_gwa2_era5.nc"
)

buffer_meters = 5000.0
resolution_meters = 20.0
output_resolution = 30.0
target_epsg = "3035"

start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 1, 1)

max_shadow_length = 2500
days_resolution = 14
hours_resolution = 1
minutes_resolution = 60
tolerance = 1

JOBS = 8
consider_wind = False  # Set to True to apply the wind-speed filter


def process_turbine(idx, row, output_dir):
    """
    Process one turbine and export its shadow-flicker raster outputs.
    """
    turbine_id = f"T{idx + 1}"

    lat = float(row["Latitude"])
    lon = float(row["Longitude"])
    hub_height = float(row["Hub_height_m"])
    rotor_diameter = float(row["Diameter_m"])

    # Optional wind filtering
    filtered = None

    if consider_wind:
        filtered = {
            turbine_id: WIMBY.process_wind_data(
                era5_wind_path,
                bias_correction_path,
                lat=lat,
                lon=lon,
                height_meters=hub_height,
                cutin_speed=3.5,
                cutout_speed=12.5,
                start_datetime=start_date,
                end_datetime=end_date,
                r_days=days_resolution,
                r_hours=hours_resolution,
                r_minutes=minutes_resolution,
            )
        }

    # Shadow-flicker simulation
    sf_raster, sf_reclass = WIMBY.sf_multiple_turbines(
        turbine_data=[(turbine_id, lat, lon)],
        target_epsg_merged=target_epsg,
        base_dsm_path=base_dsm_path,
        HUB_HEIGHT=hub_height,
        ROTOR_DIAMETER=rotor_diameter,
        buffer_meters=buffer_meters,
        resolution_meters=resolution_meters,
        target_epsg=target_epsg,
        start_date=start_date,
        end_date=end_date,
        max_shadow_length=max_shadow_length,
        days_resolution=days_resolution,
        hours_resolution=hours_resolution,
        minutes_resolution=minutes_resolution,
        tolerance=tolerance,
        n_jobs=1,  # Each process handles one turbine
        output_resolution=output_resolution,
        filtered_map=filtered,
        output_dir=output_dir,
    )

    # Export reclassified map
    out_reclass = os.path.join(
        output_dir,
        f"{turbine_id}_reclassified.tif",
    )

    sf_reclass.to_dataset(name="SF_freq_h").rio.to_raster(
        out_reclass,
        dtype="int32",
        driver="GTiff",
        compress="LZW",
    )

    # Export per-pixel shadow-flicker hours
    hours_map = (
        (sf_raster["SF_freq_h"] > 0)
        .astype("uint16")
        .sum(dim="time")
    )

    hours_map.rio.write_crs(
        sf_raster.rio.crs,
        inplace=True,
    )

    out_hours = os.path.join(
        output_dir,
        f"{turbine_id}_hours.tif",
    )

    hours_map.rio.to_raster(
        out_hours,
        dtype="uint16",
        compress="LZW",
    )

    return turbine_id


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Load turbine data
    df = pd.read_csv(CSV_PATH)

    # Assign turbine IDs: T1, T2, T3, ...
    df["turbine_id"] = df.index.map(lambda i: f"T{i + 1}")

    # Output folder
    output_dir = PROJECT_ROOT / "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Detect turbines that already have both output TIFF files
    reclassified_tifs = {
        filename.split("_")[0]
        for filename in os.listdir(output_dir)
        if filename.endswith("_reclassified.tif")
    }

    hours_tifs = {
        filename.split("_")[0]
        for filename in os.listdir(output_dir)
        if filename.endswith("_hours.tif")
    }

    processed_ids = reclassified_tifs & hours_tifs

    if processed_ids:
        print(
            f"Detected {len(processed_ids)} turbines with both output TIFFs. "
            "These turbines will be skipped."
        )
    else:
        print(
            "No turbines with both output TIFFs were detected. "
            "All onshore turbines will be processed."
        )

    # Select turbines that still need to be processed
    to_run = df[
        ~df["turbine_id"].isin(processed_ids)
    ].copy()

    if to_run.empty:
        print(
            "Nothing to process. All onshore turbines already have "
            "both output TIFFs."
        )
        return

    print(
        f"{len(to_run)} onshore turbines remaining to process."
    )

    # Tell WIMBY where the NetCDF files are located
    WIMBY.ERA5_PATH = era5_wind_path
    WIMBY.BIAS_PATH = bias_correction_path

    # Parallel execution
    with ProcessPoolExecutor(
        max_workers=JOBS,
        initializer=WIMBY.init_worker,
    ) as executor:

        futures = {
            executor.submit(
                process_turbine,
                idx,
                row,
                output_dir,
            ): row["turbine_id"]
            for idx, row in to_run.iterrows()
        }

        for future in as_completed(futures):
            turbine_id = futures[future]

            try:
                completed_turbine_id = future.result()
                print(f"{completed_turbine_id} completed.")

            except Exception as error:
                print(
                    f"[ERROR] Turbine {turbine_id} failed: "
                    f"{error!r}"
                )

    print("All scheduled turbine processing tasks have finished.")


if __name__ == "__main__":
    # Required on Windows when using multiprocessing
    freeze_support()
    main()
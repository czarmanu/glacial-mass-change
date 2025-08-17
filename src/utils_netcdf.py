# *****************************************************************************
# utils_netcdf.py
# *****************************************************************************

# Purpose:
# Provide utilities for:
# 1) Downloading the WGMS NetCDF dataset ZIP
# 2) Extracting the archive
# 3) Loading the NetCDF (.nc4) file
# 4) Computing global glacier mass change diagnostics
# 5) Generating global and regional plots

# Author(s):
# Manu Tom, 2025-

import time
import zipfile
import requests
import logging
import xarray as xr
import config
from pathlib import Path
from utils_plots import plot_global_mass_change, plot_regional_grid


# ............................................................................#
# Step 1: Download the ZIP file from the WGMS website
# ............................................................................#
def download_data(force=False):
    """
    Downloads the WGMS glacier mass change dataset ZIP file
    if not already present or if force=True.
    Saves the file as defined in config.ZIP_FILENAME.

    Args:
        force (bool): If True, re-download the ZIP even if it exists.
    """

    # Handle existing ZIP file
    if config.ZIP_FILENAME.exists():
        if force:
            config.ZIP_FILENAME.unlink()
            logging.info("Deleted existing ZIP file due to --force-download.")
        else:
            logging.info("ZIP file already exists. Skipping download.")
            return

    # Download ZIP file from remote URL
    start = time.perf_counter()
    logging.info("Downloading dataset...")
    with requests.get(config.DATA_URL, stream=True) as r:
        r.raise_for_status()
        with open(config.ZIP_FILENAME, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Report download time
    elapsed = time.perf_counter() - start
    logging.info(f"Download complete. Time taken: {elapsed:.2f} seconds.")


# ............................................................................#
# Step 2: Extract the ZIP archive to the local data directory
# ............................................................................#
def unzip_data(force=False):
    """
    Extracts the ZIP archive to config.DATA_DIR unless already extracted
    or force=True is set.

    Args:
        force (bool): If True, remove and re-extract data directory.
    """

    # If target directory exists, delete it only if force=True
    target_dir = config.DATA_DIR / config.DATA_VERSION
    if target_dir.exists():
        if force:
            import shutil
            shutil.rmtree(target_dir)
            logging.info("Deleted existing extracted folder.")
        else:
            logging.info("Data already extracted. Skipping unzip.")
            return

    # Extract ZIP contents to data directory
    start = time.perf_counter()
    logging.info("Unzipping data...")
    with zipfile.ZipFile(config.ZIP_FILENAME, 'r') as zip_ref:
        zip_ref.extractall(config.DATA_DIR)

    # Report time taken for extraction
    elapsed = time.perf_counter() - start
    logging.info(f"Extraction complete. Time taken: {elapsed:.2f} seconds.")


# ............................................................................#
# Step 3: Locate and open the NetCDF (.nc4) file inside the extracted folder
# ............................................................................#
def load_netcdf_file():
    """
    Searches the extracted NetCDF directory
    and loads the first .nc4 file found.

    Returns:
        xarray.Dataset: The loaded NetCDF dataset.
    Raises:
        FileNotFoundError: If no .nc4 file is found.
    """
    start = time.perf_counter()

    # Search for the first .nc4 file in the expected subdirectory
    try:
        file = next(
            f for f in config.NETCDF_SUBDIR.iterdir() if f.suffix == ".nc4"
        )
    except StopIteration:
        raise FileNotFoundError("No NetCDF file found in extracted directory.")

    # Open the NetCDF file using xarray
    logging.info(f"Opening NetCDF file: {file}")
    ds = xr.open_dataset(file)

    # Log the time taken to load the file
    elapsed = time.perf_counter() - start
    logging.info(f"Loaded NetCDF file. Time taken: {elapsed:.2f} seconds.")
    return ds


def compute_time_taken(start):
    elapsed = time.perf_counter() - start
    logging.info("Time taken: %.2f seconds.", elapsed)


def analyze_and_plot(ds, plot_only=False, summary_only=False):
    """
    Compute and visualize glacier mass change metrics.

    Operations:
    - Aggregate NetCDF fields to compute global annual mass change (Gt),
      area-weighted global mean mass change (m w.e.), associated
      uncertainties, and total glacier area.
    - Generate a global time series plot with ±1σ uncertainty.
    - Generate a multi-panel regional plot (19 regions, Fig. 10 style).

    Args:
        ds (xarray.Dataset): Loaded NetCDF dataset.
        plot_only (bool): If True, generate only plots
        without any further stats.
        summary_only (bool): Kept for backwards compatibility, ignored here.

    Outputs:
        - Global time series figure saved to config.PLOT_FILE
        - Regional multi-panel figure saved to config.PLOT_FILE_INDIVIDUAL
    """
    logging.info("Analyzing data...")

    # Data arrays
    da_gt = ds["glacier_mass_change_gt"]
    da_mwe = ds["glacier_mass_change_mwe"]
    da_area = ds["glacier_area_km2"]
    unc_gt = ds["uncertainty_gt"]
    unc_mwe = ds["uncertainty_mwe"]

    # Aggregations
    global_gt = da_gt.sum(dim=["lat", "lon"])
    area_sum = da_area.sum(dim=["lat", "lon"])
    global_mwe = (da_mwe * da_area).sum(dim=["lat", "lon"]) / area_sum
    global_area_km2 = area_sum

    # Uncertainty propagation
    global_unc_gt = (unc_gt ** 2).sum(dim=["lat", "lon"]) ** 0.5
    weights = da_area / area_sum
    global_unc_mwe = ((weights ** 2) * (unc_mwe ** 2)).sum(
        dim=["lat", "lon"]
    ) ** 0.5

    # Tidy DataFrame
    df = xr.merge(
        [
            global_gt.rename("global_mass_change_gt"),
            global_unc_gt.rename("global_uncertainty_gt"),
            global_mwe.rename("global_mass_change_mwe"),
            global_unc_mwe.rename("global_uncertainty_mwe"),
            global_area_km2.rename("global_glacier_area_km2"),
        ]
    ).to_dataframe().reset_index()

    if not summary_only:
        t0 = time.perf_counter()
        plot_global_mass_change(
            df, config.PLOT_FILE,
            value_col="global_mass_change_gt",
            unc_col="global_uncertainty_gt",
            y_label="Annual mass change (Gt)"
            )
        logging.info("Saved plot to %s. Time taken: %.2f seconds.",
                     config.PLOT_FILE, time.perf_counter() - t0)

        # Build and save 19-panel figure (auto-discovers region codes)
        t0 = time.perf_counter()
        plot_regional_grid(
            base_dir=Path(config.CSV_SUBDIR),
            out_path=Path(config.PLOT_FILE_INDIVIDUAL),
            ncols=4,
            )
        logging.info("Saved plot to %s. Time taken: %.2f seconds.",
                     config.PLOT_FILE_INDIVIDUAL, time.perf_counter() - t0)

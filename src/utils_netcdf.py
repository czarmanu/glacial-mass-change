# *****************************************************************************
# utils_netcdf.py
# *****************************************************************************

# Purpose:
# 1) Download the netCDF (.nc4) data from Dussaillant et al. (2025)
# 2) Unzip it
# 3) Manipulate some data
# 4) Plot results in figures and tables

# Author(s):
# Manu Tom, 2025-

import time
import zipfile
import requests
import logging
import xarray as xr
import config
from utils_plots import plot_global_mass_change


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


def analyze_and_plot(ds, plot_only=False, summary_only=False):
    """
    Computes global mean glacier mass change time series,
    saves summary statistics, and/or generates a time series plot.

    Args:
        ds (xarray.Dataset): Loaded NetCDF dataset.
        plot_only (bool): If True, skip summary and only generate plot.
        summary_only (bool): If True, skip plot and only compute summary.

    Outputs:
        - CSV summary statistics saved to config.SUMMARY_FILE
        - Time series plot saved to config.PLOT_FILE
    """
    logging.info("Analyzing data...")
    start = time.perf_counter()

    da = ds['glacier_mass_change_gt']
    df = (
        da.mean(dim=['lat', 'lon'])
        .to_dataframe()
        .reset_index()[['time', 'glacier_mass_change_gt']]
    )

    if not plot_only:
        df.describe().to_csv(config.SUMMARY_FILE)
        logging.info(f"Saved summary table to {config.SUMMARY_FILE}")

    if not summary_only:
        plot_global_mass_change(df, config.PLOT_FILE)

    elapsed = time.perf_counter() - start
    logging.info(f"Analysis complete. Time taken: {elapsed:.2f} seconds.")

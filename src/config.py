# *****************************************************************************
# config.py
# *****************************************************************************

# Purpose: Configuration parameters file

# Author(s):
# Manu Tom, 2025-

from pathlib import Path

# Version tag for the WGMS glacier mass change dataset
DATA_VERSION = "wgms-amce-2025-02b"

# URL to download the corresponding ZIP archive
DATA_URL = f"https://wgms.ch/downloads/{DATA_VERSION}.zip"

# Local filename for the downloaded ZIP
ZIP_FILENAME = Path(f"data/{DATA_VERSION}.zip")

# Root directory for storing all WGMS-related data
DATA_DIR = Path("data/wgms_data")

# Directory where the ZIP archive will be extracted
EXTRACTED_DIR = DATA_DIR / DATA_VERSION

# Subdirectory within the extracted folder containing NetCDF files
NETCDF_SUBDIR = EXTRACTED_DIR / "global-gridded"

# Output CSV summary file containing mass change statistics
SUMMARY_FILE = Path("results/summary_statistics.csv")

# Output figure file showing time series plots
PLOT_FILE = Path("results/mass_change_timeseries.png")

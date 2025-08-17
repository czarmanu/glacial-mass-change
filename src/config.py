# *****************************************************************************
# config.py
# *****************************************************************************

# Purpose: Set all configuration parameters

# Author(s):
# Manu Tom, 2025-

from pathlib import Path
from typing import Dict

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
CSV_SUBDIR = EXTRACTED_DIR / "individual-glacier"

# Output figure file showing time series plots
PLOT_FILE = Path("results/mass_change_timeseries_global.png")
PLOT_FILE_INDIVIDUAL = Path("results/mass_change_timeseries_individual.png")

# Output CSV summary file containing mass change statistics
SUMMARY_TABLE = Path("results/summary_table_mass_change.csv")

# years to include in annual time series
YEAR_START = 1976
YEAR_END = 2024

# years to include in the summary table
YEAR_START_TABLE = YEAR_START
YEAR_END_TABLE = 2016

# Mapping of 3-letter RGI region codes to numbered long names
REGION_CODE_NUMS: Dict[str, str] = {
    "ALA": "01-Alaska",
    "WNA": "02-Western Canada US",
    "ACN": "03-Arctic Canada North",
    "ACS": "04-Arctic Canada South",
    "GRL": "05-Greenland Periphery",
    "ISL": "06-Iceland",
    "SJM": "07-Svalbard",
    "SCA": "08-Scandinavia",
    "RUA": "09-Russian Arctic",
    "ASN": "10-North Asia",
    "CEU": "11-Central Europe",
    "CAU": "12-Caucasus Middle East",
    "ASC": "13-Central Asia",
    "ASW": "14-South Asia West",
    "ASE": "15-South Asia East",
    "TRP": "16-Low Latitudes",
    "SAN": "17-Southern Andes",
    "NZL": "18-New Zealand",
    "ANT": "19-Antarctica",
}

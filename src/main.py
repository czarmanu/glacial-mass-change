# *****************************************************************************
# main.py
# *****************************************************************************

# Purpose:
# Main function for annual glacial mass change time series creation

# Author(s):
# Manu Tom, 2025-

# *****************************************************************************
# Example invocation
# *****************************************************************************
# python3 main.py
# python3 main.py --plot-only
# python3 main.py --summary-only
# python3 main.py --force-download

import os
import argparse
import config
import logging
from utils_netcdf import (
    download_data,
    unzip_data,
    load_netcdf_file,
    analyze_and_plot
)

# ............................................................................#
# Configure logging: show timestamp, level, and message for INFO and above
# ............................................................................#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ............................................................................#
# Create the 'results' and 'data' directories if they doesn't exist
# ............................................................................#
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ............................................................................#
# Main function (with enhanced CLI flexibility)
# ............................................................................#
def main():
    """
    Parse command-line arguments and run glacial mass balance workflow.

    Supports flexible combinations:
    - Full run (default)
    - Plot-only mode (--plot-only)
    - Summary-only mode (--summary-only)
    - Forced redownload (--force-download)

    Ensures mutual exclusivity between --plot-only and --summary-only.
    """
    parser = argparse.ArgumentParser(
        description="Glacial Mass Balance Time Series Tool")

    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots, skip download and summary")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only compute summary, skip plots")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download and re-unzip of data")

    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.plot_only and args.summary_only:
        logging.error(
            "Cannot use both --plot-only and --summary-only at the same time.")
        return

    # Download and unzip if not skipped
    if not args.plot_only and not args.summary_only:
        if args.force_download or not config.ZIP_FILENAME.exists():
            download_data(force=args.force_download)
        if (
            args.force_download
            or not (config.DATA_DIR / config.DATA_VERSION).exists()
        ):
            unzip_data(force=args.force_download)

    # Load dataset
    ds = load_netcdf_file()

    # Analyze and plot
    analyze_and_plot(ds, plot_only=args.plot_only,
                     summary_only=args.summary_only)


# ............................................................................#
# Entry point
# ............................................................................#
if __name__ == "__main__":
    main()

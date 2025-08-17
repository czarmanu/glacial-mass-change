# *****************************************************************************
# main.py
# *****************************************************************************

# Purpose:
# Orchestrates the glacial mass change workflow:
# - Download and unzip WGMS dataset
# - Generate global and regional plots from NetCDF
# - Build summary table from per-glacier CSVs

# Author(s):
# Manu Tom, 2025-

# *****************************************************************************
# Example invocation
# *****************************************************************************
# python3 main.py
# python3 main.py --plot-only
# python3 main.py --summary-only
# python3 main.py --force-download

import argparse
import logging
import os
from pathlib import Path
import time

import config
from utils_csv import build_summary_table
from utils_netcdf import (
    analyze_and_plot,
    download_data,
    load_netcdf_file,
    unzip_data,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)


def main() -> None:
    """
    Parse command-line arguments and run the glacial mass change workflow.

    Modes:
    - Full run (default): download/unzip if needed,
      plot global/regional time series,
      and build summary table.
    - --plot-only: only generate plots, skip the summary table.
    - --summary-only: only build the summary table, skip plots.
    - --force-download: force re-download and re-extraction of dataset.
    """
    parser = argparse.ArgumentParser(
        description="Glacial Mass Balance Time Series Tool"
    )

    # Summary uses CSV inputs; plots use the NetCDF file.
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots, skip Table-8-like summary.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only build Table-8-like summary (SUMMARY_TABLE), skip plots.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download and re-unzip of data",
    )

    args = parser.parse_args()
    if args.summary_only:
        logging.info("Mode: summary-only")
    elif args.plot_only:
        logging.info("Mode: plot-only")
    elif args.force_download:
        logging.info("Mode: force-download")
    else:
        logging.info("Mode: full run")

    # Validate mutually exclusive flags
    if args.plot_only and args.summary_only:
        logging.error(
            "Cannot use both --plot-only and --summary-only at the same time."
        )
        return

    # Download/unzip when NOT plot-only (covers full + summary-only)
    if not args.plot_only:
        if args.force_download or not config.ZIP_FILENAME.exists():
            download_data(force=args.force_download)
        if (
            args.force_download
            or not (config.DATA_DIR / config.DATA_VERSION).exists()
        ):
            unzip_data(force=args.force_download)

    # In summary-only mode, build SUMMARY_TABLE and return early
    if args.summary_only:
        t0 = time.perf_counter()
        # ds is optional because use_grid_area=False by default
        logging.info("Opening per-glacier CSV files from %s",
                     config.CSV_SUBDIR)
        logging.info("Analyzing CSV data...")
        tbl = build_summary_table(
            Path(config.CSV_SUBDIR),
            ds=None,
            year_start=config.YEAR_START_TABLE,
            year_end=config.YEAR_END_TABLE,
            use_grid_area=False,
        )
        tbl.to_csv(config.SUMMARY_TABLE, index=False)
        logging.info(
            "Saved summary table to %s. Time taken: %.2f seconds.",
            config.SUMMARY_TABLE,
            time.perf_counter() - t0,
        )
        return

    # Load dataset (NetCDF) for plotting runs
    ds = load_netcdf_file()

    # Analyze and plot (summary_only is ignored here; plotting only)
    analyze_and_plot(ds, plot_only=args.plot_only, summary_only=False)

    # Only build SUMMARY_TABLE if not in plot-only mode
    if not args.plot_only:
        t0 = time.perf_counter()
        tbl = build_summary_table(
            Path(config.CSV_SUBDIR),
            ds=ds,
            year_start=config.YEAR_START_TABLE,
            year_end=config.YEAR_END_TABLE,
            use_grid_area=False,
        )
        tbl.to_csv(config.SUMMARY_TABLE, index=False)
        logging.info(
            "Saved summary table to %s. Time taken: %.2f seconds.",
            config.SUMMARY_TABLE,
            time.perf_counter() - t0,
        )


if __name__ == "__main__":
    main()

# *****************************************************************************
# utils_plots.py
# *****************************************************************************

# Purpose:
# Plot routines for glacier mass change analysis.

# Author(s):
# Manu Tom, 2025-

from pathlib import Path
import logging
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from typing import Iterable, List, Optional
from utils_csv import (
    build_regional_timeseries_gt,
    discover_region_codes,
    )
import config


def plot_global_mass_change(
    df: pd.DataFrame,
    out_path,
    value_col: str = "global_mass_change_gt",
    unc_col: Optional[str] = None,
    time_col: str = "time",
    y_label: str = "Annual mass change (Gt)",
    title: Optional[str] = None,
) -> None:
    """
    Plot a time series of global glacier mass change with optional ±1σ
    uncertainty shading.

    Args:
        df (pd.DataFrame): Must contain time, value, and optionally
            uncertainty columns. Time must be datetime-like.
        out_path: File path to save the figure.
        value_col (str): Column with the value to plot (e.g., Gt).
        unc_col (Optional[str]): Column with 1σ uncertainty (same units).
        time_col (str): Column with datetime values.
        y_label (str): Y-axis label.
        title (Optional[str]): Figure title.

    Returns:
        None
    """
    # Ensure required columns exist in the input DataFrame
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError("Required columns are missing from dataframe.")

    # Convert time and value columns to correct dtypes (datetime, numeric)
    ser_time = pd.to_datetime(df[time_col])
    ser_val = pd.to_numeric(df[value_col])

    # Create a new figure and axis (8x4.8 inches) with constrained layout
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")

    # Plot the main time series line
    ax.plot(ser_time, ser_val, linewidth=1.8)

    # If uncertainty column is provided, plot ±1σ shading
    if unc_col is not None and unc_col in df.columns:
        ser_unc = pd.to_numeric(df[unc_col])
        lower = ser_val - ser_unc
        upper = ser_val + ser_unc
        ax.fill_between(ser_time, lower, upper, alpha=0.25)

    # Set axis labels
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    # Optionally add a plot title
    if title:
        ax.set_title(title)

    # Configure X-axis to show years every 5 years, format as YYYY
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=45)

    # Add grid lines for readability
    ax.grid(True, linewidth=0.5, alpha=0.5)

    # Ensure output path has a file extension (.png if missing)
    outp = Path(out_path)
    if outp.suffix == "":
        outp = outp.with_suffix(".png")

    # Save figure as high-resolution PNG and close the figure to free memory
    fig.savefig(outp, dpi=300)
    plt.close(fig)


def plot_regional_grid(
    base_dir: Path,
    out_path: Path,
    region_codes: Iterable[str] | None = None,
    ncols: int = 4,
) -> None:
    """
    Build a multi-panel plot (like Fig. 10 regional subplots) with one panel
    per region showing annual mass change (Gt) and ±1σ shading.

    Args:
        base_dir: Directory containing region CSVs.
        out_path: Output image path ('.png' will be appended if no suffix).
        region_codes: Optional explicit list/set/tuple of region codes.
            If None, they will be discovered from filenames.
        ncols: Number of subplot columns.

    Returns:
        None
    """
    base_dir = Path(base_dir)
    out_path = Path(out_path)

    # Ensure a .png suffix if none is provided
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".png")

    # Discover region codes from filenames if not provided explicitly
    if region_codes is None:
        region_codes = discover_region_codes(base_dir)
    region_codes = list(region_codes)
    if not region_codes:
        raise ValueError("No region codes found. Check input directory.")

    # Collect all per-region time series DataFrames here
    series_list: List[pd.DataFrame] = []

    # Helper function to check if a per-region CSV file exists
    def _series_exists(code: str) -> bool:
        p = base_dir / f"{code}_gla_MEAN-CAL-mass-change-series_obs_unobs.csv"
        return p.exists()

    # Loop over all configured regions (ensures correct Fig. 10 order)
    for code in config.REGION_CODE_NUMS.keys():
        try:
            if code == "SAN":  # aggregate SA1/SA2 (and SAN if present)
                subs = [c for c in ("SAN", "SA1", "SA2") if _series_exists(c)]
                if not subs:
                    continue
                parts: List[pd.DataFrame] = []
                for sub in subs:
                    try:
                        df_sub = build_regional_timeseries_gt(base_dir, sub)
                    except FileNotFoundError:
                        continue
                    # Force the region label to 'SAN' so we get one panel
                    df_tmp = df_sub.copy()
                    df_tmp["region"] = "SAN"
                    parts.append(df_tmp)
                if not parts:
                    continue
                # Combine all SAN-related subregion DataFrames
                df_r = pd.concat(parts, ignore_index=True)
                logging.info("SAN aggregated from: %s", ",".join(subs))
            else:
                # Default case: load time series for a single region
                df_r = build_regional_timeseries_gt(base_dir, code)

            # Add regional DataFrame to the list
            series_list.append(df_r)
        except FileNotFoundError as exc:
            logging.warning("Skipping %s: %s", code, exc)
        except Exception as exc:  # pragma: no cover
            logging.warning("Skipping %s due to error: %s", code, exc)

    if not series_list:
        raise ValueError("No regional series could be built.")

    # Concatenate all regional series into one DataFrame
    df_all = pd.concat(series_list, ignore_index=True)

    # Define subplot grid layout based on number of regions
    # and requested columns
    n = len(config.REGION_CODE_NUMS)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    # Create figure and subplots with constrained layout
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.0 * ncols, 2.8 * nrows),
        layout="constrained",
        squeeze=False,
    )

    # Ensure subplots follow official Fig. 10 regional order
    region_order: List[str] = list(config.REGION_CODE_NUMS.keys())

    # Fill each subplot with the region’s time series
    for i, code in enumerate(region_order, start=1):
        if code not in df_all["region"].unique():
            # If this region has no data (rare), leave axis blank
            r, c = divmod(i - 1, ncols)
            axes[r][c].axis("off")
            continue

        # Identify subplot position
        r, c = divmod(i - 1, ncols)
        ax = axes[r][c]

        # Aggregate per-timepoint values: sum mass, propagate uncertainty (RSS)
        dfr = (
            df_all[df_all["region"] == code]
            .groupby("time", as_index=False)
            .agg(
                mass_gt=("mass_gt", "sum"),
                sigma_gt=("sigma_gt", lambda s: float((s**2).sum() ** 0.5)),
            )
            .sort_values("time")
        )
        # Extract arrays for plotting
        y = dfr["mass_gt"].to_numpy().astype(float).ravel()
        s = dfr["sigma_gt"].to_numpy().astype(float).ravel()

        # Plot time series line and ±1σ uncertainty shading
        ax.plot(dfr["time"], y, linewidth=1.6)
        ax.fill_between(dfr["time"], y - s, y + s, alpha=0.25)

        # Set subplot title (index + region code)
        ax.set_title(f"{i}: {code}")

        # Configure axis appearance
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelrotation=45)

        # Add Y-axis label only for the first column
        if c == 0:
            ax.set_ylabel("Annual mass change (Gt)")

    # Disable unused subplots (e.g., if regions < grid size)
    total_axes = nrows * ncols
    for j in range(n, total_axes):
        rr, cc = divmod(j, ncols)
        axes[rr][cc].axis("off")

    # Save final multi-panel figure and close
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

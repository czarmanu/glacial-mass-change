# *****************************************************************************
# utils_csv.py
# *****************************************************************************
# Purpose:
# Utilities for working with per-glacier CSV files:
# - Build annual regional time series of glacier mass change (Gt) with ±1σ
# - Discover available RGI region codes
# - Compute time-mean glacier area per region from CSVs or the gridded NetCDF
# - Replicate a Table-8-like regional/global summary (mean area
#    and mean mass change)

# Assumptions:
# - For each region code (e.g., ACN) there are at least these two files:
#   <REG>_gla_MEAN-CAL-mass-change-series_obs_unobs.csv
#   <REG>_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs.csv
# - Yearly columns are wide-form (e.g., "1960", "1961", ..., "2024").
# - Values in series/error files are in m w.e.; "Area" is in km^2.

# Author(s):
# Manu Tom, 2025-

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import config
from pathlib import Path
from typing import Dict, List, Tuple, Final


def _year_columns(df: pd.DataFrame) -> List[str]:
    """Return sorted list of year-like columns (e.g., '1960'..'2024')."""
    # Select columns that contain only digits (likely year labels)
    years = [c for c in df.columns if c.strip().isdigit()]
    # Sort the year strings numerically (e.g., 1960 < 2024)
    years_sorted = sorted(years, key=int)
    return years_sorted


def _melt_years(df: pd.DataFrame, id_vars: List[str]) -> pd.DataFrame:
    """Melt wide year columns to long format with 'year' and 'value'."""
    # Get the list of year columns in sorted order
    years = _year_columns(df)
    # Reshape from wide format (years as columns) to long format
    out = df.melt(
        id_vars=id_vars,     # columns to keep fixed (identifiers)
        value_vars=years,    # year columns to melt
        var_name="year",     # new column name for year
        value_name="value",  # new column name for value
    )
    # Convert 'year' column from string to integer type
    out["year"] = out["year"].astype(int)
    return out


def _read_region_series(
    base_dir: Path,
    region_code: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the per-glacier series and total error CSVs for a region.

    Returns:
        (df_series, df_sigma) in long format with columns including:
        ['RGIId','REGION','CenLon','CenLat','Area','WGMS_ID','year', ...]
        'mwe' is m w.e.; 'sigma_mwe' is 1σ m w.e.
    """
    # Build file paths for the series and error CSVs of a given region
    f_series = base_dir / (
        f"{region_code}_gla_MEAN-CAL-mass-change-series_obs_unobs.csv"
    )
    f_total = base_dir / (
        f"{region_code}_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs.csv"
    )

    # Check that both files exist, raise a clear error if missing
    if not f_series.exists():
        raise FileNotFoundError(
            f"Missing series CSV for {region_code}: {f_series}"
        )
    if not f_total.exists():
        raise FileNotFoundError(
            f"Missing TOTAL-ERROR CSV for {region_code}: {f_total}"
        )

    # ........................................................................#
    # Helper function (local scope only): read and standardize a CSV
    # ........................................................................#
    def _load_and_standardize(path: Path) -> pd.DataFrame:
        # Read CSV with auto-detection of delimiter (comma or tab)
        df = pd.read_csv(path, sep=None, engine="python")

        # Normalize column names: strip whitespace, drop duplicates/unnamed
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        drop_cols = [c for c in df.columns if c.lower().startswith("unnamed:")]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Create case-insensitive lookup dictionary for column names
        lower_map = {c.lower(): c for c in df.columns}

        # Helper to pick the first matching candidate column name
        def pick(*cands: str) -> str:
            for cand in cands:
                if cand in lower_map:
                    return lower_map[cand]
            return ""

        # Try to resolve standard metadata columns with flexible naming
        col_rgi = pick("rgiid", "rgi_id", "rgi id")
        col_reg = pick("region")
        col_lon = pick("cenlon", "center_lon", "lon")
        col_lat = pick("cenlat", "center_lat", "lat")

        # Find an area-like column; choose the one with fewest NaNs
        area_like = [c for c in df.columns if "area" in c.lower()]
        if not area_like:
            raise ValueError("No area-like column found.")
        area_candidates = []
        for c in area_like:
            tmp = pd.to_numeric(df[c], errors="coerce")
            area_candidates.append((c, int(tmp.isna().sum())))
        area_candidates.sort(key=lambda t: t[1])
        col_area = area_candidates[0][0]

        # Attempt to resolve WGMS_ID column
        col_wgms = pick("wgms_id", "wgmsid", "wgms")

        # Build renaming dictionary to unify column naming
        rename_map: Dict[str, str] = {}
        if col_rgi:
            rename_map[col_rgi] = "RGIId"
        if col_reg:
            rename_map[col_reg] = "REGION"
        if col_lon:
            rename_map[col_lon] = "CenLon"
        if col_lat:
            rename_map[col_lat] = "CenLat"
        if col_area:
            rename_map[col_area] = "Area"
        if col_wgms:
            rename_map[col_wgms] = "WGMS_ID"

        # Apply renaming to standardized column names
        df = df.rename(columns=rename_map)

        # Ensure Area column exists after standardization
        if "Area" not in df.columns:
            raise ValueError(
                "Column 'Area' missing in inputs after standardization."
            )

        # Force 'Area' to numeric (invalid entries become NaN)
        df["Area"] = pd.to_numeric(df["Area"], errors="coerce")

        # Coerce all year-like columns (e.g. "1960") to numeric values
        years = [c for c in df.columns if c.strip().isdigit()]
        for c in years:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    # ........................................................................#
    # Apply helper to both the series and error CSVs
    # ........................................................................#
    df_s = _load_and_standardize(f_series)
    df_e = _load_and_standardize(f_total)

    # Select identifier variables available in the input (RGIId may be missing)
    use_cols = ["RGIId", "REGION", "CenLon", "CenLat", "Area", "WGMS_ID"]
    id_vars = [c for c in use_cols if c in df_s.columns]

    # Reshape series file: wide → long format with 'year' and 'mwe'
    df_s_long = _melt_years(df_s, id_vars=id_vars).rename(
        columns={"value": "mwe"}
    )
    # Reshape error file: wide → long format with 'year' and 'sigma_mwe'
    df_e_long = _melt_years(df_e, id_vars=id_vars).rename(
        columns={"value": "sigma_mwe"}
    )

    # Aggregate error data by (RGIId, year) or just year if RGIId missing
    if "RGIId" in df_e_long.columns:
        gb_keys = ["RGIId", "year"]
    else:
        gb_keys = ["year"]
    df_e_long = (
        df_e_long.assign(
            sigma_mwe=pd.to_numeric(df_e_long["sigma_mwe"], errors="coerce")
        )
        .groupby(gb_keys, as_index=False)
        .agg(sigma_mwe=("sigma_mwe", "mean"))
    )

    # Decide merge keys: prefer ["RGIId","year"] if both datasets have RGIId
    merge_keys = ["year"]
    if ("RGIId" in df_s_long.columns) and ("RGIId" in df_e_long.columns):
        merge_keys = ["RGIId", "year"]

    # Merge series (mwe) and error (sigma_mwe) on the chosen keys
    df = pd.merge(
        df_s_long,
        df_e_long,
        on=merge_keys,
        how="left",
        validate="m:1",  # ensures error is uniquely matched
    )

    # Ensure sigma_mwe column exists, is numeric, and fill NaNs with 0.0
    df["sigma_mwe"] = pd.to_numeric(df["sigma_mwe"], errors="coerce").fillna(
        0.0
    )

    # Return standardized long-form DataFrames (series + error)
    return df, df_e_long


def build_regional_timeseries_gt(
    base_dir: Path,
    region_code: str,
) -> pd.DataFrame:
    """
    Aggregate a region's per-glacier series to annual mass change (Gt) and 1σ.

    Returns:
        DataFrame with columns: ['time','region','mass_gt','sigma_gt']
        where 'time' is datetime64 ('<year>-01-01').
    """
    # Read and standardize the per-glacier series + error CSVs for this region
    df, _ = _read_region_series(base_dir, region_code)

    # Convert m w.e. (meters water equivalent) into Gt (gigatonnes)
    # per glacier-year.
    # Conversion: 1 m w.e. * 1 km² = 0.001 Gt
    factor = 0.001
    df["gt"] = df["mwe"] * df["Area"] * factor

    # Convert 1σ uncertainty (m w.e.) to Gt per glacier-year
    df["sigma_gt_glacier"] = df.get("sigma_mwe", 0.0) * df["Area"] * factor

    # Aggregate across all glaciers in the region for each year:
    # - Sum of Gt (total annual mass change)
    # - Root-sum-of-squares for uncertainty propagation
    agg_df = df.groupby("year", as_index=False).agg(
        mass_gt=("gt", "sum"),
        sigma_gt=("sigma_gt_glacier", lambda s: float((s**2).sum() ** 0.5)),
    )

    # Add a datetime column ('YYYY-01-01') for plotting convenience
    agg_df["time"] = pd.to_datetime(agg_df["year"].astype(str) + "-01-01")
    # Add region code label to identify which region the data belongs to
    agg_df["region"] = region_code

    # Restrict output to the configured analysis period (YEAR_START–YEAR_END)
    agg_df = agg_df[
        (agg_df["year"] >= config.YEAR_START)
        & (agg_df["year"] <= config.YEAR_END)
    ]

    # Return only the relevant columns for plotting/analysis
    return agg_df[["time", "region", "mass_gt", "sigma_gt"]]


def discover_region_codes(base_dir: Path) -> List[str]:
    """
    Discover region codes from files matching '*_gla_*series_obs_unobs.csv'.
    Returns codes like 'ACN', 'ALA', etc., sorted alphabetically.
    """
    codes = set()  # use a set to avoid duplicates

    # Scan directory for all per-glacier series files
    for pth in base_dir.glob("*_gla_*series_obs_unobs.csv"):
        # Extract the region code (first 3 letters before first underscore)
        code = pth.name.split("_", maxsplit=1)[0]

        # Only keep valid 3-letter codes (RGI-style region abbreviations)
        if len(code) == 3:
            codes.add(code)

    # Return codes as a sorted list (alphabetical order for consistency)
    return sorted(codes)


def _region_mean_area_from_grid(
    base_dir: Path,
    region_codes: List[str],
    ds: xr.Dataset,
    year_start: int,
    year_end: int,
) -> float:
    """
    Time-mean regional glacier area (km^2) over [year_start, year_end],
    using the gridded 'glacier_area_km2' variable. Grid cells are selected
    by snapping glacier centroids to the nearest 0.5° grid cell (via
    nearest-neighbour selection with tolerance).
    """
    # Collect unique glacier centroid coordinates (lat, lon) across all regions
    pts: set[tuple[float, float]] = set()
    for code in region_codes:
        try:
            df_s, _ = _read_region_series(base_dir, code)
        except FileNotFoundError:
            # Skip region if CSVs not available
            continue
        if not {"CenLat", "CenLon"}.issubset(df_s.columns):
            # Skip region if centroid coordinates are missing
            continue

        # Extract glacier centroid coordinates (lat, lon)
        coords = (
            df_s[["CenLat", "CenLon"]]
            .rename(columns={"CenLat": "lat", "CenLon": "lon"})
            .dropna()
            .drop_duplicates()
            .copy()
        )

        # Snap to 0.5° grid centers; we’ll still use nearest with tolerance
        coords["lat_g"] = (coords["lat"] * 2.0).round() / 2.0
        coords["lon_g"] = (coords["lon"] * 2.0).round() / 2.0

        # Add snapped coordinates to set (unique lat/lon pairs)
        pts.update(map(tuple, coords[["lat_g", "lon_g"]].to_numpy()))

    # If no valid centroids found, return NaN
    if not pts:
        return float("nan")

    # Separate into numpy arrays of lats and lons
    lats = np.array([p[0] for p in pts], dtype=float)
    lons = np.array([p[1] for p in pts], dtype=float)

    # Create DataArrays for nearest-neighbour selection
    lat_da = xr.DataArray(lats, dims="points", name="lat")
    lon_da = xr.DataArray(lons, dims="points", name="lon")
    # Select glacier area from gridded dataset at nearest grid cells
    # Tolerance of 0.26° ensures correct snap to 0.5° grid
    try:
        area_pts = ds["glacier_area_km2"].sel(
            lat=lat_da, lon=lon_da, method="nearest", tolerance=0.26
        )
    except Exception:
        # Return NaN if nearest-neighbour selection fails
        return float("nan")

    # Drop points that failed the tolerance match (become NaN everywhere)
    valid = ~np.isnan(area_pts.isel(time=0).values)
    if not np.any(valid):
        return float("nan")
    area_pts = area_pts.isel(points=np.where(valid)[0])

    # Restrict area dataset to requested year window
    try:
        area_win = area_pts.sel(
            time=slice(f"{year_start}-01-01", f"{year_end}-12-31")
        )
    except Exception:
        # Fallback: manually filter by year integers if time indexing fails
        years = np.asarray(area_pts["time"].values)
        years_int = np.array([int(str(y)[:4]) for y in years])
        keep = (years_int >= year_start) & (years_int <= year_end)
        if not np.any(keep):
            return float("nan")
        area_win = area_pts.isel(time=np.where(keep)[0])

    # Sum across selected grid cells, then average over time
    area_series = area_win.sum(dim="points")
    mean_area = float(area_series.mean(dim="time").values)
    return mean_area


def build_summary_table(
    base_dir: Path,
    ds: xr.Dataset | None = None,
    year_start: int = 1976,
    year_end: int = 2016,
    use_grid_area: bool = False,
) -> pd.DataFrame:
    """
    Construct a summary table across RGI regions.

    For each region:
      - 'RGI region': number-long name plus code
      - 'Mean area (year_start–year_end) [km^2]':
        default from CSVs (unique glaciers), optionally from gridded NetCDF
      - 'Mass change (year_start–year_end) [Gt/yr]':
        mean of annual regional Gt over the window

    Also appends a 'Global' row summing areas and mass change across regions.

    Returns:
        pandas.DataFrame ordered by RGI region numbering.
    """
    # Fixed region order (from config) to ensure reproducibility
    region_codes_all: Final[tuple[str, ...]] = tuple(
        config.REGION_CODE_NUMS.keys())
    order: list[str] = list(region_codes_all)

    def _csv_area_fallback(codes: List[str]) -> float:
        """
        Compute mean glacier area directly from CSVs if grid dataset not used.

        - Deduplicates glaciers by 'RGIId' if available
        - If no RGIId: dedupe by centroid coords + area as proxy
        - Avoids over-counting when glaciers appear multiple years
        - Returns sum of unique glacier areas across all given codes
        """
        # Sum unique glacier areas once per glacier to avoid year-multiplying
        vals: List[float] = []
        for sub in codes:
            try:
                df_s, _ = _read_region_series(base_dir, sub)
            except FileNotFoundError:
                continue

            # Deduplicate glaciers before summing area.
            if "RGIId" in df_s.columns:
                df_unique = df_s.drop_duplicates(subset=["RGIId"])
            else:
                # Fallback: dedupe by stable coords + area
                keep_cols = [c for c in [
                    "CenLat", "CenLon", "Area"] if c in df_s.columns]
                df_unique = df_s.drop_duplicates(
                    subset=keep_cols) if keep_cols else df_s

            vals.append(pd.to_numeric(df_unique[
                "Area"], errors="coerce").sum())

        return float(np.nansum(vals)) if vals else float("nan")

    def _safe_round(x: float) -> float | int | None:
        """
        Round to nearest integer, but return None if input is NaN or missing.
        Prevents NaNs propagating into the summary table.
        """
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return int(round(float(x)))

    rows: List[Dict[str, object]] = []
    # Iterate over each RGI region in fixed order
    for code in order:
        codes_to_use: List[str] = [code]

        # Special case for SAN region: sometimes stored as SA1/SA2
        if code == "SAN":
            alts: List[str] = []
            for alt in ("SAN", "SA1", "SA2"):
                p = base_dir / (
                    f"{alt}_gla_MEAN-CAL-mass-change-series_obs_unobs.csv"
                )
                if p.exists():
                    alts.append(alt)
            codes_to_use = alts or ["SAN"]

        # ....................................................................#
        # Mass change (Gt/yr) from CSVs (windowed years)
        # ....................................................................#
        annual_list: List[pd.DataFrame] = []
        for sub in codes_to_use:
            try:
                df_s, _ = _read_region_series(base_dir, sub)
            except FileNotFoundError:
                continue

            # Convert from m w.e. × km² → Gt
            factor = 0.001
            df_tmp = df_s.copy()
            df_tmp["gt"] = (
                pd.to_numeric(df_tmp["mwe"], errors="coerce")
                * pd.to_numeric(df_tmp["Area"], errors="coerce")
                * factor
            )

            # Restrict to window [year_start, year_end]
            yrs = df_tmp["year"].to_numpy()
            win = (yrs >= year_start) & (yrs <= year_end)
            df_tmp = df_tmp.loc[win]

            # Sum to regional annual mass change
            reg_yr = df_tmp.groupby("year", as_index=False)["gt"].sum()
            annual_list.append(reg_yr)

        if not annual_list:
            continue  # Skip region if no valid data

        # Combine multiple CSVs if needed (e.g., SAN split into SA1/SA2)
        reg_all = (
            pd.concat(annual_list, ignore_index=True)
            .groupby("year", as_index=False)["gt"].sum()
        )
        mass_rate = float(reg_all["gt"].mean())

        # ....................................................................#
        # Mean area: default to robust CSV sum; optionally use grid
        # ....................................................................#
        if use_grid_area and ds is not None:
            # Use mean glacierized grid cell area (preferred if available)
            mean_area = _region_mean_area_from_grid(
                base_dir=base_dir,
                region_codes=codes_to_use,
                ds=ds,
                year_start=year_start,
                year_end=year_end,
                )
            # Fallback to CSV if grid area not valid
            if np.isnan(mean_area):
                mean_area = _csv_area_fallback(codes_to_use)
        else:
            # Default: robust CSV-based estimate
            mean_area = _csv_area_fallback(codes_to_use)

        # Append region row
        rows.append(
            {
                "RGI region":
                    f"{config.REGION_CODE_NUMS.get(code, code)} ({code})",
                f"Mean area ({year_start}-{year_end}) [km^2]": _safe_round(
                    mean_area
                ),
                f"Mass change ({year_start}-{year_end}) [Gt/yr]": mass_rate,
            }
        )

    # Assemble DataFrame
    out = pd.DataFrame(rows)
    if not out.empty:
        order_index = {k: i for i, k in enumerate(order)}
        out["_ord"] = (
            out["RGI region"].str.extract(r"\((\w+)\)").iloc[:, 0].map(
                order_index
            )
        )
        out = out.sort_values("_ord").drop(columns="_ord").reset_index(
            drop=True
        )

    # Add global row at the end
    if not out.empty:
        global_area = out.filter(like="[km^2]").sum(numeric_only=True).iloc[0]
        global_mass = out.filter(like="[Gt/yr]").sum(numeric_only=True).iloc[0]
        out.loc[len(out)] = {
            "RGI region": "Global",
            f"Mean area ({year_start}-{year_end}) [km^2]":
                int(round(global_area)),
            f"Mass change ({year_start}-{year_end}) [Gt/yr]":
                global_mass,
            }

    return out

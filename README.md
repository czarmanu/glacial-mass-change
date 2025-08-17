![CI](https://github.com/czarmanu/glacial-mass-change/actions/workflows/github_actions_CI.yml/badge.svg)  
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/czarmanu/glacial-mass-change/blob/main/LICENSE)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14206902.svg)](https://doi.org/10.5904/wgms-amce-2025-02b)

# Annual mass-change estimates (world’s glaciers, time series)

This Python project automates the download, extraction, analysis, and visualization of the WGMS gridded glacier mass change (global, annual) dataset.

![Demo Plot](mass_change_timeseries.png)

---

## Requirements

- Python 3.10+

Install dependencies:

```bash
pip3 install -r requirements.txt
```

---

## Usage

### Run the full analysis:

```bash
python3 src/main.py
```

### Optional CLI flags:

```bash
python3 src/main.py --plot-only         # Only generate plots
python3 src/main.py --summary-only      # Only compute summary table
python3 src/main.py --force-download    # Force re-download and re-unzip
```

---

## Code structure

| File                              | Purpose                                                                |
|-----------------------------------|------------------------------------------------------------------------|
| `src/main.py`                     | Orchestrates download, extraction, analysis, plotting                  |
| `src/config.py`                   | Stores constants such as file paths and URLs                           |
| `src/utils_netcdf.py`             | Helper functions for NetCDF handling and analysis                      |
| `src/utils_csv.py`                | Helper functions for CSV handling and analysis                         |
| `src/utils_plots.py`              | Helper functions for plotting                                          |
| `tests/test_file_exists.py`       | Unit test to verify the presence and structure of the NetCDF file.     |
| `tests/test_netcdf_structure.py`  | Unit test to verify the structure and content of the NetCDF file.      |

---

## Development

Run tests:

```bash
pytest
```

Check code style:

```bash
flake8 .
```

---

## Features

- Downloads the official ZIP file from [wgms.ch](https://wgms.ch/mass_change_estimates/)
- Unzips and extracts the `.nc4` NetCDF and per-glacier CSVs
- Computes global mass-change time series and uncertainties
- Generates the outputs listed below.

---

## Outputs

- `results/mass_change_timeseries_global.png` — Annual global glacier mass change with ±1σ.  
- `results/mass_change_timeseries_individual.png` — 19-panel RGI regional mass change with ±1σ.  
- `results/summary_table_mass_change.csv` — Mean regional area and mean mass change (Gt/yr).

---

## File structure

- NetCDF (global gridded) — 0.5° lat–lon grid, hydrological years 1976–2024:  
  ```
  global-gridded/global-gridded-annual-glacier-mass-change.nc4
  ```

- Per-glacier CSVs:  
  Directory:
  ```
  individual-glacier/
  ```
  Files per region (`<REG>` = 3-letter code like ACN, ALA, …):
  - `<REG>_gla_MEAN-CAL-mass-change-series_obs_unobs.csv` — annual mass-change series (m w.e.) and `Area` (km²).
  - `<REG>_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs.csv` — corresponding 1σ uncertainties (m w.e.).
  - Note: Southern Andes may appear as `SA1` and `SA2`; the code aggregates them into `SAN`.

---

## Dimensions and data variables (NetCDF)

- **Dimensions**:
  - `time`: Hydrological year (e.g., 1976-01-01)
  - `lat`: Latitude (WGS 84 – EPSG:4326)
  - `lon`: Longitude (WGS 84 – EPSG:4326)

- **Data variables**:
  - `glacier_mass_change_gt`: Glacier mass change in Gigatons (Gt)  
  - `glacier_mass_change_mwe`: Glacier mass change in meters water equivalent (m w.e.)  
  - `glacier_area_km2`: Glacier area in km²  
  - `uncertainty_gt`: Glacier mass change uncertainty in Gt  
  - `uncertainty_mwe`: Glacier mass change uncertainty in m w.e.

---

## Citation

Dussaillant, I., Hugonnet, R., Huss, M., Berthier, E., Bannwart, J., Paul, F., and Zemp, M. (2025):  
*Annual mass-change estimates for the world's glaciers. Individual glacier time series and gridded data products.*  
https://doi.org/10.5904/wgms-amce-2025-02b

---

## Related Publication

Dussaillant, I., Hugonnet, R., Huss, M., Berthier, E., Bannwart, J., Paul, F., and Zemp, M. (2025):  
*Annual mass change of the world's glaciers from 1976 to 2024 by temporal downscaling of satellite data with in-situ observations.*  
Earth System Science Data. https://doi.org/10.5194/essd-17-1977-2025

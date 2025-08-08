# *****************************************************************************
# test_netcdf_structure.py
# *****************************************************************************

# Purpose:
# Unit test for verifying the structure and content of the NetCDF file.

# Author(s):
# Manu Tom, 2025-

from utils_netcdf import load_netcdf_file


def test_netcdf_structure():
    """

    This test ensures:
        1. Expected variables and dimensions exist.
        2. Dimensions are non-empty.
        3. Data type is numeric.
        4. Values are within expected range and not all NaN.
    """
    ds = load_netcdf_file()

    # Assert key variable exists
    assert "glacier_mass_change_gt" in ds.variables, (
        "Missing glacier_mass_change_gt"
        )

    # Assert dimension names
    for dim in ["lat", "lon", "time"]:
        assert dim in ds.dims, f"Missing dimension: {dim}"

    # Assert nonzero dimension lengths
    for dim in ["lat", "lon", "time"]:
        assert ds.dims[dim] > 0, f"Dimension {dim} is empty"

    # Assert type of data variable
    assert ds["glacier_mass_change_gt"].dtype.kind in {"f", "i"}, \
        "Unexpected data type for glacier_mass_change_gt"

    # Assert reasonable data range
    data = ds["glacier_mass_change_gt"].values
    assert not (data > 1000).any(), "Values too large"
    assert not (data < -5000).any(), "Values too small"
    assert not ds["glacier_mass_change_gt"].isnull().all(), (
        "All values are NaN"
        )

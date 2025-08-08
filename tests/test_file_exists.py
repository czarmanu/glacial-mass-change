# *****************************************************************************
# test_file_exists.py
# *****************************************************************************

# Purpose:
# Unit test for verifying the presence and structure of the NetCDF file.

# Author(s):
# Manu Tom, 2025-

from utils_netcdf import load_netcdf_file


# ............................................................................#
# Step 1: Unit test
# ............................................................................#
def test_file_exists():
    """
    Test that the NetCDF file exists and includes the expected variable.
    """
    try:
        ds = load_netcdf_file()
        assert "glacier_mass_change_gt" in ds
    except FileNotFoundError:
        assert False, "NetCDF file not found"

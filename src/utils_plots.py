# *****************************************************************************
# utils_plots.py
# *****************************************************************************

# Purpose:
# Plot routines for glacier mass change analysis.

# Author(s):
# Manu Tom, 2025-

import matplotlib.pyplot as plt


def plot_global_mass_change(df, filename):
    """
    Plots global glacier mass change time series.

    Args:
        df (pd.DataFrame): Must contain 'time' and 'glacier_mass_change_gt'
        filename (Path): File path to save the PNG
    """
    plt.figure(figsize=(10, 5))
    plt.plot(
        df['time'],
        df['glacier_mass_change_gt'],
        label='Global average mass change'
    )
    plt.xlabel("Year")
    plt.ylabel("Mass change (kg/m²)")
    plt.title("Global Glacier Mass Change Time Series")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

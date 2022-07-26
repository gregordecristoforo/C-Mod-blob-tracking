import itertools
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pickle
from polygon_to_mask import get_poly_mask
from mask_functions import (
    isolate_SOL,
    subtract_background,
    calculate_blob_density_small_ds,
    calculate_blob_density_large_ds,
)

ds = xr.load_dataset("data/old_data/short_dataset_coordinates_included.nc")
ds = subtract_background(ds)
ds = isolate_SOL(ds)

ds_full = xr.load_dataset("data/1091216028_full_data/1091216028.nc")
ds_full = subtract_background(ds_full)
ds_full = isolate_SOL(ds_full)

ds = calculate_blob_density_small_ds(ds)
ds_full = calculate_blob_density_large_ds(ds_full)


def extract_profiles(ds, variable):
    density_values = ds[variable].mean(dim="time").values
    density_values = density_values.flatten()
    Rs = ds.R.values.flatten()
    return Rs, density_values


# blob_values = ds.blob_density.mean(dim=("time")).values
# blob_values = blob_values.flatten()

# mean_values = ds.SOL_density.mean(dim=("time")).values
# mean_values = mean_values.flatten()

# mean_values_full = ds_full.SOL_density.mean(dim=("time")).values
# mean_values_full = mean_values_full.flatten()

# Rs_blobs = np.flip(ds.R.values, axis=(1))  # orientation different to frames

# Rs = ds.R.values.flatten()
# Rs_blobs = Rs_blobs.flatten()

# plt.scatter(extract_profiles(ds, 'frames'), label='SOL_density')
Rs, mean_values = extract_profiles(ds, "SOL_density")
plt.scatter(Rs, mean_values, label="SOL_density")
Rs, mean_values_full = extract_profiles(ds_full, "SOL_density")
plt.scatter(Rs, mean_values_full, label="SOL_density")
Rs, blob_values = extract_profiles(ds, "blob_density")
plt.scatter(Rs, blob_values, label="blob_density")

Rs, blob_values = extract_profiles(ds_full, "blob_density")
plt.scatter(Rs, blob_values, label="blob_density")

plt.legend()
plt.show()

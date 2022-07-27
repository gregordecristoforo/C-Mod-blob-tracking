import itertools
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pickle
from polygon_to_mask import get_poly_mask
from mask_functions import (
    calculate_blob_density_low_threshold,
    isolate_SOL,
    subtract_background,
    calculate_blob_density_small_ds,
    calculate_blob_density_large_ds,
    extract_profiles,
)

ds = xr.load_dataset("data/old_data/short_dataset_coordinates_included.nc")
ds = subtract_background(ds)
ds = isolate_SOL(ds)

ds = calculate_blob_density_small_ds(ds)

Rs, mean_values = extract_profiles(ds, "SOL_density")
plt.scatter(Rs, mean_values, label="SOL_density")

Rs, blob_values = extract_profiles(ds, "blob_density")
plt.scatter(Rs, blob_values, label="blob_density")


ds_low_threshold = calculate_blob_density_low_threshold(ds)
Rs, blob_low_threshold = extract_profiles(
    ds_low_threshold, "blob_density_low_threshold"
)
plt.scatter(Rs, blob_low_threshold, label="blob_density_low_threshold")


plt.legend()
plt.show()

# ds = subtract_background(ds)
# ds = isolate_SOL(ds)
# ds_full = xr.load_dataset("data/1091216028_full_data/1091216028.nc")

# # ds.frames.mean(dim=("time")).plot()
# # plt.show()
# ds_full = subtract_background(ds_full)
# ds_full = isolate_SOL(ds_full)

# ds_full = calculate_blob_density_large_ds(ds_full)

# ds.SOL_density.mean(dim=("time")).plot()
# plt.show()
# ds.blob_density.mean(dim=("time")).plot()
# plt.show()

# plt.scatter(extract_profiles(ds, 'frames'), label='SOL_density')
# Rs, mean_values_full = extract_profiles(ds_full, "SOL_density")
# plt.scatter(Rs, mean_values_full, label="SOL_density")


# Rs, blob_values = extract_profiles(ds_full, "blob_density")
# blob_values = (blob_values / 23) * 25
# plt.scatter(Rs, blob_values, label="blob_density")

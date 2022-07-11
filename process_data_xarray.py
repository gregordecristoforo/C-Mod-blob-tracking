import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt

ds = xr.load_dataset("data/short_dataset_coordinates_included.nc")

new_x = np.linspace(ds.x[0], ds.x[-1], ds.dims["x"] * 4)
new_y = np.linspace(ds.y[0], ds.y[-1], ds.dims["y"] * 4)

ds = ds.interp(x=new_x, y=new_y, method="cubic")
del ds["x"]
del ds["y"]

mean_of_frames = ds.frames.mean(dim="time")
std_of_frames = ds.frames.std(dim="time")

ds["frames"] = (ds.frames - mean_of_frames) / std_of_frames
ds["frames"] = ds.frames - ds.frames.min(dim="time")
ds["frames"] = ds.frames / ds.frames.max(dim="time")

ds.to_netcdf("data/short_dataset_upsampled_normalized.nc")

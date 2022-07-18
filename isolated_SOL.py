import itertools
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

ds = xr.load_dataset("data/short_dataset_coordinates_included.nc")

R_LCFS = np.load("data/R_LCFS.npy") * 100
Z_LCFS = np.load("data/Z_LCFS.npy") * 100
R_LIM = np.load("data/R_LIM.npy") * 100
Z_LIM = np.load("data/Z_LIM.npy") * 100


def get_SOL_mask(ds, R_LCFS, Z_LCFS, R_LIM, Z_LIM):
    SOL_mask = np.zeros((64, 64))
    f_LCFS = interpolate.interp1d(
            Z_LCFS,
            R_LCFS,
            kind="cubic",
        )
    f_LIM = interpolate.interp1d(
            Z_LIM,
            R_LIM,
            kind="cubic",
        )
        
    for i, j in itertools.product(range(64), range(64)):
        local_R_LCFS = f_LCFS(ds.Z.values[j,i])
        local_R_LIM = f_LIM(ds.Z.values[j,i])

        if local_R_LCFS < ds.R.values[j,i] and local_R_LIM > ds.R.values[j,i]:
            SOL_mask[j,i] = 1
    return SOL_mask


SOL_mask = get_SOL_mask(ds, R_LCFS, Z_LCFS, R_LIM, Z_LIM)

ds["SOL_density"] = xr.where(SOL_mask, ds.frames, 0)
ds.SOL_density.mean(dim=("time", "y")).plot()
ds.frames.mean(dim=("time", "y")).plot()

plt.show()


from scipy import interpolate
import itertools
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def create_test_dataset():
    ds = xr.open_dataset("data/old_data/short_dataset_coordinates_included.nc")
    new_var = np.tile(ds.x.values, (ds.y.size, 1))
    new_var = np.tile(new_var, (ds.time.size, 1, 1))
    ds["test_frames"] = (("time", "y", "x"), new_var)
    return ds


def get_SOL_mask(ds, R_LCFS, Z_LCFS, R_LIM, Z_LIM):
    SOL_mask = np.zeros((64, 64))
    f_LCFS = interpolate.interp1d(Z_LCFS, R_LCFS, kind="cubic",)
    f_LIM = interpolate.interp1d(Z_LIM, R_LIM, kind="cubic",)

    for i, j in itertools.product(range(64), range(64)):
        local_R_LCFS = f_LCFS(ds.Z.values[j, i])
        local_R_LIM = f_LIM(ds.Z.values[j, i])

        if local_R_LCFS < ds.R.values[j, i] and local_R_LIM > ds.R.values[j, i]:
            SOL_mask[j, i] = 1
    return SOL_mask


def isolate_SOL(ds):
    R_LCFS = np.load("data/R_LCFS.npy") * 100
    Z_LCFS = np.load("data/Z_LCFS.npy") * 100
    R_LIM = np.load("data/R_LIM.npy") * 100
    Z_LIM = np.load("data/Z_LIM.npy") * 100

    SOL_mask = get_SOL_mask(ds, R_LCFS, Z_LCFS, R_LIM, Z_LIM)
    ds["SOL_density"] = xr.where(SOL_mask, ds.test_frames, 0)
    return ds


def plot_test_profile(ds):
    mean_values = ds.SOL_density.mean(dim=("time")).values
    mean_values = mean_values.flatten()
    Rs = ds.R.values.flatten()

    plt.scatter(Rs, mean_values, label="SOL_density")
    plt.xlabel("R [m]")
    plt.ylabel("SOL_density")
    plt.show()


ds = create_test_dataset()
ds = isolate_SOL(ds)
plot_test_profile(ds)

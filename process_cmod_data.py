import numpy as np
import netCDF4
import pickle

shot = "short_dataset_upsampled_normalized.nc"
filename = f"data/{shot}"


R_LCFS = np.load(f"data/R_LCFS.npy")

Z_LCFS = np.load(f"data/Z_LCFS.npy")


file2read = netCDF4.Dataset(filename, "r")
print(file2read.variables.keys())
frames = np.transpose(np.asarray(file2read.variables["frames"]), (2, 1, 0))[::-1, :, :]
R = 0.01 * np.transpose(np.asarray(file2read.variables["R"]), (1, 0))
Z = 0.01 * np.transpose(np.asarray(file2read.variables["Z"]), (1, 0))
time = np.asarray(file2read.variables["time"])

t_ref = np.round(time[0], 2)

if len(time) > 2000:
    idx_start = len(time) // 2
    frames = frames[:, :, idx_start : idx_start + 1000]
    time = time[idx_start : idx_start + 1000]


minR, maxR = np.min(R), np.max(R)
minZ, maxZ = np.min(Z), np.max(Z)
slope_R = 1.0 / (maxR - minR)
offset_R = -slope_R * minR
slope_Z = 1.0 / (maxZ - minZ)
offset_Z = -slope_Z * minZ

X = slope_R * R + offset_R
Y = slope_Z * Z + offset_Z
X_LCFS = slope_R * R_LCFS + offset_R
Y_LCFS = slope_Z * Z_LCFS + offset_Z

grid = np.linspace(0.0, 1.0, 256)
shear_contour_y = np.array(range(256))
shear_contour_x = np.array([]).astype(int)

blob_mask = np.zeros((1, 256, 256, len(time)))
for i in shear_contour_y:
    x = X_LCFS[np.argmin(np.abs(Y_LCFS - grid[i]))]
    idx_x_shear = np.argmin(np.abs(grid - x))
    shear_contour_x = np.append(shear_contour_x, idx_x_shear)
    blob_mask[0, : idx_x_shear + 1, i, :] = np.ones(
        (idx_x_shear + 1, np.shape(blob_mask)[3])
    )


data_input = {
    "brt_arr": frames,
    "r_arr": R,
    "z_arr": Z,
    "shear_contour_x": shear_contour_x,
    "shear_contour_y": shear_contour_y,
}


with open(f"data/{shot}_{str(t_ref)}_raw.pickle", "wb") as handle:
    pickle.dump(data_input, handle, protocol=pickle.HIGHEST_PROTOCOL)

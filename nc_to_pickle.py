from shapely.geometry import LineString, Point
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt

ds = xr.open_dataset("ICRF_on_1.072_short.nc")
n = ds.frames.values
R = ds.R.values
Z = ds.Z.values

r_fine = np.load("r_fine.npy")
z_fine = np.load("z_fine.npy")

dataset = {}
dataset["brt_arr"] = n
dataset["brt_arr"] = np.swapaxes(dataset["brt_arr"], 0, 1)
dataset["brt_arr"] = np.swapaxes(dataset["brt_arr"], 1, 2)
dataset["r_arr"] = np.flip(R, axis=1)
dataset["z_arr"] = np.flip(Z, axis=1)

line = np.vstack((r_fine, z_fine)).T
separatrix = LineString(line * 100)


R_index = []
for i in range(64):
    distances = []
    for j in range(64):
        point = Point(R[i][j], Z[i][j])
        distances.append(point.distance(separatrix))
    R_index.append(64 - np.argmin(distances))

Z_index = np.arange(0, 64)


plt.contourf(dataset["r_arr"], dataset["z_arr"], dataset["brt_arr"][:, :, 0])
for i in range(64):
    plt.scatter(
        dataset["r_arr"][Z_index[i], R_index[i]],
        dataset["z_arr"][Z_index[i], R_index[i]],
    )
plt.plot(r_fine * 100, z_fine * 100)
plt.show()

Z_index = np.arange(0, 256)

y_low = np.linspace(0, 1, 64)
y_high = np.linspace(0, 1, 256)
fit_params = np.polyfit(y_low, R_index, 2)
p = np.poly1d(fit_params)
sep_interp = np.rint(p(y_high) * 4).astype(int)

dataset["shear_contour_x"] = sep_interp
dataset["shear_contour_y"] = np.arange(0, 256)
plt.plot(dataset["shear_contour_x"], dataset["shear_contour_y"])
plt.show()

file = open("ICRF_on_1.072_short.pickle", "wb")
pickle.dump(dataset, file)

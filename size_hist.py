import pickle
import cosmoplots
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

ds_on = pickle.load(
    open("../../Documents/CMod data/raymond_shots/1110310009/ds_ICRF_on.pickle", "rb")
)
ds_off = pickle.load(
    open("../../Documents/CMod data/raymond_shots/1110310009/ds_ICRF_off.pickle", "rb")
)

sizes_on = [blob.sizes for blob in ds_on]
width_x_on = [blob.width_x for blob in ds_on]
width_y_on = [blob.width_y for blob in ds_on]
size_on = [item for sublist in sizes_on for item in sublist]
width_x_on = [item for sublist in width_x_on for item in sublist]
width_y_on = [item for sublist in width_y_on for item in sublist]

elongation_on = np.zeros(len(sizes_on))
for i in range(len(sizes_on)):
    elongation_on[i] = (
        width_x_on[i]
        * width_y_on[i]
        / size_on[i]
        * max(width_y_on[i], width_x_on[i])
        / min(width_x_on[i], width_y_on[i])
    )

plt.hist(elongation_on, density=True, alpha = 0.5, label='ICRF on', bins = 32, range=(0,10))

sizes_off = [blob.sizes for blob in ds_off]
width_x_off = [blob.width_x for blob in ds_off]
width_y_off = [blob.width_y for blob in ds_off]
size_off = [item for sublist in sizes_off for item in sublist]
width_x_off = [item for sublist in width_x_off for item in sublist]
width_y_off = [item for sublist in width_y_off for item in sublist]

elongation_off = np.zeros(len(sizes_off))
for i in range(len(sizes_off)):
    elongation_off[i] = (
        width_x_off[i]
        * width_y_off[i]
        / size_off[i]
        * max(width_y_off[i], width_x_off[i])
        / min(width_x_off[i], width_y_off[i])
    )

plt.hist(elongation_off, density=True, alpha=0.5, label='ICRF off ', bins = 32, range=(0,10))

plt.legend()
print(f"elongation on {np.mean(elongation_on)}")
print(f"elongation off {np.mean(elongation_off)}")
plt.xlim(0,10)
plt.xlabel('Elongation')
plt.ylabel('P(Elongation)')
plt.show()

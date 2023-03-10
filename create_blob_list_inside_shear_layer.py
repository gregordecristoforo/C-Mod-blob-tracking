import pickle
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

ds_on = pickle.load(
    open("../../Documents/CMod data/raymond_shots/1110310009/ds_ICRF_on.pickle", "rb")
)
ds_off = pickle.load(
    open("../../Documents/CMod data/raymond_shots/1110310009/ds_ICRF_off.pickle", "rb")
)

R_LCFS = (
    np.load("/home/gregor/Documents/CMod data/raymond_shots/1110310009/R_LCFS.npy")
    * 100
)
Z_LCFS = (
    np.load("/home/gregor/Documents/CMod data/raymond_shots/1110310009/Z_LCFS.npy")
    * 100
)
r_limiter = (
    np.load("/home/gregor/Documents/CMod data/raymond_shots/1110310009/r_limiter.npy")
    * 100
)
z_limiter = (
    np.load("/home/gregor/Documents/CMod data/raymond_shots/1110310009/z_limiter.npy")
    * 100
)


def cross_layer(R, Z):
    a = 6.15
    b = -558.84
    Z_reff = R * a + b
    return Z < Z_reff


blobs_inside_shear_layer = []
for blob in ds_on:
    layer_positions = cross_layer(blob.center_of_mass_R, blob.center_of_mass_Z)
    if not layer_positions[0]:
        blobs_inside_shear_layer.append(blob)

print(f"number of blobs on: {len(ds_on)}")
print(f"number of blobs inside: {len(blobs_inside_shear_layer)}")

pickle.dump(blobs_inside_shear_layer, open("blobs_inside_layer_on.pickle", "wb"))

blobs_inside_shear_layer = []
for blob in ds_off:
    layer_positions = cross_layer(blob.center_of_mass_R, blob.center_of_mass_Z)
    if not layer_positions[0]:
        blobs_inside_shear_layer.append(blob)

print(f"number of blobs off: {len(ds_on)}")
print(f"number of blobs inside: {len(blobs_inside_shear_layer)}")


pickle.dump(blobs_inside_shear_layer, open("blobs_inside_layer_off.pickle", "wb"))

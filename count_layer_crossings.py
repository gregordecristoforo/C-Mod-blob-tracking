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

R_LCFS = np.load('/home/gregor/Documents/CMod data/raymond_shots/1110310009/R_LCFS.npy')*100
Z_LCFS = np.load('/home/gregor/Documents/CMod data/raymond_shots/1110310009/Z_LCFS.npy')*100
r_limiter = np.load('/home/gregor/Documents/CMod data/raymond_shots/1110310009/r_limiter.npy')*100
z_limiter = np.load('/home/gregor/Documents/CMod data/raymond_shots/1110310009/z_limiter.npy')*100

def cross_layer(R, Z):
    a = 6.15
    b = -558.84
    Z_reff = R*a + b
    return Z < Z_reff

stats_off = {}
stats_on = {}

sizes_on = [np.mean(blob.sizes) for blob in ds_on]

number_of_crossings = 0
blob_sizes_crossings = []
for blob in ds_on:
    layer_positions = cross_layer(blob.center_of_mass_R,blob.center_of_mass_Z)
    if (layer_positions[0] == False) and (True in layer_positions):
        number_of_crossings += 1
        blob_sizes_crossings.append(np.mean(blob.sizes))

print(f"ICRF on {number_of_crossings} crossings")
upper_border_on = np.mean(sizes_on) + np.mean(sizes_on)
valid_blobs_on = [size for size in blob_sizes_crossings if size < upper_border_on]
print(f"ICRF on {len(valid_blobs_on)} valid blobs")

sizes_off = [np.mean(blob.sizes) for blob in ds_off]

number_of_crossings = 0
blob_sizes_crossings = []
for blob in ds_off:
    layer_positions = cross_layer(blob.center_of_mass_R,blob.center_of_mass_Z)
    if (layer_positions[0] == False) and (True in layer_positions):
        number_of_crossings += 1
        blob_sizes_crossings.append(np.mean(blob.sizes))

print(f"ICRF off {number_of_crossings} crossings")
upper_border_off = np.mean(sizes_off) + np.mean(sizes_off)
valid_blobs_off = [size for size in blob_sizes_crossings if size < upper_border_off]
print(f"ICRF off {len(valid_blobs_off)} valid blobs")


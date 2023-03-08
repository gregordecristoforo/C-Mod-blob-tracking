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


print(f"Number of blobs ICRF off: {len(ds_off)}")
print(f"Number of blobs ICRF on: {len(ds_on)}")

stats_off = {}
stats_on = {}

stats_on["lifetimes"] = [blob.life_time for blob in ds_on]
stats_off["lifetimes"] = [blob.life_time for blob in ds_off]

# print(f"lifetime on {np.mean(stats_on['lifetimes'])}")
# print(f"lifetime off {np.mean(stats_off['lifetimes'])}")
#
# plt.hist(stats_on['lifetimes'], density=True)
# plt.hist(stats_off['lifetimes'], density=True)
# plt.show()

index = np.random.randint(0,len(ds_on), size=100)
for i in index:
    blob = ds_on[i]
    plt.plot(blob.center_of_mass_R, blob.center_of_mass_Z)
    # print(blob.life_time)

R = [90, 90.65]
Z = [-5, -1]
f = interpolate.interp1d(R, Z, fill_value='extrapolate')

plt.plot([89.5,91.5], [f(89.5), f(91.5)], '--k')
plt.plot(R_LCFS,Z_LCFS, '--k')
# plt.plot(r_limiter,z_limiter, '--k')

plt.show()

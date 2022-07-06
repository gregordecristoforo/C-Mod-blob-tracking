from blob import Blob
import pickle
import matplotlib.pyplot as plt
import numpy as np

blob_list = pickle.load(open("1091216028_1.45_raft_blobs.pickle", "rb"))

lifetimes = [blob.life_time for blob in blob_list]
amplitudes = [blob.amplitudes for blob in blob_list]
velocities_x = [blob.velocities_R for blob in blob_list]
sizes = [blob.sizes for blob in blob_list]
# centers_of_mass_xs = [blob.centers_of_mass_x for blob in blob_list]

vx_max = [np.max(v) for v in velocities_x if len(v) > 0]
vx_mean = [np.mean(v) for v in velocities_x if len(v) > 0]

amp_max = [np.max(amp) for amp in amplitudes if len(amp) > 1]
amp_mean = [np.mean(amp) for amp in amplitudes if len(amp) > 1]

size_max = [np.max(size) for size in sizes if len(size) > 1]
size_mean = [np.mean(size) for size in sizes if len(size) > 1]

# plt.scatter(size_max, vx_max)
# plt.show()

plt.hist(amp_max, bins=64)
plt.show()

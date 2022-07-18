from blob import Blob
import pickle
import matplotlib.pyplot as plt
import numpy as np


blob_list = pickle.load(open("data/1091216028_1.45_raft_blobs.pickle", "rb"))
# blob_list = pickle.load(open("data/gregor_upsampling_method_blobs.pickle", "rb"))

blob_ids = [blob.blob_id for blob in blob_list]

for blob in blob_list:
    # plt.plot(blob.velocities_R)
    blob.smoothen_all_parameters()
    # blob.remove_frames_close_to_borders()
    # plt.plot(blob.velocities_R)
    # plt.show()

    # plt.title(blob.blob_id)
    # plt.plot(blob.sizes_Z)
    # blob.smoothen_all_parameters()
    # plt.plot(blob.sizes_Z)
    # plt.show()
# blob_list = pickle.load(open("data/new_blobs.pickle", "rb"))
# print(blob_list[0].amplitudes)


lifetimes = [blob.life_time for blob in blob_list]
amplitudes = [blob.amplitudes for blob in blob_list]
velocities_x = [blob.velocities_x for blob in blob_list]
velocities_y = [blob.velocities_y for blob in blob_list]
sizes = [blob.sizes for blob in blob_list]
widths_x = [blob.width_x for blob in blob_list]
widths_y = [blob.width_y for blob in blob_list]
centers_of_mass_x = [blob._centers_of_mass_x[0] for blob in blob_list]


# # centers_of_mass_xs = [blob.centers_of_mass_x for blob in blob_list]

vx_max = [np.max(v) for v in velocities_x if len(v) > 0]
vx_mean = [np.mean(v) for v in velocities_x if len(v) > 0]

vy_max = [np.max(v) for v in velocities_y if len(v) > 0]
vy_mean = [np.mean(v) for v in velocities_y if len(v) > 0]

amp_max = [np.max(amp) for amp in amplitudes if len(amp) > 1]
amp_mean = [np.mean(amp) for amp in amplitudes if len(amp) > 1]

size_max = [np.max(size) for size in sizes if len(size) > 1]
size_mean = [np.mean(size) for size in sizes if len(size) > 1]

width_R_max = [np.max(width_R) for width_R in widths_x if len(width_R) > 1]
width_R_mean = [np.mean(width_R) for width_R in widths_x if len(width_R) > 1]

width_Z_max = [np.max(width_Z) for width_Z in widths_y if len(width_Z) > 1]
width_Z_mean = [np.mean(width_Z) for width_Z in widths_y if len(width_Z) > 1]

print(np.mean(centers_of_mass_x) / 255 * 0.06)
# plt.scatter(size_max, vx_max)
# plt.show()
print(np.mean(vy_mean))
plt.hist(vy_mean, bins=64)
plt.show()

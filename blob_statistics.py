import pickle
import matplotlib.pyplot as plt
import numpy as np


# blobs_quick = pickle.load(
#     open(
#         "data/1091216028_quick_interpolation/quick_interpolation_raft_blobs.pickle",
#         "rb",
#     )
# )
# blob_ids = [blob.blob_id for blob in blobs_quick]
# print(f"Number of blobs: {len(blob_ids)}")

high_threshold_blob_list = pickle.load(
    open("data/1091216028_low_threshold/low_threshold_raft_blobs.pickle", "rb")
)
blob_ids = [blob.blob_id for blob in high_threshold_blob_list]
print(f"Number of blobs: {len(blob_ids)}")

low_threshold_blob_list = pickle.load(
    open("data/1091216028_1.45_raft_blobs.pickle", "rb")
)

blob_ids = [blob.blob_id for blob in low_threshold_blob_list]
print(f"Number of blobs: {len(blob_ids)}")

blob_dict_high = {}
blob_dict_low = {}

for (dictionary, blob_list) in zip(
    [blob_dict_low, blob_dict_high], [high_threshold_blob_list, low_threshold_blob_list]
):
    dictionary["lifetimes"] = [blob.life_time for blob in blob_list]

    dictionary["lifetimes"] = [blob.life_time for blob in blob_list]
    dictionary["amplitudes"] = [blob.amplitudes for blob in blob_list]
    dictionary["velocities_x"] = [blob.velocities_x for blob in blob_list]
    dictionary["velocities_R"] = [blob.velocities_R for blob in blob_list]
    dictionary["velocities_y"] = [blob.velocities_y for blob in blob_list]
    dictionary["velocities_Z"] = [blob.velocities_Z for blob in blob_list]
    dictionary["sizes"] = [blob.sizes for blob in blob_list]
    dictionary["widths_x"] = [blob.width_x for blob in blob_list]
    dictionary["widths_y"] = [blob.width_y for blob in blob_list]
    dictionary["widths_R"] = [blob.width_R for blob in blob_list]
    dictionary["widths_Z"] = [blob.width_Z for blob in blob_list]
    dictionary["centers_of_mass_x"] = [blob._centers_of_mass_x[0] for blob in blob_list]

    dictionary["vx_max"] = [np.max(v) for v in dictionary["velocities_x"] if len(v) > 0]
    dictionary["vx_mean"] = [
        np.mean(v) for v in dictionary["velocities_x"] if len(v) > 0
    ]

    dictionary["vR_max"] = [np.max(v) for v in dictionary["velocities_R"] if len(v) > 0]
    dictionary["vR_mean"] = [
        np.mean(v) for v in dictionary["velocities_R"] if len(v) > 0
    ]

    dictionary["vy_max"] = [np.max(v) for v in dictionary["velocities_y"] if len(v) > 0]
    dictionary["vy_mean"] = [
        np.mean(v) for v in dictionary["velocities_y"] if len(v) > 0
    ]

    dictionary["vZ_max"] = [np.max(v) for v in dictionary["velocities_Z"] if len(v) > 0]
    dictionary["vZ_mean"] = [
        np.mean(v) for v in dictionary["velocities_Z"] if len(v) > 0
    ]

    dictionary["amp_max"] = [
        np.max(amp) for amp in dictionary["amplitudes"] if len(amp) > 1
    ]
    dictionary["amp_mean"] = [
        np.mean(amp) for amp in dictionary["amplitudes"] if len(amp) > 1
    ]

    dictionary["size_max"] = [
        np.max(size) for size in dictionary["sizes"] if len(size) > 1
    ]
    dictionary["size_mean"] = [
        np.mean(size) for size in dictionary["sizes"] if len(size) > 1
    ]

    dictionary["width_x_max"] = [
        np.max(width_R) for width_R in dictionary["widths_x"] if len(width_R) > 1
    ]
    dictionary["width_x_mean"] = [
        np.mean(width_R) for width_R in dictionary["widths_x"] if len(width_R) > 1
    ]

    dictionary["width_R_max"] = [
        np.max(width_R) for width_R in dictionary["widths_R"] if len(width_R) > 1
    ]
    dictionary["width_R_mean"] = [
        np.mean(width_R) for width_R in dictionary["widths_R"] if len(width_R) > 1
    ]

    dictionary["width_y_max"] = [
        np.max(width_Z) for width_Z in dictionary["widths_y"] if len(width_Z) > 1
    ]
    dictionary["width_y_mean"] = [
        np.mean(width_Z) for width_Z in dictionary["widths_y"] if len(width_Z) > 1
    ]

    dictionary["width_Z_max"] = [
        np.max(width_Z) for width_Z in dictionary["widths_Z"] if len(width_Z) > 1
    ]
    dictionary["width_Z_mean"] = [
        np.mean(width_Z) for width_Z in dictionary["widths_Z"] if len(width_Z) > 1
    ]

plt.hist(blob_dict_low["lifetimes"], bins=64)
plt.hist(blob_dict_high["lifetimes"], bins=64)
plt.show()

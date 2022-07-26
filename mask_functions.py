import itertools
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import pickle
from polygon_to_mask import get_poly_mask


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
    ds["SOL_density"] = xr.where(SOL_mask, ds.frames, 0)
    return ds


def subtract_background(ds):

    R_LCFS = np.load("data/R_LCFS.npy") * 100
    Z_LCFS = np.load("data/Z_LCFS.npy") * 100
    R_LIM = np.load("data/R_LIM.npy") * 100
    Z_LIM = np.load("data/Z_LIM.npy") * 100

    SOL_mask = get_SOL_mask(ds, R_LCFS, Z_LCFS, R_LIM, Z_LIM)
    tmp = xr.where(SOL_mask, ds.frames, np.inf)
    ds["frames"] = ds["frames"] - tmp.min()
    return ds


def calculate_blob_density_small_ds(ds):
    blob_list = pickle.load(open("data/1091216028_1.45_raft_blobs.pickle", "rb"))

    ds["blob_density"] = xr.zeros_like(ds.frames)
    for blob in blob_list:
        for i in range(len(blob.frames_of_appearance)):
            frame = blob.frames_of_appearance[i]
            single_frame_data = ds.frames.isel(time=frame).values
            mask = get_poly_mask(blob._polygon_of_predicted_blobs[i], 64, 64)

            # plt.contourf(single_frame_data)
            # plt.contour(mask.T)
            # plt.show()

            blob_single_frame = np.where(mask.T, single_frame_data, 0)

            # plt.contourf(blob_single_frame)
            # plt.show()

            blob_single_frame = np.flip(
                blob_single_frame, axis=(1)
            )  # orientation different to frames

            # plt.contourf(blob_single_frame)
            # plt.show()

            ds["blob_density"].isel(time=frame).values += blob_single_frame
    return ds


def calculate_blob_density_large_ds(ds):
    ds["blob_density"] = xr.zeros_like(ds.frames)

    file_index = [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
    ]
    for blob_file in file_index:
        blob_list = pickle.load(
            open(
                f"data/1091216028_full_data/1091216028_1.4_{blob_file}_raft_blobs.pickle",
                "rb",
            )
        )
        for blob in blob_list:
            print(blob._polygon_of_predicted_blobs)
            for i in range(len(blob.frames_of_appearance)):
                frame = blob.frames_of_appearance[i]
                single_frame_data = ds.frames.isel(time=frame).values
                mask = get_poly_mask(blob._polygon_of_predicted_blobs[i], 64, 64)

                # plt.contourf(single_frame_data)
                # plt.contour(mask.T)
                # plt.show()

                blob_single_frame = np.where(mask.T, single_frame_data, 0)

                # plt.contourf(blob_single_frame)
                # plt.show()

                blob_single_frame = np.flip(
                    blob_single_frame, axis=(1)
                )  # orientation different to frames

                # plt.contourf(blob_single_frame)
                # plt.show()

                ds["blob_density"].isel(time=frame).values += blob_single_frame
    return ds

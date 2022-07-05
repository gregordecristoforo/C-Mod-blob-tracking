import xarray as xr
import pickle
from polygon_to_mask import get_poly_mask
import matplotlib.pyplot as plt

ds = xr.load_dataset("short_dataset_coordinates_included.nc")

# ds.frames = (ds.frames - ds.frames.mean(dim="time")) / (ds.frames.std(dim="time"))
blob_list = pickle.load(open("1091216028_1.45_raft_blobs.pickle", "rb"))

blob = blob_list[70]
print(blob.file_name)

# for i in range(len(blob.polygon_of_predicted_blobs)):
#     mask = get_poly_mask(blob.polygon_of_predicted_blobs[i], 64, 64)
#     plt.contour(mask)

plt.figure()
ds["norm_frames"] = (ds.frames - ds.frames.mean(dim="time")) / ds.frames.std(dim="time")
ds.norm_frames.isel(time=blob.frames_of_appearance[1] + 0).plot()
mask = get_poly_mask(blob.polygon_of_predicted_blobs[1], 64, 64)
plt.contour(mask.T)

plt.show()

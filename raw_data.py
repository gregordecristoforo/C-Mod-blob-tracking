import xarray as xr
import pickle
from polygon_to_mask import get_poly_mask
import matplotlib.pyplot as plt

ds = xr.load_dataset("short_dataset_coordinates_included.nc")

# ds.frames = (ds.frames - ds.frames.mean(dim="time")) / (ds.frames.std(dim="time"))
blob_list = pickle.load(open("list_of_blobs.pickle", "rb"))

blob = blob_list[22]

print(blob.sizes)

# for i in range(len(blob.polygon_of_predicted_blobs)):
#     mask = get_poly_mask(blob.polygon_of_predicted_blobs[i], 64, 64)
#     plt.contour(mask)

plt.figure()
ds.frames.isel(time=blob.frames_of_appearance[15]).plot()
mask = get_poly_mask(blob.polygon_of_predicted_blobs[15], 64, 64)
plt.contour(mask.T)

plt.show()

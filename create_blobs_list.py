import contextlib
from polygon_to_mask import get_poly_mask
import pickle
import matplotlib.pyplot as plt
import numpy as np
from blob import Blob

data = pickle.load(open("1091216028_1.45_raft.pickle", "rb"))
# example_polygon = data["output_tracking"][301][0][4]

blob_ids = []
VIoUs = []
centers_of_mass_x = []
centers_of_mass_y = []
polygon_of_predicted_blobs = []
polygon_of_brightness_contours = []

for i in range(len(data["output_tracking"])):

    frame_data = data["output_tracking"][i]

    with contextlib.suppress(IndexError):
        blob_ids.append(frame_data[0][0])
        VIoUs.append(frame_data[0][1])
        centers_of_mass_x.append(frame_data[0][2])
        centers_of_mass_y.append(frame_data[0][3])
        polygon_of_predicted_blobs.append(frame_data[0][4])
        polygon_of_brightness_contours.append(frame_data[0][5])

last_blob_id = 0
list_of_blobs = []

temp_VIoUs = []
temp_centers_of_mass_x = []
temp_centers_of_mass_y = []
temp_polygon_of_predicted_blobs = []
temp_polygon_of_brightness_contours = []

blob_ids.append(np.inf)
for i in range(len(blob_ids)):

    if blob_ids[i] != last_blob_id:
        blob = Blob(
            last_blob_id,
            temp_VIoUs,
            temp_centers_of_mass_x,
            temp_centers_of_mass_y,
            temp_polygon_of_predicted_blobs,
            temp_polygon_of_brightness_contours,
        )
        list_of_blobs.append(blob)

        temp_VIoUs = []
        temp_centers_of_mass_x = []
        temp_centers_of_mass_y = []
        temp_polygon_of_predicted_blobs = []
        temp_polygon_of_brightness_contours = []

    if blob_ids[i] != np.inf:
        temp_VIoUs.append(VIoUs[i])
        temp_centers_of_mass_x.append(centers_of_mass_x[i])
        temp_centers_of_mass_y.append(centers_of_mass_y[i])
        temp_polygon_of_predicted_blobs.append(polygon_of_predicted_blobs[i])
        temp_polygon_of_brightness_contours.append(polygon_of_brightness_contours[i])
    last_blob_id = blob_ids[i]

list_of_blobs.pop(0)


with open("list_of_blobs.pickle", "wb") as handle:
    pickle.dump(list_of_blobs, handle)

print(list_of_blobs[0].life_time)

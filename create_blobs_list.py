import pickle
import numpy as np
from blob import Blob
import click


@click.command()
@click.option(
    "--raft_output",
    default="data/1091216028_1.45_raft.pickle",
    help="raft output file.",
)
@click.option("--raw_data", default="data/1091216028_1.45.nc", help="raw data file.")
def create_blob_list(raft_output, raw_data):
    data = pickle.load(open(raft_output, "rb"))

    blob_ids = []
    VIoUs = []
    centers_of_mass_x = []
    centers_of_mass_y = []
    polygon_of_predicted_blobs = []
    polygon_of_brightness_contours = []
    frames_of_appearance = []

    for i in range(len(data["output_tracking"])):
        blobs_in_frame = data["output_tracking"][i]

        if blobs_in_frame is None:
            continue

        for blob in range(len(blobs_in_frame)):
            blob_ids.append(blobs_in_frame[blob][0])
            VIoUs.append(blobs_in_frame[blob][1])
            centers_of_mass_x.append(blobs_in_frame[blob][2])
            centers_of_mass_y.append(blobs_in_frame[blob][3])
            polygon_of_predicted_blobs.append(blobs_in_frame[blob][4])
            polygon_of_brightness_contours.append(blobs_in_frame[blob][5])
            frames_of_appearance.append(i)

    # sorting the blobs by blob_id
    tmp = [[blob_ids[i], i] for i in range(len(blob_ids))]
    tmp.sort()
    sort_index = [x[1] for x in tmp]

    blob_ids = [blob_ids[i] for i in sort_index]
    VIoUs = [VIoUs[i] for i in sort_index]
    centers_of_mass_x = [centers_of_mass_x[i] for i in sort_index]
    centers_of_mass_y = [centers_of_mass_y[i] for i in sort_index]
    polygon_of_predicted_blobs = [polygon_of_predicted_blobs[i] for i in sort_index]
    polygon_of_brightness_contours = [
        polygon_of_brightness_contours[i] for i in sort_index
    ]
    frames_of_appearance = [frames_of_appearance[i] for i in sort_index]

    last_blob_id = 0
    list_of_blobs = []

    temp_VIoUs = []
    temp_centers_of_mass_x = []
    temp_centers_of_mass_y = []
    temp_polygon_of_predicted_blobs = []
    temp_polygon_of_brightness_contours = []
    temp_frames_of_appearance = []

    blob_ids.append(np.inf)
    for i in range(len(blob_ids)):

        if blob_ids[i] != last_blob_id:
            blob = Blob(
                raw_data,
                last_blob_id,
                temp_VIoUs,
                temp_centers_of_mass_x,
                temp_centers_of_mass_y,
                temp_polygon_of_predicted_blobs,
                temp_polygon_of_brightness_contours,
                temp_frames_of_appearance,
            )
            list_of_blobs.append(blob)

            temp_VIoUs = []
            temp_centers_of_mass_x = []
            temp_centers_of_mass_y = []
            temp_polygon_of_predicted_blobs = []
            temp_polygon_of_brightness_contours = []
            temp_frames_of_appearance = []

        if blob_ids[i] != np.inf:
            temp_VIoUs.append(VIoUs[i])
            temp_centers_of_mass_x.append(centers_of_mass_x[i])
            temp_centers_of_mass_y.append(centers_of_mass_y[i])
            temp_polygon_of_predicted_blobs.append(polygon_of_predicted_blobs[i])
            temp_polygon_of_brightness_contours.append(
                polygon_of_brightness_contours[i]
            )
            temp_frames_of_appearance.append(frames_of_appearance[i])
        last_blob_id = blob_ids[i]

    list_of_blobs.pop(0)
    pickle.dump(
        list_of_blobs, open(raft_output.replace(".pickle", "_blobs.pickle"), "wb")
    )


if __name__ == "__main__":
    create_blob_list()

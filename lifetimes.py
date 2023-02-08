from cProfile import label
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cosmoplots

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

full_data_rf_on = pickle.load(
    open(
        "/home/gregor/Documents/CMod data/raymond_shots/ICRF_on_1.066_raft_blobs.pickle",
        "rb",
    )
)
full_data_rf_on_2 = pickle.load(
    open(
        "/home/gregor/Documents/CMod data/raymond_shots/ICRF_on_1.072_raft_blobs.pickle",
        "rb",
    )
)
full_data_rf_off = pickle.load(
    open(
        "/home/gregor/Documents/CMod data/raymond_shots/ICRF_off_raft_blobs.pickle",
        "rb",
    )
)

full_data_rf_on += full_data_rf_on_2

blob_ids = [blob.blob_id for blob in full_data_rf_on]

print(f"Number of blobs rf on: {len(blob_ids)}")

blob_ids = [blob.blob_id for blob in full_data_rf_off]
print(f"Number of blobs rf off: {len(blob_ids)}")

rf_on = {}
rf_off = {}

dict_list = [rf_on, rf_off]
blob_lists = [full_data_rf_on, full_data_rf_off]

for (dictionary, blob_list) in zip(dict_list, blob_lists):
    dictionary["lifetimes"] = [blob.life_time for blob in blob_list]

# mean_elongation_rf_on = [np.mean(blob) for blob in rf_on["sizes"]]
# mean_elongation_rf_off = [np.mean(blob) for blob in rf_off["sizes"]]


print(f"rf_on mean lifetime: {np.mean(rf_on['lifetimes'])}")
print(f"rf_off mean lifetime: {np.mean(rf_off['lifetimes'])}")
plt.hist(
    rf_on["lifetimes"], label="RF off", alpha=0.5, density=True, bins=32,
)
plt.hist(rf_off["lifetimes"], label="RF on", alpha=0.5, density=True, bins=32)
plt.xlabel(r"$\mathrm{blob\, sizes\, [cm^2]}$")
plt.ylabel(r"$\mathrm{P(blob\, sizes)}$")
plt.legend()
plt.show()

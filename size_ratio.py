from cProfile import label
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cosmoplots

axes_size = cosmoplots.set_rcparams_dynamo(plt.rcParams, num_cols=1, ls="thin")

full_data_rf_on = pickle.load(
    open("/home/gregor/Documents/CMod data/raymond_shots/115090424_2_raft_blobs.pickle", "rb")
)
full_data_rf_off = pickle.load(
    open("/home/gregor/Documents/CMod data/raymond_shots/115090428_raft_blobs.pickle", "rb")
)

blob_ids = [blob.blob_id for blob in full_data_rf_on]
print(f"Number of blobs: {len(blob_ids)}")


rf_on = {}
rf_off = {}

dict_list = [rf_on,rf_off]
blob_lists = [full_data_rf_on,full_data_rf_off]

for (dictionary, blob_list) in zip(dict_list, blob_lists):
    dictionary["lifetimes"] = [blob.life_time for blob in blob_list]

    dictionary["com_x"] = [blob._centers_of_mass_x for blob in blob_list]
    dictionary["com_y"] = [blob._centers_of_mass_y for blob in blob_list]
    dictionary["width_x"] = [blob.width_x for blob in blob_list]
    dictionary["width_y"] = [blob.width_y for blob in blob_list]
    dictionary['size_ratio'] = [np.array(blob.width_Z)/np.array(blob.width_R) for blob in blob_list]


mean_ratios_rf_on = [np.mean(blob) for blob in rf_on["size_ratio"]]
mean_ratios_rf_off = [np.mean(blob) for blob in rf_off["size_ratio"]]
number_blobs_rf_on = np.arange(len(mean_ratios_rf_on))
number_blobs_rf_off = np.arange(len(mean_ratios_rf_off))

plt.hist(mean_ratios_rf_off, label='RF off',alpha=0.5,density=True, range=(0,5),bins=32)
plt.hist(mean_ratios_rf_on, label='RF on', alpha=0.5,density=True,range=(0,5),bins=32)
plt.ylabel(r'$P(\overline{\delta_Z}/\overline{\delta_R})$')
plt.xlabel(r'$\overline{\delta_Z}/\overline{\delta_R}$')
plt.legend()
plt.savefig("blob_elongation_hist.pdf",bbox_inches="tight")
plt.show()
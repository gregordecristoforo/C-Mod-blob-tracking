import pickle
import matplotlib.pyplot as plt
import numpy as np

blob_list = []
files = [
    "data/1091216028_full_data/1091216028_1.4_01_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_02_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_03_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_04_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_05_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_06_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_07_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_08_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_09_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_10_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_11_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_12_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_13_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_14_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_17_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_18_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_19_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_20_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_21_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_22_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_23_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_24_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_25_raft_blobs.pickle",
]
output_file = "data/1091216028_full_data/1091216028_all_blobs.pickle"
for file in files:
    blob_list.extend(pickle.load(open(file, "rb")))

pickle.dump(blob_list, open(output_file, "wb"))

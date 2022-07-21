import pickle
import matplotlib.pyplot as plt
import numpy as np

blob_list = []
files = [
    "data/1091216028_full_data/1091216028_1.4_01_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_02_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_03_raft_blobs.pickle",
    "data/1091216028_full_data/1091216028_1.4_04_raft_blobs.pickle",
]
output_file = "data/1091216028_full_data/1091216028_all_blobs.pickle"
for file in files:
    blob_list.extend(pickle.load(open(file, "rb")))

pickle.dump(blob_list, open(output_file, "wb"))


blob_list = pickle.load(
    open("data/1091216028_full_data/1091216028_1.4_02_raft_blobs.pickle", "rb")
)

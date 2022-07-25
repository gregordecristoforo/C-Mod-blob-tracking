import pickle
import numpy as np
import matplotlib.pyplot as plt
import pickle

blobs_harry = pickle.load(open("data/blob_stats_1091216028_1.45_raft.pickle", "rb"))
blobs_gregor = pickle.load(open("data/1091216028_1.45_raft_blobs.pickle", "rb"))

keys = list(blobs_harry.keys())

size_max_harry = []
size_mean_harry = []
size_mean_gregor = []
size_max_gregor = []

for i in range(98):
    key = keys[i]
    size_H = blobs_harry[key]["size_r"]
    # blobs_gregor[i].smoothen_all_parameters(window_length = 5, polyorder = 1)
    size_G = blobs_gregor[i].width_R
    size_max_harry.append(np.max(size_H))
    size_mean_harry.append(np.mean(size_H))
    size_mean_gregor.append(np.mean(size_G))
    size_max_gregor.append(np.max(size_G))

plot_range = np.linspace(0, 0.04, 100)
hist, bin_edges = np.histogram(size_max_harry, bins=plot_range)
plt.scatter(bin_edges[:-1], hist, label="Harry")
hist, bin_edges = np.histogram(size_max_gregor, bins=plot_range)
plt.scatter(bin_edges[:-1], hist, label="Gregor")
plt.xlabel("max size [m]")
plt.legend()

plt.figure()
plt.xlabel("mean size [m]")
hist, bin_edges = np.histogram(size_mean_harry, bins=plot_range)
plt.scatter(bin_edges[:-1], hist, label="Harry")
hist, bin_edges = np.histogram(size_mean_gregor, bins=plot_range)
plt.scatter(bin_edges[:-1], hist, label="Gregor")
plt.legend()
plt.show()

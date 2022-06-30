from blob import Blob
import pickle
import matplotlib.pyplot as plt

blob_list = pickle.load(open("list_of_blobs.pickle", "rb"))

lifetimes = [blob.life_time for blob in blob_list]

plt.hist(lifetimes)
plt.show()

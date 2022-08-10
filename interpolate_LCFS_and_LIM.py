import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

R_LCFS = np.load("data/R_LCFS.npy")
Z_LCFS = np.load("data/Z_LCFS.npy")
R_LIM = np.load("data/R_LIM.npy")
Z_LIM = np.load("data/Z_LIM.npy")

f_LCFS = interpolate.interp1d(Z_LCFS, R_LCFS, kind="cubic",)
f_LIM = interpolate.interp1d(Z_LIM, R_LIM, kind="cubic",)

z_fine = np.linspace(-0.08, 0.01, 1000)
R_LCFS = f_LCFS(z_fine)
R_LIM = f_LIM(z_fine)

np.savetxt("data/LCFS_interpolated.txt", np.c_[R_LCFS, z_fine])
np.savetxt("data/LIM_interpolated.txt", np.c_[R_LIM, z_fine])

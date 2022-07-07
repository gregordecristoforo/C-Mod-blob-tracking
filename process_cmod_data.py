import numpy as np
import netCDF4
from scipy.interpolate import Rbf
import bz2
import pickle
import _pickle as cPickle
import copy

shot = 1100721028
subtr = True
filename = '/nobackup/users/harryhan/2_Data/cmod_shot/shot_' + str(shot) + '/shot_1_100_721_028.nc'

shot = 1091216028
subtr = False
filename = '/nobackup/users/harryhan/2_Data/cmod_shot/shot_' + str(shot) + '/short_dataset_coordinates_included.nc'

if subtr:
    save_file = '/nobackup/users/harryhan/2_Data/cmod_shot/shot_' + str(shot) + '/cmod_shot_' + str(shot) + '_processed_subtr.pbz2'
else:
    save_file = '/nobackup/users/harryhan/2_Data/cmod_shot/shot_' + str(shot) + '/cmod_shot_' + str(shot) + '_processed_nosubtr.pbz2'

R_LCFS = np.load('/nobackup/users/harryhan/2_Data/cmod_shot/shot_' + str(shot) + '/R_LCFS.npy')
Z_LCFS = np.load('/nobackup/users/harryhan/2_Data/cmod_shot/shot_' + str(shot) + '/Z_LCFS.npy')

file2read = netCDF4.Dataset(filename, 'r')
frames = np.transpose(np.asarray(file2read.variables['frames']), (2,1,0))[::-1,:,:]
R = 0.01*np.transpose(np.asarray(file2read.variables['R']), (1,0))
Z = 0.01*np.transpose(np.asarray(file2read.variables['Z']), (1,0))
time = np.asarray(file2read.variables['time'])

t_ref = np.round(time[0], 2)

if len(time) > 2000:
    idx_start = len(time)//2
    frames = frames[:,:,idx_start:idx_start+1000]
    time = time[idx_start:idx_start+1000]


minR, maxR = np.min(R), np.max(R)
minZ, maxZ = np.min(Z), np.max(Z)
slope_R = 1./(maxR-minR)
offset_R = -slope_R*minR
slope_Z = 1./(maxZ-minZ)
offset_Z = -slope_Z*minZ

X = slope_R*R + offset_R
Y = slope_Z*Z + offset_Z
X_LCFS = slope_R*R_LCFS + offset_R
Y_LCFS = slope_Z*Z_LCFS + offset_Z

grid = np.linspace(0., 1., 256)
shear_contour_y = np.array(range(256))
shear_contour_x = np.array([]).astype(int)

blob_mask = np.zeros((1, 256, 256, len(time)))
for i in shear_contour_y:
    x = X_LCFS[np.argmin(np.abs(Y_LCFS - grid[i]))]
    idx_x_shear = np.argmin(np.abs(grid - x))
    shear_contour_x = np.append(shear_contour_x, idx_x_shear)
    blob_mask[0, :idx_x_shear+1, i, :] = np.ones((idx_x_shear+1, np.shape(blob_mask)[3]))





data_input = {'brt_arr':frames, 'r_arr':R, 'z_arr':Z, 'shear_contour_x':shear_contour_x, 'shear_contour_y':shear_contour_y}
with open('/nobackup/users/harryhan/GPI-blob-tracking/data/real_gpi/' + str(shot) + '_' + str(t_ref) + '_raw.pickle', 'wb') as handle:
    pickle.dump(data_input, handle, protocol=pickle.HIGHEST_PROTOCOL)



'''
brt_true = np.zeros((256, 256, len(time)))
y_grid_up, x_grid_up = np.meshgrid(np.linspace(0., 1., 256), np.linspace(0., 1., 256))
for t in range(len(time)):
    brt_true[:,:,t] = Rbf(X, Y, frames[:,:,t], function='cubic')(x_grid_up, y_grid_up)

data_input = {'brt_true':brt_true, 'shear_contour_x':shear_contour_x, 'shear_contour_y':shear_contour_y, 'blob_mask':blob_mask}


def process(dataset, n_x_up=256, n_y_up=256):
    def normalize_brt(brt, idx_shear_x, idx_shear_y):
        mean_brt_arr = np.repeat(np.mean(brt, axis=2)[:, :, np.newaxis], np.shape(brt)[2], axis=2)
        std_view = np.std(brt, axis=2)
        if subtr:
            brt = (brt - mean_brt_arr) / np.repeat(std_view[:, :, np.newaxis], np.shape(brt)[2], axis=2)
            max_brt_outside = np.array([])
            min_brt_outside = np.array([])
            for i in range(10, len(idx_shear_x)-10):
                max_brt_outside = np.append(max_brt_outside, np.max(brt[10:-10, 10:-10, :][idx_shear_x[i]-10:, idx_shear_y[i]-10, :], axis=1))
                min_brt_outside = np.append(min_brt_outside, np.min(brt[10:-10, 10:-10, :][idx_shear_x[i]-10:, idx_shear_y[i]-10, :], axis=1))
            
            brt_upper_cap = np.mean(max_brt_outside)
            brt_lower_cap = np.mean(min_brt_outside)
            
            #brt = np.minimum(brt, brt_upper_cap)
            #brt = np.maximum(brt, brt_lower_cap)
            #brt -= np.min(brt)
            #brt /= np.max(brt)
            #brt /= brt_upper_cap
            brt -= brt_lower_cap
            brt /= (brt_upper_cap - brt_lower_cap)
        else:
            max_brt_outside = np.array([])
            min_brt_outside = np.array([])
            for i in range(10, len(idx_shear_x)-10):
                max_brt_outside = np.append(max_brt_outside, np.max(brt[10:-10, 10:-10, :][idx_shear_x[i]-10:, idx_shear_y[i]-10, :], axis=1))
                min_brt_outside = np.append(min_brt_outside, np.min(brt[10:-10, 10:-10, :][idx_shear_x[i]-10:, idx_shear_y[i]-10, :], axis=1))
            
            brt_upper_cap = np.mean(max_brt_outside)
            brt_lower_cap = np.mean(min_brt_outside)
            brt /= brt_upper_cap
        
        return brt
    
    images = []
    objects = []
    #Iterate over the data files, and the data will be concatenated at the end.
    for i_data, data in enumerate(dataset):
        brt_true = data['brt_true'] #The true video data containing value 1 for a single point of blob and 0 otherwise. Size: (number of x) X (number of y) X (number of t)
        n_x, n_y, n_t = np.shape(brt_true)
        brt_true = normalize_brt(copy.deepcopy(brt_true), data['shear_contour_x'], data['shear_contour_y'])
        
        blob_mask = data['blob_mask'] #Size: (number of objects) X (number of x) X (number of y) X (number of t)
        
        images_i = []
        #Iterate over the frames, and the data will be concatenated at the end.
        for idx, t in enumerate(range(n_t)):
            #The class label 1 is for the region inside of the shear layer.
            x1, x2 = 0, np.max(data['shear_contour_x'])
            y1, y2 = np.min(data['shear_contour_y']), np.max(data['shear_contour_y'])
            boxes = np.array([[x1, y1, x2, y2]])
            labels = [1]
            num_objs = 1
            obj_idx = [0]
            
            #Up-sampling of the data to match the data size (12) X (10) X (number of t) with the label size (number of x) X (number of y) X (number of t)
            blob_mask_t = blob_mask[:,:,:,t]
            frame_upsampled = brt_true[:,:,t]
            
            #The up-sampled frame is transposed and prepended with two additional dimensions, to match with the desired input size (number of datapoints) X (3 RGB) X (number of y) X (number of x)
            frame_upsampled_3chan = np.repeat(frame_upsampled.T[np.newaxis, np.newaxis, :, :], 3, axis=1)
            images_i = np.concatenate([images_i, frame_upsampled_3chan], axis=0) if len(images_i) > 0 else frame_upsampled_3chan
        
        images = np.concatenate([images, images_i], axis=0) if len(images) > 0 else images_i
    
    return images






brt_true = data_input['brt_true']
n_x, n_y, n_t = np.shape(brt_true)

images = process([data_input], n_x_up=n_x, n_y_up=n_y)

output_dict = {'normbrt':np.transpose(images[:,0,:,:], (0,2,1)), 'shear_contour_x':shear_contour_x, 'shear_contour_y':shear_contour_y}
with bz2.BZ2File(save_file, 'w') as f:
    cPickle.dump(output_dict, f)

'''

import pickle
import numpy as np
from matplotlib.path import Path
import scipy.io as spio
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.interpolate import Rbf
from scipy import optimize
from scipy.optimize import curve_fit
import shapely
from shapely.geometry import Polygon
import argparse
import os
from glob import glob

def get_poly_mask(points, poly, n_x, n_y):
    mask = np.zeros((n_x, n_y))
    if poly.type == 'MultiPolygon':
        for poly_i in list(poly):
            x, y = poly_i.exterior.coords.xy
            mask[Path([(x[j], y[j]) for j in range(len(x))]).contains_points(points).reshape(n_x, n_y)] = 1.
    else:
        x, y = poly.exterior.coords.xy
        mask[Path([(x[j], y[j]) for j in range(len(x))]).contains_points(points).reshape(n_x, n_y)] = 1.
    return mask.astype(bool)

def connect_two_points(p1x, p1y, p2x, p2y, xgrid, ygrid, mask_arr):
    slope = (p2y - p1y)/(p2x - p1x)
    b = ((p1y + p2y) - slope*(p1x + p2x))/2.
    x_start = np.min([p1x, p2x])
    x_end = np.max([p1x, p2x])
    for xind in range(np.argmin(np.abs(xgrid - x_start)), np.argmin(np.abs(xgrid - x_end)) + 1):
        y = slope*xgrid[xind] + b
        mask_arr[xind, np.argmin(np.abs(ygrid - y))] = True
    return mask_arr

def re_rotate(r, z, cm_r, cm_z, angle):
    return (cm_r + r*np.cos(-angle) + z*np.sin(-angle), cm_z -r*np.sin(-angle) + z*np.cos(-angle))

def max_size(blob_pts_1, blob_pts_2, grid, n_1):
    binsize = 4. * (np.max(grid) - np.min(grid)) / n_1
    bins = np.linspace(np.min(blob_pts_1), np.max(blob_pts_1), int((np.max(blob_pts_1) - np.min(blob_pts_1))/binsize))
    size = 0.
    for i in range(len(bins) - 1):
        i_2 = blob_pts_2[(blob_pts_1 > bins[i]) & (blob_pts_1 <= bins[i+1])]
        if len(i_2) == 0:
            continue
        size_i = (np.max(i_2) - np.min(i_2)) / 2.
        if size_i > size:
            size = size_i
    return size

def estimate_blob_stats(args):
    mat_dat = spio.loadmat(args.path + args.shot + '_rho.mat')
    rho = np.transpose(mat_dat['rho_highres'], axes=[2,1,0])
    t_rho = mat_dat['t_rho'].T[0]
    rho_tref = rho[np.argmin(np.abs(t_rho - float(args.time))), :, :]
    r_grid = mat_dat['R_grid'][0]
    rho_grid = mat_dat['rho_grid'][0][:, np.argmin(np.abs(t_rho - float(args.time)))]
    r_highres = mat_dat['R_highres'][0]
    z_highres = mat_dat['Z_highres'][0]
    z_grid_frame, r_grid_frame = np.meshgrid(z_highres, r_highres)
    
    k_b = 1.380649e-23
    m_i = 2*1.67262192369e-27
    gamma = 1.
    e = 1.602176634e-19
    c = 299792458.
    eps_x = 0.5
    mat_dat = spio.loadmat(args.path + args.shot + '_' + args.time  + '_myra.mat')
    rho_prof = mat_dat['rho_prof'][0]
    ne_prof = mat_dat['ne_prof'][0]
    te_prof = mat_dat['te_prof'][0]
    r_lp = mat_dat['r_lp']
    z_lp = mat_dat['z_lp']
    lp_up = mat_dat['lp_up']
    lp_down = mat_dat['lp_down']
    r_bphi = mat_dat['r_bphi']
    z_bphi = mat_dat['z_bphi']
    bphi = mat_dat['bphi']
    bphi_fit_func = Rbf(r_bphi, z_bphi, bphi, function='cubic')
    def tanh_fit_func(x, a, b, c):
        return a * (np.tanh(b * (x - c)) + 1.0)
    
    te_fit_params, te_fit_params_covariance = optimize.curve_fit(tanh_fit_func, rho_prof, te_prof, p0=[np.max(te_prof), -10., 1.])
    ne_fit_params, ne_fit_params_covariance = optimize.curve_fit(tanh_fit_func, rho_prof, ne_prof, p0=[np.max(ne_prof), -10., 1.])
    
    n_x, n_y = 256, 256
    x_frame = np.linspace(0., 1., n_x)
    y_frame = np.linspace(0., 1., n_y)
    y_grid, x_grid = np.meshgrid(y_frame, x_frame)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    drdx, drdy = np.gradient(rho_tref)
    
    blob_info = {}
    durations, waitings = {}, {}
    durations['value'], durations['rho'], durations['r'], durations['z'] = np.array([]), np.array([]), np.array([]), np.array([])
    waitings['value'], waitings['rho'], waitings['r'], waitings['z'] = np.array([]), np.array([]), np.array([]), np.array([])
    last_ending_times = np.zeros((n_x, n_y)) - 1.
    
    filenames = sorted(glob(os.path.join(args.path, args.shot + '_' + args.time + '_' + '*_' + args.model + '.pickle')))
    for seg, filename in enumerate(filenames):
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        
        output = data['output']
        output_tracking = data['output_tracking']
        
        if args.model == 'mrcnn':
            brt = output
            n_x = np.shape(output)[1]
            n_y = np.shape(output)[2]
        else:
            brt = 1. - output[:, :n_x, :, 0]
            n_x = np.shape(output)[1]//2
            n_y = np.shape(output)[2]
        
        times = list(output_tracking.keys())
        current_ids = []
        #pulse_start_time = np.zeros((n_x, n_y)) - 1.
        #t_prev = -1.
        for t in times:
            if output_tracking[t]:
                masks = np.zeros((n_x, n_y)).astype(bool)
                for tracking in output_tracking[t]:
                    blob_id, viou, cx, cy, polygon_pred, _ = tracking
                    blob_id = 1000*seg + blob_id
                    if blob_id not in current_ids:
                        current_ids.append(blob_id)
                    
                    mask = get_poly_mask(points, polygon_pred, n_x, n_y)
                    masks = masks | mask
                    
                    cm_r = (cx/n_x)*(np.max(r_highres) - np.min(r_highres)) + np.min(r_highres)
                    cm_z = (cy/n_y)*(np.max(z_highres) - np.min(z_highres)) + np.min(z_highres)
                    idx_cm_r = np.argmin(np.abs(r_grid_frame[:,0] - cm_r))
                    idx_cm_z = np.argmin(np.abs(z_grid_frame[0,:] - cm_z))
                    slope = drdy[idx_cm_r, idx_cm_z] / drdx[idx_cm_r, idx_cm_z]
                    angle = np.arctan(slope)
                    r_grid_rot = (r_grid_frame - cm_r) * np.cos(angle) + (z_grid_frame - cm_z) * np.sin(angle)
                    z_grid_rot = -(r_grid_frame - cm_r) * np.sin(angle) + (z_grid_frame - cm_z) * np.cos(angle)
                    
                    if polygon_pred.type == 'MultiPolygon':
                        print('Blob ' + str(blob_id) + ' at t = ' + str(t) + ' is a Multipolygon.')
                        r_mins, r_maxs, z_mins, z_maxs = [], [], [], []
                        valid_poly_i = []
                        for i_poly, poly in enumerate(polygon_pred):
                            mask_i = get_poly_mask(points, poly, n_x, n_y)
                            if mask_i.any():
                                blob_pts_r = r_grid_rot[mask_i]
                                blob_pts_z = z_grid_rot[mask_i]
                                r_mins.append(re_rotate(np.min(blob_pts_r), blob_pts_z[np.argmin(blob_pts_r)], cm_r, cm_z, angle))
                                r_maxs.append(re_rotate(np.max(blob_pts_r), blob_pts_z[np.argmax(blob_pts_r)], cm_r, cm_z, angle))
                                z_mins.append(re_rotate(blob_pts_r[np.argmin(blob_pts_z)], np.min(blob_pts_z), cm_r, cm_z, angle))
                                z_maxs.append(re_rotate(blob_pts_r[np.argmax(blob_pts_z)], np.max(blob_pts_z), cm_r, cm_z, angle))
                                valid_poly_i.append(i_poly)
                        
                        valid_polygons = [polygon_pred[i] for i in valid_poly_i]
                        for i in range(len(valid_polygons)):
                            for j in range(len(valid_polygons)):
                                if i < j:
                                    mask = connect_two_points(r_mins[i][0], r_mins[i][1], r_mins[j][0], r_mins[j][1], r_highres, z_highres, mask)
                                    mask = connect_two_points(r_maxs[i][0], r_maxs[i][1], r_maxs[j][0], r_maxs[j][1], r_highres, z_highres, mask)
                                    mask = connect_two_points(z_mins[i][0], z_mins[i][1], z_mins[j][0], z_mins[j][1], r_highres, z_highres, mask)
                                    mask = connect_two_points(z_maxs[i][0], z_maxs[i][1], z_maxs[j][0], z_maxs[j][1], r_highres, z_highres, mask)
                    
                    blob_pts_r = r_grid_rot[mask]
                    blob_pts_z = z_grid_rot[mask]
                    
                    size = max_size(blob_pts_r, blob_pts_z, r_highres, n_x)
                    size_2 = max_size(blob_pts_z, blob_pts_r, z_highres, n_y)
                    
                    maxbrt = np.max(brt[t,:,:][mask])
                    if blob_id not in blob_info:
                        blob_info[blob_id] = {'t':[t], 'cx':[cx], 'cy':[cy], 'cm_r':[cm_r], 'cm_z':[cm_z], 'size':[size], 'size_2':[size_2], 'viou':[viou], 'mask':[mask], 'angle':[angle], 'maxbrt':[maxbrt]}
                    else:
                        blob_info[blob_id]['t'].append(t)
                        blob_info[blob_id]['cx'].append(cx)
                        blob_info[blob_id]['cy'].append(cy)
                        blob_info[blob_id]['cm_r'].append(cm_r)
                        blob_info[blob_id]['cm_z'].append(cm_z)
                        blob_info[blob_id]['size'].append(size)
                        blob_info[blob_id]['size_2'].append(size_2)
                        blob_info[blob_id]['viou'].append(viou)
                        blob_info[blob_id]['mask'].append(mask)
                        blob_info[blob_id]['angle'].append(angle)
                        blob_info[blob_id]['maxbrt'].append(maxbrt)
                
                #if t_prev + 1 < t:
                #    durations_valid = np.append(durations_valid, t_prev - pulse_start_time[pulse_start_time != -1.])
                #    pulse_start_time = np.zeros((n_x, n_y)) - 1.
                
                #pulse_start_time[(masks) & (pulse_start_time == -1.)] = t
                #durations_valid = np.append(durations_valid, t - pulse_start_time[(~masks) & (pulse_start_time != -1.)])
                #pulse_start_time[(~masks) & (pulse_start_time != -1.)] = -1.
                #t_prev = t
        
        for blob_id in current_ids:
            t_traj = np.array(blob_info[blob_id]['t'])
            if len(t_traj) < 10:
                continue
            cm_r_traj = np.array(blob_info[blob_id]['cm_r'])
            cm_z_traj = np.array(blob_info[blob_id]['cm_z'])
            size_traj = np.array(blob_info[blob_id]['size'])
            size_2_traj = np.array(blob_info[blob_id]['size_2'])
            
            num_pts = 5
            num_pts_oddified = num_pts + num_pts%2 - 1
            window_size, poly_order = num_pts_oddified, 1
            cm_r_traj_smooth = savgol_filter(cm_r_traj, window_size, poly_order)
            cm_z_traj_smooth = savgol_filter(cm_z_traj, window_size, poly_order)
            size_traj_smooth = savgol_filter(size_traj, window_size, poly_order)
            size_2_traj_smooth = savgol_filter(size_2_traj, window_size, poly_order)
            
            blob_info[blob_id]['cm_r'] = cm_r_traj_smooth
            blob_info[blob_id]['cm_z'] = cm_z_traj_smooth
            blob_info[blob_id]['size'] = size_traj_smooth
            blob_info[blob_id]['size_2'] = size_2_traj_smooth
            v_r_traj = []
            v_th_traj = []
            v_R_traj = []
            v_z_traj = []
            t_margin = 5#3#2
            blob_center_pixels = np.zeros((n_x, n_y)).astype(bool)
            mask_traj = np.zeros((len(t_traj), n_x, n_y)).astype(bool)
            for i in range(len(t_traj)):
                if i < t_margin or i >= len(t_traj)-t_margin:
                    v_r_traj.append(np.nan)
                    v_th_traj.append(np.nan)
                    v_R_traj.append(np.nan)
                    v_z_traj.append(np.nan)
                    continue
                
                idx_cm_r = np.argmin(np.abs(r_grid_frame[:,0] - cm_r_traj_smooth[i]))
                idx_cm_z = np.argmin(np.abs(z_grid_frame[0,:] - cm_z_traj_smooth[i]))
                slope = drdy[idx_cm_r, idx_cm_z] / drdx[idx_cm_r, idx_cm_z]
                
                dr = cm_r_traj_smooth[np.min([i+t_margin, len(t_traj)-1])] - cm_r_traj_smooth[np.max([i-t_margin, 0])]
                dz = cm_z_traj_smooth[np.min([i+t_margin, len(t_traj)-1])] - cm_z_traj_smooth[np.max([i-t_margin, 0])]
                t_i = t_traj[np.max([i-t_margin, 0])]
                t_f = t_traj[np.min([i+t_margin, len(t_traj)-1])]
                
                v_r = 2000000.3692258536 * np.cos(np.arctan(dz/dr) - np.arctan(slope)) * np.sqrt(dr**2 + dz**2) / (t_f - t_i)
                v_th = 2000000.3692258536 * np.sin(np.arctan(dz/dr) - np.arctan(slope)) * np.sqrt(dr**2 + dz**2) / (t_f - t_i)
                v_r_traj.append(v_r)
                v_th_traj.append(v_th)
                v_R = 2000000.3692258536 * np.cos(np.arctan(dz/dr)) * np.sqrt(dr**2 + dz**2) / (t_f - t_i)
                v_z = 2000000.3692258536 * np.sin(np.arctan(dz/dr)) * np.sqrt(dr**2 + dz**2) / (t_f - t_i)
                v_R_traj.append(v_R)
                v_z_traj.append(v_z)
                
                #mask = blob_info[blob_id]['mask'][i]
                #yind = ((-dr/dz)*np.array(range(n_x)) + blob_info[blob_id]['cy'][i] + (dr/dz)*blob_info[blob_id]['cx'][i]).astype(int)
                #radius = 0.5*min([size_traj_smooth[i], size_2_traj_smooth[i]])
                radius = 0.2*min([size_traj_smooth[i], size_2_traj_smooth[i]])
                blob_center_pixels[np.where(radius**2 > (r_grid_frame - cm_r_traj_smooth[i])**2 + (z_grid_frame - cm_z_traj_smooth[i])**2)] = True
                mask_traj[i,:,:] = blob_info[blob_id]['mask'][i]
            
            blob_center_pixels[blob_info[blob_id]['mask'][0]] = False
            
            pulse_start_time = np.zeros((n_x, n_y)) - 1.
            pulse_end_time = np.zeros((n_x, n_y)) - 1.
            pulse_start_time_save = np.zeros((n_x, n_y)) - 1.
            buffer = 5
            '''
            for i in range(len(t_traj)):
                mask = blob_info[blob_id]['mask'][i]
                idx_beginning = (mask) & (blob_center_pixels) & (pulse_start_time == -1.)
                idx_ending = (~mask) & (blob_center_pixels) & (pulse_start_time != -1.)
                idx_finished = (idx_beginning) & (pulse_end_time != -1.) & (t_traj[i] - pulse_end_time > buffer) & (pulse_end_time - pulse_start_time_save > buffer)
                idx_new = (idx_beginning) & (pulse_end_time == -1.)
                
                pulse_start_time[idx_beginning] = t_traj[i]
                durations_valid = np.append(durations_valid, pulse_end_time[idx_finished] - pulse_start_time_save[idx_finished])
                durations_rho_valid = np.append(durations_rho_valid, rho_tref[idx_finished])
                #pulse_start_time[idx_transient] = pulse_end_time[idx_transient] - duration_save[idx_transient]
                pulse_start_time_save[(idx_finished) | (idx_new)] = t_traj[i]
                pulse_end_time[idx_beginning] = -1.
                
                pulse_end_time[(pulse_end_time == -1.) & (idx_ending)] = t_traj[i]
                #duration_save[idx_ending] = t_traj[i] - pulse_start_time[idx_ending]
                #durations_valid = np.append(durations_valid, t_traj[i] - pulse_start_time[idx_ending])
                pulse_start_time[idx_ending] = -1.
                
                if i == len(t_traj)-1:
                    #durations_valid = np.append(durations_valid, t_traj[i] - pulse_start_time[(blob_center_pixels) & (pulse_start_time != -1.) & (pulse_start_time < t_traj[i]-1)])
                    idx_last = (pulse_end_time != -1.) & (t_traj[i] - pulse_end_time > buffer) & (pulse_end_time - pulse_start_time_save > buffer)
                    durations_valid = np.append(durations_valid, pulse_end_time[idx_last] - pulse_start_time_save[idx_last])
                    durations_rho_valid = np.append(durations_rho_valid, rho_tref[idx_last])
            '''
            durations_arr = np.zeros((n_x, n_y))
            waitings_arr = np.zeros((n_x, n_y)) - 1.
            mask_covered = np.zeros((n_x, n_y)).astype(bool)
            for i in range(len(t_traj)):
                idx_beginning = (mask_traj[i,:,:]) & (blob_center_pixels) & (pulse_start_time == -1.)
                pulse_start_time[idx_beginning] = t_traj[i]
                idx_waiting = (mask_traj[i,:,:]) & (last_ending_times != -1.) & (waitings_arr == -1.)
                waitings_arr[idx_waiting] = seg*1000 + t_traj[i] - last_ending_times[idx_waiting]
                last_ending_times[idx_waiting] = -1.
                
                idx_ending = (~mask_traj[i,:,:]) & (blob_center_pixels) & (pulse_start_time != -1.)
                idx_ending_smallgap = (idx_ending) & (mask_traj[min(i+buffer, len(t_traj)-1),:,:])
                if len(idx_ending_smallgap) > 0:
                    for j in range(buffer):
                        if i+j < len(t_traj):
                            mask_traj[i+j,:,:][(idx_ending_smallgap) & (~mask_traj[i+j,:,:])] = True
                
                idx_ending = (~mask_traj[i,:,:]) & (blob_center_pixels) & (pulse_start_time != -1.)
                durations_arr[idx_ending] = np.maximum(durations_arr[idx_ending], t_traj[i] - pulse_start_time[idx_ending])
                pulse_start_time[idx_ending] = -1.
                
                idx_wait_begin = (~mask_traj[i,:,:]) & (mask_covered)
                pulse_end_time[idx_wait_begin] = seg*1000 + t_traj[i]
                mask_covered[mask_traj[i,:,:]] = True
            
            last_ending_times[pulse_end_time != -1.] = pulse_end_time[pulse_end_time != -1.]
            durations['value'] = np.append(durations['value'], durations_arr[durations_arr > buffer])
            durations['rho'] = np.append(durations['rho'], rho_tref[durations_arr > buffer])
            durations['r'] = np.append(durations['r'], r_grid_frame[durations_arr > buffer])
            durations['z'] = np.append(durations['z'], z_grid_frame[durations_arr > buffer])
            waitings['value'] = np.append(waitings['value'], waitings_arr[waitings_arr > buffer])
            waitings['rho'] = np.append(waitings['rho'], rho_tref[waitings_arr > buffer])
            waitings['r'] = np.append(waitings['r'], r_grid_frame[waitings_arr > buffer])
            waitings['z'] = np.append(waitings['z'], z_grid_frame[waitings_arr > buffer])
            
            
            v_r_traj_smooth = savgol_filter(v_r_traj, window_size, poly_order)
            v_th_traj_smooth = savgol_filter(v_th_traj, window_size, poly_order)
            v_R_traj_smooth = savgol_filter(v_R_traj, window_size, poly_order)
            v_z_traj_smooth = savgol_filter(v_z_traj, window_size, poly_order)
            
            #for i in range(t_margin):
            #    v_r_traj_smooth[i] = v_r_traj_smooth[t_margin]
            #    v_r_traj_smooth[len(t_traj)-1-i] = v_r_traj_smooth[len(t_traj)-1-t_margin]
            
            blob_info[blob_id]['v_r'] = v_r_traj_smooth
            blob_info[blob_id]['v_th'] = v_th_traj_smooth
            blob_info[blob_id]['v_R'] = v_R_traj_smooth
            blob_info[blob_id]['v_z'] = v_z_traj_smooth
            
            lambda_traj = []
            theta_traj = []
            v_hat_traj = []
            a_hat_traj = []
            size_v_traj = []
            for i in range(len(t_traj)):
                cm_r = cm_r_traj_smooth[i]
                cm_z = cm_z_traj_smooth[i]
                v_r = v_r_traj_smooth[i]
                a_b = size_traj_smooth[i]
                r = cm_r
                
                if not np.isnan(v_R_traj_smooth[i]) and not np.isnan(v_z_traj_smooth[i]):
                    angle = np.pi/2. + np.arctan(v_z_traj_smooth[i]/v_R_traj_smooth[i])
                    r_grid_rot = (r_grid_frame - cm_r) * np.cos(angle) + (z_grid_frame - cm_z) * np.sin(angle)
                    z_grid_rot = -(r_grid_frame - cm_r) * np.sin(angle) + (z_grid_frame - cm_z) * np.cos(angle)
                    blob_pts_r = r_grid_rot[mask]
                    blob_pts_z = z_grid_rot[mask]
                    size_v = max_size(blob_pts_r, blob_pts_z, r_highres, n_x)
                else:
                    size_v = np.nan
                size_v_traj.append(size_v)
                
                idx_cm_r = np.argmin(np.abs(r_grid_frame[:,0] - cm_r))
                idx_cm_z = np.argmin(np.abs(z_grid_frame[0,:] - cm_z))
                rho = rho_tref[idx_cm_r, idx_cm_z]
                
                n_e = tanh_fit_func(rho, ne_fit_params[0], ne_fit_params[1], ne_fit_params[2])
                t_e_eV = tanh_fit_func(rho, te_fit_params[0], te_fit_params[1], te_fit_params[2])
                
                b_phi = np.abs(bphi_fit_func(cm_r, cm_z).item())
                
                idx_r = np.argmin(np.abs(r_lp[:,0] - cm_r))
                idx_z = np.argmin(np.abs(z_lp[0,:] - cm_z))
                l_p = lp_down[idx_r, idx_z]
                
                t_e_K = t_e_eV * 11604.
                c_s = np.sqrt(k_b*(gamma*t_e_K)/m_i)
                rho_s = m_i*c_s / (e*b_phi)
                a_hat = a_b*(r**(1./5.)) / ((l_p**(2./5.))*(rho_s**(4./5.)))
                v_hat = v_r / (c_s*((2.*l_p*(rho_s**2)/(r**3))**(1./5.)))
                Lambda = (1.7e-18)*n_e*l_p / (t_e_eV**2)
                Theta = a_hat**(5./2.)
                
                lambda_traj.append(Lambda)
                theta_traj.append(Theta)
                v_hat_traj.append(v_hat)
                a_hat_traj.append(a_hat)
            
            lambda_traj_smooth = savgol_filter(lambda_traj, window_size, poly_order)
            theta_traj_smooth = savgol_filter(theta_traj, window_size, poly_order)
            v_hat_traj_smooth = savgol_filter(v_hat_traj, window_size, poly_order)
            a_hat_traj_smooth = savgol_filter(a_hat_traj, window_size, poly_order)
            #size_v_traj_smooth = savgol_filter(size_v_traj, window_size, poly_order)
            blob_info[blob_id]['lambda'] = lambda_traj_smooth
            blob_info[blob_id]['theta'] = theta_traj_smooth
            blob_info[blob_id]['v_hat'] = v_hat_traj_smooth
            blob_info[blob_id]['a_hat'] = a_hat_traj_smooth
            blob_info[blob_id]['size_v'] = size_v_traj
    
    return blob_info, durations, waitings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../GPI-blob-tracking/data/real_gpi/')
    parser.add_argument('--shot', default='65472')
    parser.add_argument('--time', default='0.35')
    parser.add_argument('--model', default='raft')
    args = parser.parse_args()
    
    blob_info, durations, waitings = estimate_blob_stats(args)
    data = {'blob_info':blob_info, 'durations':durations, 'waitings':waitings}
    with open('blob_stats_' + args.shot + '_' + args.time + '_' + args.model + '.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)











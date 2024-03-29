import contextlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import List
import numpy as np
from polygon_to_mask import get_poly_mask
import xarray as xr
from scipy.signal import savgol_filter
from scipy import interpolate
import shapely.geometry as geom
from shapely.ops import nearest_points
import os.path


class Blob:
    def __init__(
        self,
        file_name: str,
        blob_id: int,
        VIoU: List[float],
        centers_of_mass_x: List[float],
        centers_of_mass_y: List[float],
        polygon_of_predicted_blobs: List[Polygon],
        polygon_of_brightness_contours: List[Polygon],
        frames_of_appearance: List[int],
    ):
        self._file_name = file_name
        self.blob_id = blob_id
        self._VIoU = VIoU
        self._centers_of_mass_x = centers_of_mass_x
        self._centers_of_mass_y = centers_of_mass_y
        self.center_of_mass_R, self.center_of_mass_Z = self._find_center_of_mass_R_Z()
        self._polygon_of_predicted_blobs = polygon_of_predicted_blobs
        self._polygon_of_brightness_contours = polygon_of_brightness_contours
        self.frames_of_appearance = frames_of_appearance
        self.life_time = len(self._VIoU)
        self._sampling_frequency = self._extract_sampling_frequency()
        self.velocities_x = self._calculate_velocity_x()
        self.velocities_y = self._calculate_velocity_y()
        self.sizes, self.width_x, self.width_y = self._calculate_sizes()
        self.amplitudes = self._calculate_amplitudes()
        self.velocities_R, self.velocities_Z = self._calculate_velocities_R_Z()
        self.width_R, self.width_Z = self._calculate_sizes_R_Z()
        # self.rhos, self.poloidal_positions = self._calculate_rho_poloidal_values()
        # self.velocity_rho = self._calculate_velocity_rho()
        self.smoothen_all_parameters()
        # self.plot_single_frames() # useful for debugging
        # self.remove_blobs_outside_of_SOL()
        # self._remove_unnecessary_properties()

    def __repr__(self) -> str:
        return (
            f"Blob id: {self.blob_id} \n"
            "most important variables: life_time [frames], velocities_R [m/s], velocities_Z [m/s], width_R [m], width_Z [m], sizes [m^3], amplitudes [counts]"
        )

    def _calculate_velocity_x(self):
        if self.life_time == 0:
            return 0
        dx_norm = (
            self._extract_dx() / 4
        )  # /4 because of reducing sampling from 256 to 64
        return np.diff(self._centers_of_mass_x) * self._sampling_frequency * dx_norm

    def _calculate_velocity_y(self):
        if self.life_time == 0:
            return 0
        dy_norm = (
            self._extract_dy() / 4
        )  # /4 because of reducing sampling from 256 to 64
        return np.diff(self._centers_of_mass_y) * self._sampling_frequency * dy_norm

    def _find_center_of_mass_R_Z(self):
        ds = self._load_raw_data()
        x_interpol = (
            64 - np.array(self._centers_of_mass_x) / 4
        )  # x index is inverted in the raw data
        y_interpol = np.array(self._centers_of_mass_y) / 4
        R_values = []
        Z_values = []
        for i in range(len(x_interpol)):
            R_values.append(
                ds.R.interp(x=x_interpol[i], y=y_interpol[i], method="cubic").values
            )
            Z_values.append(
                ds.Z.interp(x=x_interpol[i], y=y_interpol[i], method="cubic").values
            )
        return np.array(R_values), np.array(Z_values)

    def _calculate_velocities_R_Z(self):
        if self.blob_id != 0:
            if self.life_time == 0:
                velocity_R, velocity_Z = 0, 0
            R_values, Z_values = self._find_center_of_mass_R_Z()
            velocity_R = np.diff(R_values * 0.01) * self._sampling_frequency
            velocity_Z = np.diff(Z_values * 0.01) * self._sampling_frequency
            return velocity_R, velocity_Z
        return 0, 0

    def _calculate_sizes(self):
        _sizes = []
        _sizes_x = []
        _sizes_y = []
        for frame in range(len(self._polygon_of_predicted_blobs)):
            mask = get_poly_mask(self._polygon_of_predicted_blobs[frame], 64, 64)
            rows, cols = np.where(mask)
            try:
                size_x = np.max(rows) - np.min(rows)
                size_y = np.max(cols) - np.min(cols)
            except Exception:
                size_x = 0
                size_y = 0
            _sizes.append(
                mask.sum() * self._extract_dx() * self._extract_dy()
            )  # size in m^2
            _sizes_x.append(size_x * self._extract_dx())  # size in m
            _sizes_y.append(size_y * self._extract_dy())  # size in m
        return _sizes, _sizes_x, _sizes_y

    def _calculate_sizes_R_Z(self):

        ds = self._load_raw_data()

        _sizes_R = []
        _sizes_Z = []
        for i in range(len(self.frames_of_appearance)):
            R_grid = ds.R.values
            Z_grid = ds.Z.values
            R_grid = np.flip(R_grid, axis=(1))  # orientation different to frames
            mask = get_poly_mask(self._polygon_of_predicted_blobs[i], 64, 64)
            R_values = R_grid[mask.T]
            Z_values = Z_grid[mask.T]
            try:
                size_R = (np.max(R_values) - np.min(R_values)) * 0.01  # in m
                size_Z = (np.max(Z_values) - np.min(Z_values)) * 0.01  # in m
            except Exception:
                size_R = 0
                size_Z = 0
            _sizes_R.append(size_R)
            _sizes_Z.append(size_Z)
        return _sizes_R, _sizes_Z

    def _calculate_amplitudes(self):
        ds = self._load_raw_data()

        amplitudes = []
        for i in range(len(self.frames_of_appearance)):
            frame = self.frames_of_appearance[i]
            single_frame_data = ds.frames.isel(time=frame).values
            mask = get_poly_mask(self._polygon_of_predicted_blobs[i], 64, 64)
            blob_density = single_frame_data[mask.T]
            try:
                amplitudes.append(np.max(blob_density))
            except Exception:
                amplitudes.append(0)

        return amplitudes

    def plot_single_frames(self):
        ds = self._load_raw_data()

        for i in range(len(self.frames_of_appearance)):
            frame = self.frames_of_appearance[i]
            single_frame_data = ds.frames.isel(time=frame).values
            mask = get_poly_mask(self._polygon_of_predicted_blobs[i], 64, 64)

            plt.contourf(single_frame_data)
            plt.contour(mask.T)
            plt.show()

        return

    def _load_raw_data(self):
        return xr.load_dataset(self._file_name)

    def _extract_sampling_frequency(self):
        ds = self._load_raw_data()
        return 1 / np.diff(ds.time.values)[0]

    def _extract_dx(self):
        ds = self._load_raw_data()
        return np.abs(ds.R.diff("x").values[0, 0]) * 1#0.01  # 0.01 for cm to m conversion

    def _extract_dy(self):
        ds = self._load_raw_data()
        return (
            np.mean(ds.Z.diff("y").values) * 1#0.01
        )  # mean values since min is 0.09920597 and max is 0.099206686

    def smoothen_all_parameters(self, window_length=5, polyorder=1):
        try:
            self.amplitudes = savgol_filter(self.amplitudes, window_length, polyorder)
            self.sizes = savgol_filter(self.sizes, window_length, polyorder)
            self.width_x = savgol_filter(self.width_x, window_length, polyorder)
            self.width_y = savgol_filter(self.width_y, window_length, polyorder)
            self.width_R = savgol_filter(self.width_R, window_length, polyorder)
            self.width_Z = savgol_filter(self.width_Z, window_length, polyorder)
            self.center_of_mass_R = savgol_filter(self.center_of_mass_R, window_length, polyorder)
            self.center_of_mass_Z = savgol_filter(self.center_of_mass_Z, window_length, polyorder)

            self.velocities_x = savgol_filter(
                self.velocities_x, window_length - 2, polyorder
            )
            self.velocities_y = savgol_filter(
                self.velocities_y, window_length - 2, polyorder
            )
            self.velocity_rho = savgol_filter(
                self.velocity_rho, window_length - 2, polyorder
            )
        except Exception:
            print(self.__repr__(), ": blob lifetime to short for savgol filter")

    def remove_frames_close_to_borders(self, max_x=235, min_y=20, max_y=235):
        for i in range(len(self.frames_of_appearance)):
            if (
                self._centers_of_mass_x[i] > max_x
                or self._centers_of_mass_y[i] < min_y
                or self._centers_of_mass_y[i] > max_y
            ):
                self.frames_of_appearance[i] = None
                self.amplitudes[i] = None
                self.sizes[i] = None
                self.width_x[i] = None
                self.width_y[i] = None
                with contextlib.suppress(Exception):
                    self.velocities_x[i] = None
                    self.velocities_y[i] = None
                self.life_time -= 1

    def remove_blobs_outside_of_SOL(self):
        directory = os.path.dirname(self._file_name)
        R_LCFS = np.load(f"{directory}/R_LCFS.npy") * 100
        Z_LCFS = np.load(f"{directory}/Z_LCFS.npy") * 100
        R_LIM = np.load(f"{directory}/R_LIM.npy") * 100
        Z_LIM = np.load(f"{directory}/Z_LIM.npy") * 100

        f_LCFS = interpolate.interp1d(Z_LCFS, R_LCFS, kind="cubic",)
        f_LIM = interpolate.interp1d(Z_LIM, R_LIM, kind="cubic",)
        R_values, Z_values = self._find_center_of_mass_R_Z()

        for i in range(len(R_values)):
            local_R_LCFS = f_LCFS(Z_values[i])
            local_R_LIM = f_LIM(Z_values[i])

            if R_values[i] < local_R_LCFS or R_values[i] > local_R_LIM:
                self.frames_of_appearance[i] = None
                self.amplitudes[i] = None
                self.sizes[i] = None
                self.width_x[i] = None
                self.width_y[i] = None
                self.width_R[i] = None
                self.width_Z[i] = None
                # self.rhos[i] = None
                with contextlib.suppress(Exception):
                    self.velocities_x[i] = None
                    self.velocities_y[i] = None
                    self.velocities_R[i] = None
                    self.velocities_Z[i] = None
                self.life_time -= 1

    def _calculate_rho_poloidal_values(self):
        directory = os.path.dirname(self._file_name)
        
        R_LCFS = np.load(f"{directory}/R_LCFS.npy")
        Z_LCFS = np.load(f"{directory}/Z_LCFS.npy")
        R_LIM = np.load(f"{directory}/R_limiter.npy")
        Z_LIM = np.load(f"{directory}/Z_limiter.npy")
        LIM_coords = np.vstack((R_LIM, Z_LIM)).T
        LCFS_coords = np.vstack((R_LCFS, Z_LCFS)).T
        
        LCFS = geom.LineString(LCFS_coords)
        LIM = geom.LineString(LIM_coords)
        rhos = []
        poloidal_positions = []

        R_values, Z_values = self._find_center_of_mass_R_Z()
        if np.sum(np.isnan(R_values)) > 0:
            print("nan detected in R_values, all rho values are set to 0")
            return np.zeros(len(R_values)), np.zeros(len(R_values))

        for R, Z in zip(R_values, Z_values):
            point = geom.Point(R * 0.01, Z * 0.01)  # convert to m
            rho = self._calculate_rho_in_cm(LCFS, point)
            rhos.append(rho)
            poloidal_position = nearest_points(LCFS, point)
            poloidal_positions.append(poloidal_position)
        return rhos, poloidal_positions

    def _calculate_rho(self, LCFS, LIM, point):
        # sourcery skip: raise-specific-error
        nearest_point_on_LCFS, _ = nearest_points(LCFS, point)
        nearest_point_on_LIM, _ = nearest_points(LIM, point)
        LCFS_distance = point.distance(LCFS)
        LIM_distance = point.distance(LIM)

        if nearest_point_on_LCFS.x > point.x:
            """blob inside LCFS"""
            rho = -LCFS_distance / (LIM_distance - LCFS_distance)
        elif nearest_point_on_LCFS.x < point.x and nearest_point_on_LIM.x > point.x:
            """blob in SOL"""
            rho = LCFS_distance / (LIM_distance + LCFS_distance)
        elif nearest_point_on_LIM.x < point.x:
            """blob in limiter shadow"""
            rho = LCFS_distance / (LCFS_distance - LIM_distance)
        else:
            raise Exception("Blob position not determined correctly")

        return rho

    def _calculate_rho_in_cm(self,LCFS, point):
        # sourcery skip: raise-specific-error
        nearest_point_on_LCFS, _ = nearest_points(LCFS, point)
        LCFS_distance = point.distance(LCFS)

        if nearest_point_on_LCFS.x > point.x:
            """blob inside LCFS"""
            rho = -LCFS_distance *100
        elif nearest_point_on_LCFS.x < point.x:
            """blob in SOL"""
            rho = LCFS_distance * 100 
        else:
            raise Exception("Blob position not determined correctly")

        return rho
        
    def _calculate_velocity_rho(self):
        if self.life_time == 0:
            return 0
        return np.diff(self.rhos) * self._sampling_frequency

    def _remove_unnecessary_properties(self):
        self._VIoU = None
        self._polygon_of_brightness_contours = None

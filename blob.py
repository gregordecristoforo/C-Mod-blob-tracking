import contextlib
from shapely.geometry import Polygon
from typing import List
import numpy as np
from polygon_to_mask import get_poly_mask
import xarray as xr
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


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
        self._polygon_of_predicted_blobs = polygon_of_predicted_blobs
        self._polygon_of_brightness_contours = polygon_of_brightness_contours
        self.frames_of_appearance = frames_of_appearance
        self.life_time = len(self._VIoU)
        self._sampling_frequency = self._extract_sampling_frequency()
        self.velocities_x = self._calculate_velocity_x()
        self.velocities_y = self._calculate_velocity_y()
        self.sizes, self.width_x, self.width_y = self._calculate_sizes()
        self.amplitudes = self._calculate_amplitudes()

    def __repr__(self) -> str:
        return f"Blob with blob_id: {self.blob_id}"

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

    def _calculate_sizes(self):
        _sizes = []
        _sizes_x = []
        _sizes_y = []
        for frame in range(len(self._polygon_of_predicted_blobs)):
            mask = get_poly_mask(self._polygon_of_predicted_blobs[frame], 64, 64)
            rows, cols = np.where(mask)
            size_x = np.max(rows) - np.min(rows)
            size_y = np.max(cols) - np.min(cols)
            _sizes.append(
                mask.sum() * self._extract_dx() * self._extract_dy()
            )  # size in m^2
            _sizes_x.append(size_x * self._extract_dx())  # size in m
            _sizes_y.append(size_y * self._extract_dy())  # size in m
        return _sizes, _sizes_x, _sizes_y

    def _calculate_amplitudes(self):
        ds = self._load_raw_data()

        amplitudes = []
        for i, frame in enumerate(self.frames_of_appearance):
            single_frame_data = ds.frames.isel(time=frame).values
            mask = get_poly_mask(self._polygon_of_predicted_blobs[i], 64, 64)
            blob_density = single_frame_data[mask.T]
            amplitudes.append(np.max(blob_density))

        return amplitudes

    def _load_raw_data(self):
        return xr.load_dataset(self._file_name)

    def _extract_sampling_frequency(self):
        ds = self._load_raw_data()
        return 1 / np.diff(ds.time.values)[0]

    def _extract_dx(self):
        ds = self._load_raw_data()
        return np.abs(ds.R.diff("x").values[0, 0]) * 0.01  # 0.01 for cm to m conversion

    def _extract_dy(self):
        ds = self._load_raw_data()
        return (
            np.mean(ds.Z.diff("y").values) * 0.01
        )  # mean values since min is 0.09920597 and max is 0.099206686

    def smoothen_all_parameters(self, window_length=5, polyorder=1):
        try:
            self.amplitudes = savgol_filter(self.amplitudes, window_length, polyorder)
            self.sizes = savgol_filter(self.sizes, window_length, polyorder)
            self.width_x = savgol_filter(self.width_x, window_length, polyorder)
            self.width_y = savgol_filter(self.width_y, window_length, polyorder)

            self.velocities_x = savgol_filter(
                self.velocities_x, window_length - 2, polyorder
            )
            self.velocities_y = savgol_filter(
                self.velocities_y, window_length - 2, polyorder
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

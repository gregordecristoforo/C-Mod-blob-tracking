from shapely.geometry import Polygon
from typing import List
import numpy as np
from polygon_to_mask import get_poly_mask
import xarray as xr
from scipy.signal import savgol_filter


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
        self.file_name = file_name.replace("_raft", "")
        self.blob_id = blob_id
        self.VIoU = VIoU
        self.centers_of_mass_x = centers_of_mass_x
        self.centers_of_mass_y = centers_of_mass_y
        self.polygon_of_predicted_blobs = polygon_of_predicted_blobs
        self.polygon_of_brightness_contours = polygon_of_brightness_contours
        self.frames_of_appearance = frames_of_appearance
        self.life_time = len(self.VIoU)
        self._sampling_frequency = self._extract_sampling_frequency()
        self.velocities_R = self._calculate_velocity_R()
        self.velocities_Z = self._calculate_velocity_Z()
        self.sizes = self._calculate_sizes()
        self.amplitudes = self._calculate_amplitudes()

    def __repr__(self) -> str:
        return f"Blob with blob_id: {self.blob_id}"

    def _calculate_velocity_R(self):
        if self.life_time == 0:
            return 0
        dx_norm = (
            self._extract_dx() / 4
        )  # /4 because of reducing sampling from 256 to 64
        return np.diff(self.centers_of_mass_x) * self._sampling_frequency * dx_norm

    def _calculate_velocity_Z(self):
        if self.life_time == 0:
            return 0
        dy_norm = (
            self._extract_dy() / 4
        )  # /4 because of reducing sampling from 256 to 64
        return np.diff(self.centers_of_mass_y) * self._sampling_frequency * dy_norm

    def _calculate_sizes(self):
        _sizes = []
        for frame in range(len(self.polygon_of_predicted_blobs)):
            mask = get_poly_mask(self.polygon_of_predicted_blobs[frame], 64, 64)
            _sizes.append(mask.sum())
        return _sizes

    def _calculate_amplitudes(self):
        ds = self._load_raw_data()

        amplitudes = []
        for i, frame in enumerate(self.frames_of_appearance):
            single_frame_data = ds.frames.isel(time=frame).values
            mask = get_poly_mask(self.polygon_of_predicted_blobs[i], 64, 64)
            blob_density = single_frame_data[mask.T]
            amplitudes.append(np.max(blob_density))

        return amplitudes

    def _load_raw_data(self):
        return xr.load_dataset(f"{self.file_name}.nc")

    def _extract_sampling_frequency(self):
        ds = self._load_raw_data()
        return 1 / np.diff(ds.time.values)[0]

    def _extract_dx(self):
        ds = self._load_raw_data()
        return np.abs(ds.R.diff("x").values[0, 0])

    def _extract_dy(self):
        ds = self._load_raw_data()
        return np.mean(
            ds.Z.diff("y").values
        )  # mean values since min is 0.09920597 and max is 0.099206686

    def smoothen_all_parameters(self, window_length=5, polyorder=1):
        try:
            self.amplitudes = savgol_filter(self.amplitudes, window_length, polyorder)
            self.sizes = savgol_filter(self.sizes, window_length, polyorder)
            self.velocities_R = savgol_filter(
                self.velocities_R, window_length - 1, polyorder
            )
            self.velocities_Z = savgol_filter(
                self.velocities_Z, window_length - 1, polyorder
            )
        except Exception:
            print("blob lifetime to short for savgol filter")

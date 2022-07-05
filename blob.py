from shapely.geometry import Polygon
from typing import List
import numpy as np
from polygon_to_mask import get_poly_mask
import xarray as xr


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
        self._sampling_frequency = 390804  # Hz
        self.velocities_x = self._calculate_velocity_x()
        self.velocities_y = self._calculate_velocity_y()
        self.sizes = self._calculate_sizes()
        # self.amplitudes = self._calculate_amplitudes()

    def _calculate_velocity_x(self):
        if self.life_time == 0:
            return 0
        return np.diff(self.centers_of_mass_x) * self._sampling_frequency

    def _calculate_velocity_y(self):
        if self.life_time == 0:
            return 0
        return np.diff(self.centers_of_mass_y) * self._sampling_frequency

    def _calculate_sizes(self):
        _sizes = []
        for frame in range(len(self.polygon_of_predicted_blobs)):
            mask = get_poly_mask(self.polygon_of_predicted_blobs[frame], 64, 64)
            _sizes.append(mask.sum())
        return _sizes

    def _calculate_amplitudes():
        raise NotImplementedError

    def _load_raw_data(self):
        ds = xr.load_dataset("short_dataset_coordinates_included.nc")

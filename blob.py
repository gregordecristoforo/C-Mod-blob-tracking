from shapely.geometry import Polygon
from typing import List


class Blob:
    def __init__(
        self,
        blob_id: int,
        VIoU: List[float],
        centers_of_mass_x: List[float],
        centers_of_mass_y: List[float],
        polygon_of_predicted_blobs: List[float],
        polygon_of_brightness_contours: List[float],
    ):

        self.blob_id = blob_id
        self.VIoU = VIoU
        self.centers_of_mass_x = centers_of_mass_x
        self.centers_of_mass_y = centers_of_mass_y
        self.polygon_of_predicted_blobs = polygon_of_predicted_blobs
        self.polygon_of_brightness_contours = polygon_of_brightness_contours
        self.life_time = len(self.VIoU)

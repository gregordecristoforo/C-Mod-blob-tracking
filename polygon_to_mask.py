import numpy as np
import shapely
from shapely.geometry import Polygon
from matplotlib.path import Path
import pickle
import matplotlib.pyplot as plt


def get_poly_mask(poly, n_x, n_y):
    # e.g. n_x, n_y = 256, 256
    x_frame = np.linspace(0.0, 1.0, n_x)
    y_frame = np.linspace(0.0, 1.0, n_y)
    y_grid, x_grid = np.meshgrid(y_frame, x_frame)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    mask = np.zeros((n_x, n_y))
    if poly.type == "MultiPolygon":
        for poly_i in list(poly):
            x, y = poly_i.exterior.coords.xy
            mask[
                Path([(x[j], y[j]) for j in range(len(x))])
                .contains_points(points)
                .reshape(n_x, n_y)
            ] = 1.0
    else:
        x, y = poly.exterior.coords.xy
        mask[
            Path([(x[j], y[j]) for j in range(len(x))])
            .contains_points(points)
            .reshape(n_x, n_y)
        ] = 1.0
    return mask.astype(bool)


data = pickle.load(open("1091216028_1.45_raft.pickle", "rb"))
example_polygon = data["output_tracking"][301][0][4]

tmp = get_poly_mask(example_polygon, 256, 256)

plt.contourf(tmp.T)
plt.show()

from q2_1_eightpoint import normalize_points
from matplotlib import pyplot as plt
import numpy as np
# points is Nx2

# pts = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
# pts = np.array([[1, 1], [1, 4], [4, 1], [4, 4]])
# pts = [[1, 1]]
pts = np.vstack([np.random.uniform(-5, 5, 100), np.random.uniform(-5, 5, 100)]).T


mean_pt = np.mean(pts, axis=0)
max_pt = np.max(pts, axis=0)

pts = pts.T
plt.plot(pts[0, :], pts[1, :], 'ro')
plt.show()


T, new_pts = normalize_points(pts, max_pt, mean_pt)

plt.plot(new_pts[0, :], new_pts[1, :], 'bo')
plt.show()

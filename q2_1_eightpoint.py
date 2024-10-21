import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here


"""
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix
"""

def normalize_points(pts, max, centroid):

    T = np.array([
        [1.0/max,          0, -1*centroid[0]/max],
        [         0, 1.0/max, -1*centroid[1]/max],
        [         0,       0,                  1]
    ])

    homogenized_pts = np.hstack([pts, np.ones([pts.shape[0], 1]) ])

    return T, T @ homogenized_pts.T


def eightpoint(pts1, pts2, M):
    """
    @param pts1: Nx2 values corresponding with the (x,y) = (col, row) of the point
    @param pts2: Nx2 values corresponding with the (x,y)=  (col, row) of the point
    @param M: 1 value that represents the max of the x and y coords
    """
    # pts1 = pts1[:, ::-1]  # Nx2
    # pts2 = pts2[:, ::-1]  # Nx2

    # Normalizing the input pts1 and pts2 using the matrix T.
    # Mean of all x values
    pts1_centroid = np.mean(pts1, axis=0)
    pts2_centroid = np.mean(pts2, axis=0)

    T1, pts1_normalized = normalize_points(pts1, M, pts1_centroid)
    T2, pts2_normalized = normalize_points(pts2, M, pts2_centroid)


    equations = np.zeros([pts1.shape[0], 9])  # Nx9
    #TODO (2) Setup the eight point algorithm's equation.
    for i in range(pts1.shape[0]):
        matched_pt1 = pts1_normalized[:, i]
        matched_pt2 = pts2_normalized[:, i]

        x, y = matched_pt1[0], matched_pt1[1]
        x_dash, y_dash = matched_pt2[0], matched_pt2[1]

        equations[i, :] = np.array([x*x_dash, y*x_dash, x_dash, x*y_dash, y*y_dash, y_dash, x, y, 1])

    #TODO (3) Solve for the least square solution using SVD.
    U, S, V = np.linalg.svd(equations)
    F = V[-1].reshape([3, 3])

    #TODO (4) Use the function `_singularize` (provided) to enforce the singularity condition
    singular_F = _singularize(F)

    #TODO (5) Use the function `refineF` (provided) to refine the computed fundamental matrix.
    #        (Remember to use the normalized points instead of the original points)
    ## Want pts to come in as Nx2
    dehomogenized_normalized_pts1 = pts1_normalized[:2] / pts1_normalized[2]
    dehomogenized_normalized_pts2 = pts2_normalized[:2] / pts2_normalized[2]
    refined_f = refineF(singular_F, dehomogenized_normalized_pts1.T, dehomogenized_normalized_pts2.T)

    #TODO (6) Unscale the fundamental matrix

    refined_f = T2.T @ refined_f @ T1
    # return refined_f
    return refined_f / refined_f[2, 2]



if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    # Q2.1
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    print(f"Error: {np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F))}")

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1

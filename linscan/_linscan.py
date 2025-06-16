import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.linalg import sqrtm, inv, pinv, cholesky, det

from matplotlib import pyplot as plt
from scipy.linalg import eig

inv_sqrt2 = ( 1.0 / np.sqrt(2) )

def Bhattacharyya_bound(p1:np.array, p2:np.array, n_dims:int):
    """Calculate the distance between distributions described in Dennehy, Andrew et al. (2024). LINSCAN -- A Linearity Based Clustering Algorithm. 10.48550/arXiv.2406.17952.

    :param p1: _description_
    :type p1: np.array
    :param p2: _description_
    :type p2: np.array
    :return: _description_
    :rtype: _type_
    """

    assert p1.shape == p2.shape, f"points must have the same shape: [Ndims, 1 + Ndims], but have shapes {p1.shape} and {p2.shape}"

    p1_means, p1_covs = p1[:n_dims], np.reshape(p1[n_dims:], (n_dims, n_dims))
    p2_means, p2_covs = p2[:n_dims], np.reshape(p2[n_dims:], (n_dims, n_dims))

    ## get the square roots of the inverse matrices using scipys Blocked Schur implementation
    sqrt_1 = cholesky(p1_covs)
    sqrt_2 = cholesky(p2_covs)

    ## get the inverse of the covariance matrices
    sigma     = (sqrt_1 + sqrt_2) / 2.0
    inv_sigma = pinv(sigma)

    det_1 = det(sqrt_1)
    det_2 = det(sqrt_2)
    det_sigma = det(sigma)

    ## difference between the means of the distributions
    mean_diff = p1_means - p2_means

    dist = (1.0/8.0) * (mean_diff.T @ inv_sigma @ mean_diff) + 0.5 * np.log(det_sigma / np.sqrt(det_1 * det_2))

    # print("means_1:", p1_means)
    # print("means_2:", p2_means)
    # print("covs_1:")
    # print(p1_covs)
    # print("covs_2:")
    # print(p2_covs)
    # print()
    # print("sqrt_1:")
    # print(sqrt_1)
    # print("sqrt_2:")
    # print(sqrt_2)
    # print()
    # print("sigma:")
    # print(sigma)
    # print()
    # print("inv_sigma:")
    # print(inv_sigma)
    # print()
    # print("det_1:")
    # print(det_1)
    # print("det_2:")
    # print(det_2)
    # print("det_sigma:")
    # print(det_sigma)
    # print()
    # print("distance:", dist)
    # print("#######################################")
    # print()
    # input("continue")

    return dist

print("Bhattacharyya_bound same dist: ", Bhattacharyya_bound(
    np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1
))

def Dennehy_distance(p1:np.array, p2:np.array, n_dims:int):
    """Calculate the distance between distributions described in Dennehy, Andrew et al. (2024). LINSCAN -- A Linearity Based Clustering Algorithm. 10.48550/arXiv.2406.17952.

    :param p1: _description_
    :type p1: np.array
    :param p2: _description_
    :type p2: np.array
    :return: _description_
    :rtype: _type_
    """

    assert p1.shape == p2.shape, f"points must have the same shape: [Ndims, 1 + Ndims], but have shapes {p1.shape} and {p2.shape}"

    p1_means, p1_covs = p1[:n_dims], np.reshape(p1[n_dims:], (n_dims, n_dims))
    p2_means, p2_covs = p2[:n_dims], np.reshape(p2[n_dims:], (n_dims, n_dims))

    ## get the inverse of the covariance matrices
    inv_1 = pinv(p1_covs / np.linalg.norm(p1_covs))
    inv_2 = pinv(p2_covs / np.linalg.norm(p2_covs))

    ## get the square roots of the inverse matrices using scipys Blocked Schur implementation
    sqrt_inv_1 = cholesky(inv_1)
    sqrt_inv_2 = cholesky(inv_2)
    
    ## difference between the means of the distributions
    mean_diff = p1_means - p2_means

    dist = ( 
        0.5 * np.linalg.norm(sqrt_inv_1 @ p2_covs @ sqrt_inv_1 - np.eye(n_dims) ,ord="fro") +
        0.5 * np.linalg.norm(sqrt_inv_2 @ p1_covs @ sqrt_inv_2 - np.eye(n_dims) ,ord="fro") +
        inv_sqrt2 * np.sqrt( mean_diff.T @ inv_1 @ mean_diff ) +
        inv_sqrt2 * np.sqrt( mean_diff.T @ inv_2 @ mean_diff ) 
    )

    return dist

    
class LINSCANOriginal(BaseEstimator, ClusterMixin):
    """Perform LINSCAN on a vector array of points

    Implementation is as described in [Dennehy, Andrew et al. (2024). LINSCAN -- A Linearity Based Clustering Algorithm. 10.48550/arXiv.2406.17952.](https://www.researchgate.net/publication/381736748_LINSCAN_--_A_Linearity_Based_Clustering_Algorithm).
    """

    def __init__(self, ecc_pts:int, n_dims:int, eps:float=0.5, min_samples:int=5, algorithm:str="auto", **dbscan_args):

        self.ecc_pts_ = ecc_pts
        self.n_dims_ = n_dims

        ## the algorithm used to find nearest neighbours
        self.neighbour_algo_ = NearestNeighbors(n_neighbors=ecc_pts, algorithm=algorithm)
        
        ## the dbscan algorithm to use for clustering distributions
        self.dbscan_ = DBSCAN(eps, min_samples=min_samples, metric=lambda x,y: Dennehy_distance(x,y,n_dims), **dbscan_args)

    def fit(self, points):

        distances, indices = self.neighbour_algo_.fit(points).kneighbors(points)

        ## arrays to hold the mean and covariance matrix for each point neigbourhood
        p_space_points = np.zeros((points.shape[0], self.n_dims_ + self.n_dims_**2))

        for point_id in range(points.shape[0]):
            neighbourhood_points = points[indices[point_id]]

            p_space_points[point_id, :self.n_dims_] = np.mean(neighbourhood_points, axis = 0)

            p_space_points[point_id, self.n_dims_:] = np.ravel(np.cov(neighbourhood_points.T))
            
        self.labels_ = self.dbscan_.fit_predict(p_space_points)

    def fit_predict(self, X, y = None):

        self.fit(X, y)

        return self.labels_
    
class LINSCAN(BaseEstimator, ClusterMixin):
    """Perform LINSCAN on a vector array of points

    """

    iteration_ = 0

    def __init__(self, ecc_pts:int, n_dims:int, eps:float=0.5, dbscan_eps:float=0.2, min_samples:int=5, algorithm:str="auto", **dbscan_args):

        self.ecc_pts_ = ecc_pts
        self.n_dims_ = n_dims

        ## the algorithm used to find nearest neighbours
        self.neighbour_algo_ = NearestNeighbors(n_neighbors=ecc_pts, radius=eps, algorithm=algorithm)
        
        ## the dbscan algorithm to use for clustering distributions
        self.dbscan_ = DBSCAN(dbscan_eps, min_samples=min_samples, metric=lambda x,y: Dennehy_distance(x,y,n_dims), **dbscan_args)

    def fit(self, points):

        distances, indices = self.neighbour_algo_.fit(points).radius_neighbors(points)

        # fig = plt.figure()

        ## arrays to hold the mean and covariance matrix for each point neigbourhood
        p_space_points = np.zeros((points.shape[0], self.n_dims_ + self.n_dims_**2))

        for point_id in range(points.shape[0]):
            neighbourhood_points = points[indices[point_id]]

            if neighbourhood_points.shape[0] <= 3:
                p_space_points[point_id, :self.n_dims_] = neighbourhood_points[0]
                p_space_points[point_id, self.n_dims_:] = np.ravel(np.eye(self.n_dims_) * 1000) # <- point has no neighbours, interpret this as stupidly high variance

            else:
                p_space_points[point_id, :self.n_dims_] = np.mean(neighbourhood_points, axis = 0)

                p_space_points[point_id, self.n_dims_:] = np.ravel(np.cov(neighbourhood_points.T))

            eigvals, eigvecs = eig(np.reshape(p_space_points[point_id, self.n_dims_:], (self.n_dims_, self.n_dims_)))
            max_eigvec = eigvecs[:, np.argmax(eigvals)]

            # plt.plot(
            #     [p_space_points[point_id, 0], p_space_points[point_id, 0] + max_eigvec[0] / 10.0],
            #     [p_space_points[point_id, 1], p_space_points[point_id, 1] + max_eigvec[1] / 10.0]
            # )
            
        self.labels_ = self.dbscan_.fit_predict(p_space_points)

        # plt.savefig(f"LINSCAN_DEBUG{LINSCAN.iteration_}.png")
        # LINSCAN.iteration_ += 1

    def fit_predict(self, X, y = None):

        self.fit(X, y)

        return self.labels_
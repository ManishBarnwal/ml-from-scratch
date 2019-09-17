import logging
import numpy as np

logging.basicConfig(format='%(message)s', level=logging.DEBUG)  # setting level to debug; will log all types of logging
LOG = logging.getLogger(__name__)


class KMeansScratch:
    def __init__(self,
                 n_clusters,
                 max_iterations,
                 tolerance,
                 scale_data=True, seed=2019):

        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.scale_data = scale_data
        self._seed = seed
        self._tolerance = tolerance

        self.X = None
        self._initial_centroids = None
        self._cluster_dict_final = None
        self._cluster_mapping_point_final = None
        self.labels_ = None
        self.centroids = None

    @staticmethod
    def _get_numeric_df(df):
        return df.select_dtypes(exclude=['object', 'category'])

    @staticmethod
    def _scale_data(input_data_numeric):
        X = input_data_numeric.values
        col_min = np.min(X, axis=0)
        col_max = np.max(X, axis=0)
        X_scaled = (X - col_min) / (col_max - col_min)  # normalise data between [0, 1]

        return X_scaled

    def _initialize_centroids(self, X):
        np.random.seed(self._seed)
        rand_idx = np.random.randint(len(X), size=self.n_clusters)
        self._initial_centroids = X[rand_idx, :]

    @staticmethod
    def _calculate_distance(p1, p2):
        return np.sqrt(np.sum(np.power(p1 - p2, 2)))

    def _assign_to_nearest_centroid(self, X, centroids):
        # initialize empty dictionary to store points allocated to each cluster
        cluster_dict = {}
        cluster_mapping_point = {}
        for i in range(self.n_clusters):
            cluster_dict[i] = []
            cluster_mapping_point[i] = []

        for i in range(len(X)):
            dist = [self._calculate_distance(X[i, :], c) for c in centroids]
            clust_ind = np.argmin(dist)
            cluster_mapping_point[clust_ind].append(i)  # keep track of ind of points
            cluster_dict[clust_ind].append(X[i, :])

        return cluster_dict, cluster_mapping_point

    def _get_new_centroids(self, cluster_dict, n_features):

        new_centroids = np.zeros(shape=(1, n_features))  # initialising an empty array for concatenation

        for key in cluster_dict.keys():
            centroid_points = np.mean(cluster_dict[key], axis=0, keepdims=True)
            new_centroids = np.concatenate((new_centroids, centroid_points))

        # TODO: a better way to do this
        return new_centroids[1:(self.n_clusters + 1), :]  # exclude the first row - zeros

    def fit(self, X):
        LOG.info('Fitting KMeansScratch model')

        LOG.info('Selecting only numeric columns for clustering')
        X_numeric_df = self._get_numeric_df(X)

        if self.scale_data:
            LOG.info('Scaling data as scale_data set to {}'.format(self.scale_data))
            X_scaled = self._scale_data(X_numeric_df)
        else:
            LOG.info('Not scaling data as scale_data set to {}'.format(self.scale_data))
            X_scaled = X_numeric_df.values

        self._initialize_centroids(X_scaled)
        n_features = X_scaled.shape[1]

        cluster_dict, cluster_mapping_point = self._assign_to_nearest_centroid(X_scaled, self._initial_centroids)

        new_centroids = self._get_new_centroids(cluster_dict, n_features=n_features)
        centroid_distance = self._calculate_distance(self._initial_centroids, new_centroids)

        for i in range(1, self.max_iterations):
            LOG.info('Iteration {} of {} iterations'.format(i, self.max_iterations))
            if centroid_distance > self._tolerance:
                prev_centroids = new_centroids
                cluster_dict, cluster_mapping_point = self._assign_to_nearest_centroid(X_scaled, new_centroids)
                new_centroids = self._get_new_centroids(cluster_dict, n_features=n_features)
                centroid_distance = self._calculate_distance(prev_centroids, new_centroids)
            else:
                LOG.info('Model converged earlier at iteration {} before maximum iterations'.format(i))
                break

        self._cluster_dict_final = cluster_dict
        self._cluster_mapping_point_final = cluster_mapping_point
        self.centroids = new_centroids

        # get the cluster labels
        self.labels_ = self.predict()

    def predict(self):
        row_cluster_list = []

        for key in self._cluster_mapping_point_final:
            values = self._cluster_mapping_point_final.get(key)
            row_ind_cluster = [(v, key) for v in values]

            row_cluster_list.extend(row_ind_cluster)

        row_cluster_list.sort(key=lambda x: x[0])
        cluster_list = [c[1] for c in row_cluster_list]

        return cluster_list

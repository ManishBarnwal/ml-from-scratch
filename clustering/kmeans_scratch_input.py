from numpy import unique
from sklearn.datasets import load_iris
import pandas as pd

import time

from clustering.kmeans_scratch import KMeansScratch

iris = load_iris()
# df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# Change target to target_names & merge with main dataframe
# df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
# print(df.head())

# read larger dataset
df_input = pd.read_csv('latent_factors.csv')

# print(df_input.head())

start_time = time.time()
kmeans_scratch = KMeansScratch(n_clusters=4, max_iterations=15,
                               scale_data=True,
                               seed=10, tolerance=0.1)


kmeans_scratch.fit(df_input)

end_time = time.time()
print('Time taken to fit KmeansScratch: {} seconds \n'.format(end_time - start_time))


print('--- Doing some sanity checks ---')

cluster_labels = kmeans_scratch.labels_
print('length of cluster_labels: {} \n'.format(len(cluster_labels)))

# print(cluster_labels)
print('The unique clusters are: {} \n'.format(unique(cluster_labels)))

for c in unique(cluster_labels):
    print('No. of points in each cluster:')
    print(cluster_labels.count(c))

print('centroids of clusters: {}'.format(kmeans_scratch.centroids))
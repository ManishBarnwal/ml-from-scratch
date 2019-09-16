from numpy import unique
from sklearn.datasets import load_iris
import pandas as pd

from clustering.kmeans_scratch import KMeansScratch

iris = load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# Change target to target_names & merge with main dataframe
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df.head())


kmeans_scratch = KMeansScratch(n_clusters=3, max_iterations=50, scale_data=True, seed=1200)
print('Fitting KMeansScratch model')

kmeans_scratch.fit(df)

cluster_labels = kmeans_scratch.labels_
print(len(cluster_labels))

print(cluster_labels)
print(unique(cluster_labels))

for c in unique(cluster_labels):
    print(cluster_labels.count(c))

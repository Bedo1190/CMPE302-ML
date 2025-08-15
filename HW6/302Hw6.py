import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Question 1 – K-Means First Iteration
dataset = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1,0.6], [9,11]])

def first_kmeans_round(dataset, k=2):
    initial = dataset[random.sample(range(len(dataset)), k)]
    print("Initial chosen centroids:", initial)

    grouping = {i: [] for i in range(k)}
    for point in dataset:
        dists = [np.linalg.norm(point - c) for c in initial]
        idx = np.argmin(dists)
        grouping[idx].append(point)

    new_centroids = []
    for grp in grouping.values():
        new_centroids.append(np.mean(grp, axis=0))
    
    print("New centroids after first round:", np.array(new_centroids))

first_kmeans_round(dataset)

# Question 2 – Elbow Plot
x_points = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y_points = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
samples = list(zip(x_points, y_points))

distortions = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=1)
    model.fit(samples)
    distortions.append(model.inertia_)

plt.plot(range(1, 11), distortions, marker='x')
plt.title('Elbow Method Graph')
plt.xlabel('K Value')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# a) Meaning: Inertia_ in K-Means indicates how far points are from their respective centroids on average.
# A lower inertia implies that the data points are more closely grouped within their clusters.

# b) Elbow Method: This technique estimates the most suitable cluster count by plotting inertia versus K.
# The point where the curve bends (like an elbow) highlights a good trade-off between cluster count and compactness.
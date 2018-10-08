import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


data, target = make_blobs(n_samples=6666, n_features=2, centers=10,random_state = 66, cluster_std = 1.0)

kmeans =KMeans(n_clusters=10, init="random", max_iter=500)
kmeans.fit(data)
centers = kmeans.cluster_centers_
print(kmeans.inertia_)
plt.scatter(data[:, 0], data[:,1], c = target)
# plt.scatter(centers[:,0], centers[:,1], c = "b", marker="<")


# kmeans1 = KMeans(n_clusters=10, init=centers, max_iter=1)
# kmeans1.fit(data)
# # kmeans1.cluster_centers_ = centers
# # centers1 = kmeans1.cluster_centers_
# kmeans1.labels_ = kmeans.predict(data)
# new_SSEDM = 0
# for i in range(len(centers)):
#     print(np.sum(kmeans1.transform(data[kmeans1.labels_ == i]), axis = 0))
#     new_SSEDM += np.sum(kmeans1.transform(data[kmeans1.labels_ == i]), axis = 0)[i]
# centers1 = kmeans1.cluster_centers_
# print(new_SSEDM)
# print(kmeans1.inertia_)

# plt.scatter(centers1[:,0], centers1[:,1], c = "r", marker=">")
plt.show()
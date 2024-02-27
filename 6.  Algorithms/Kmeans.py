import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Generating random 1000 data
#Data has 1000 rows and 2 columns, and the values are randomly generated between 0 and 1, scaled by 100.
data = np.random.rand(1000, 2) * 100

#Fitting the data into the KMeans Algorithm with 3 clusters
km = KMeans(n_clusters=3, init="random")
km.fit(data)

#Printing the first cluster center
centers = km.cluster_centers_
labels = km.labels_
print("Cluser centers: ", *centers)

#Plotting the result
colors = ["r", "g", "b"]
markers = ["+", "x", "*"]
for i in range(len(data)):
  plt.plot(data[i][0], data[i][1], color=colors[labels[i]],
marker=markers[labels[i]])
plt.scatter(centers[:, 0], centers[:, 1], marker="s", s=100,
linewidths=5)
plt.show()
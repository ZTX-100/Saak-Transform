import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from time import time

data_1 = np.random.rand(100000, 4)
data_2 = np.random.rand(100000, 20)

time_1 = time()
bandwidth_1 = estimate_bandwidth(X=data_1, quantile=0.5)
meanshift_1 = MeanShift(bandwidth=bandwidth_1, bin_seeding=True).fit(data_1)
K_1 = meanshift_1.labels_
zip_data_1 = zip(data_1, K_1)
data = []
for i in range(len(np.unique(K_1))):
    data.append([dat for dat, k in zip_data_1 if k == i])
print(time() - time_1)

time_2 = time()
bandwidth_2 = estimate_bandwidth(X=data_2, quantile=0.5)
meanshift_2 = MeanShift(bandwidth=bandwidth_2, bin_seeding=True).fit(data_2)
K_2 = meanshift_2.labels
zip_data_2 = zip(data_2, K_2)
data = []
for i in range(len(np.unique(K_2))):
    data.append([dat for dat, k in zip_data_2 if k == i])
print(time() - time_2)

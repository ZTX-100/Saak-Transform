import numpy as np
# import cPickle
# import gzip
# from time import time
# from sklearn import svm
# from sklearn.feature_selection import f_classif
# from sklearn.feature_selection import SelectPercentile
# from sklearn.decomposition import PCA
# from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from sklearn.cluster import MiniBatchKMeans


# test = np.load("./coefficients/test" + str(7) + ".npy")

test_11 = np.load("./coefficients/train" + str(7) + str(7) + ".npy")
test_12 = np.load("./coefficients/train" + str(7) + str(9) + ".npy")
print("test_11.shape: {}".format(test_11.shape))
print("test_12.shape: {}".format(test_12.shape))
mean_11 = np.mean(abs(test_11), axis=0)
mean_12 = np.mean(abs(test_12), axis=0)
plt.figure(1)
plt.plot(mean_11)
plt.xlabel("568 index for the last stage coefficients")
plt.ylabel("mean value for each coefficient")
plt.title("7 images for 7 as the kernel")
plt.figure(2)
plt.plot(mean_12)
plt.xlabel("568 index for the last stage coefficients")
plt.ylabel("mean value for each coefficient")
plt.title("9 images for 7 as the kernel")
# plt.show()

# score_1 = []
# bins = np.arange(np.amin(test_1), np.amax(test_1), 0.1)
# for i in range(test_11.shape[1]):
#     hist_11, edges_11 = np.histogram(test_11[:, i], bins=bins, density=True)
#     hist_12, edges_12 = np.histogram(test_12[:, i], bins=bins, density=True)
#     hist_11[hist_11 < 0.0001] = 0.0001
#     hist_12[hist_12 < 0.0001] = 0.0001
#     score_1.append(entropy(hist_11, hist_12))
# score_1 = np.array(score_1)

test_21 = np.load("./coefficients/train" + str(9) + str(7) + ".npy")
test_22 = np.load("./coefficients/train" + str(9) + str(9) + ".npy")
print("test_21.shape: {}".format(test_21.shape))
print("test_22.shape: {}".format(test_22.shape))
mean_21 = np.mean(abs(test_21), axis=0)
mean_22 = np.mean(abs(test_22), axis=0)
plt.figure(3)
plt.plot(mean_21)
plt.xlabel("568 index for the last stage coefficients")
plt.ylabel("mean value for each coefficient")
plt.title("7 images for 9 as the kernel")
plt.figure(4)
plt.plot(mean_22)
plt.xlabel("568 index for the last stage coefficients")
plt.ylabel("mean value for each coefficient")
plt.title("9 images for 9 as the kernel")
plt.show()

# score_2 = []
# bins = np.arange(np.amin(test_2), np.amax(test_2), 0.1)
# for i in range(test_11.shape[1]):
#     hist_21, edges_21 = np.histogram(test_21[:, i], bins=bins, density=True)
#     hist_22, edges_22 = np.histogram(test_22[:, i], bins=bins, density=True)
#     hist_21[hist_21 < 0.0001] = 0.0001
#     hist_22[hist_22 < 0.0001] = 0.0001
#     score_2.append(entropy(hist_21, hist_22))
# score_2 = np.array(score_2)


# max_index_1 = np.argsort(-score_1)[9]
# max_index_2 = np.argsort(-score_2)[0]

# bins = np.arange(np.amin([np.amin(test_11[:, max_index_1]), np.amin(test_12[:, max_index_1])]), np.amin([np.amax(test_11[:, max_index_1]), np.amax(test_12[:, max_index_1])]), 0.1)
# plt.hist(test_11[:, max_index_1], bins=bins, histtype='step', color='r', label='7')
# plt.hist(test_12[:, max_index_1], bins=bins, histtype='step', color='b', label='9')
# plt.hist(test[:, max_index_1], bins=bins, color='k', label='test')
# plt.title("Histogram of 10th maximum K-L divergence value for 7 as kernel \nwith all train images for defferent classes")
# plt.xlabel("Index for coefficients: {}".format(max_index_1))
# plt.legend()
# plt.show()

# stage_1 = range(0, 256)
# stage_2 = range(1024, 1024 + 64)
# stage_3 = range(1344, 1344 + 16)
# stage_4 = range(1472, 1472 + 4)
# stage_5 = range(1500, 1500 + 1)

# test_11 = np.load("./coefficients/train" + str(7) + str(7) + ".npy")
# test_12 = np.load("./coefficients/train" + str(7) + str(9) + ".npy")
# test_11_no_dc = np.load("./coefficients_no_dc/train" + str(7) + str(7) + ".npy")
# test_12_no_dc = np.load("./coefficients_no_dc/train" + str(7) + str(9) + ".npy")
# euclidean_11 = []
# euclidean_12 = []
# for i in range(test_11.shape[0]):
#     euclidean_11.append(euclidean(test_11[i, stage_5], test_11_no_dc[i, stage_5]))
# euclidean_11 = np.array(euclidean_11)
# for j in range(test_12.shape[0]):
#     euclidean_12.append(euclidean(test_12[j, stage_5], test_12_no_dc[j, stage_5]))
# euclidean_12 = np.array(euclidean_12)

# plt.plot(euclidean_11, label='7', color='r')
# plt.plot(euclidean_12, label='9', color='k')
# plt.title("The distance of DC and the first component without DC \nfor the 5th stage with all train images (kernel 7)")
# plt.legend()
# plt.show()


# test_21 = np.load("./coefficients/train" + str(9) + str(7) + ".npy")
# test_22 = np.load("./coefficients/train" + str(9) + str(9) + ".npy")
# test_21_no_dc = np.load("./coefficients_no_dc/train" + str(9) + str(7) + ".npy")
# test_22_no_dc = np.load("./coefficients_no_dc/train" + str(9) + str(9) + ".npy")
# euclidean_21 = []
# euclidean_22 = []
# for i in range(test_21.shape[0]):
#     euclidean_21.append(euclidean(test_21[i, stage_5], test_21_no_dc[i, stage_5]))
# euclidean_21 = np.array(euclidean_21)
# for j in range(test_22.shape[0]):
#     euclidean_22.append(euclidean(test_22[j, stage_5], test_22_no_dc[j, stage_5]))
# euclidean_22 = np.array(euclidean_22)

# plt.plot(euclidean_21, label='7', color='r')
# plt.plot(euclidean_22, label='9', color='k')
# plt.title("The distance of DC and the first component without DC \nfor the 5th stage with all train images (kernel 9)")
# plt.legend()
# plt.show()

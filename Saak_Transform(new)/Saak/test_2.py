
import numpy as np
import cPickle
import gzip
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from time import time
from sklearn import svm
from sklearn.feature_selection import f_classif
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import entropy


def Unsign(train_data):
    filternum = (train_data.shape[3] - 1) / 2
    ta1 = np.concatenate((train_data[:, :, :, :1], train_data[:, :, :, 1:filternum + 1] - train_data[:, :, :, filternum + 1:]), axis=3)
    return ta1.reshape(ta1.shape[0], -1)


def evac_ftest(rep2, label):
    F, p = f_classif(rep2, label)
    low_conf = p > 0.05
    F[low_conf] = 0
    where_are_NaNs = np.isnan(F)
    F[where_are_NaNs] = 0
    return F


def window_process(train, test):

    train_shape = train.shape
    test_shape = test.shape

    train_cnt = train_shape[0]
    test_cnt = test_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]

    train_window = view_as_windows(train, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(train_cnt * w / 2 * h / 2, -1)
    test_window = view_as_windows(test, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(test_cnt * w / 2 * h / 2, -1)
    print("train_window.shape: {}".format(train_window.shape))
    print("test_window.shape: {}".format(test_window.shape))

    return train_window, test_window


def window_process_overlapping(train, test, size):

    train_shape = train.shape
    test_shape = test.shape

    train_cnt = train_shape[0]
    test_cnt = test_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]

    train_window = view_as_windows(train, (1, size, size, d), step=(1, 1, 1, d)).reshape(train_cnt * (w - (size - 1)) * (h - (size - 1)), -1)
    test_window = view_as_windows(test, (1, size, size, d), step=(1, 1, 1, d)).reshape(test_cnt * (w - (size - 1)) * (h - (size - 1)), -1)
    print("train_window.shape: {}".format(train_window.shape))
    print("test_window.shape: {}".format(test_window.shape))

    return train_window, test_window


def window_process_8_8(train, test, size):

    train_shape = train.shape
    d = train_shape[3]

    train_window = view_as_windows(train, (1, 8, 8, d), step=(1, size, size, d)).reshape(-1, 8, 8, 1)
    test_window = view_as_windows(test, (1, 8, 8, d), step=(1, size, size, d)).reshape(-1, 8, 8, 1)
    print("train_window.shape: {}".format(train_window.shape))
    print("test_window.shape: {}".format(test_window.shape))

    return train_window, test_window


def convolution(train, test, components):
    # generate sample data and label, change 60000 -> other number (number of images to learn PCA)

    train_shape = train.shape
    test_shape = test.shape
    train_cnt = train_shape[0]
    test_cnt = test_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]
    # use sample to do the DC, AC substraction
    train_window, test_window = window_process(train, test)
    # pca training

    d = train_window.shape[-1]

    train_dc = (np.mean(train_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(train_cnt, w / 2, h / 2, 1)
    test_dc = (np.mean(test_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(test_cnt, w / 2, h / 2, 1)

    mean = np.mean(train_window, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))
    # PCA weight training

    pca = PCA(n_components=components, svd_solver='full')
    pca.fit(train_window - mean)

    train = pca.transform(train_window - mean).reshape(train_cnt, w / 2, h / 2, -1)
    print("train.shape: {}".format(train.shape))
    mean = np.mean(test_window, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))
    test = pca.transform(test_window - mean).reshape(test_cnt, w / 2, h / 2, -1)
    print("test.shape: {}".format(test.shape))

    shape = train.shape
    w, h, d = shape[1], shape[2], shape[3]

    train_data = np.zeros((train_cnt, w, h, 1 + d))
    test_data = np.zeros((test_cnt, w, h, 1 + d))

    train_data[:, :, :, :1] = train_dc[:, :, :, :]
    test_data[:, :, :, :1] = test_dc[:, :, :, :]
    train_data[:, :, :, 1:1 + d] = train[:, :, :, :].copy()
    # train_data[:, :, :, 1 + d:] = -train[:, :, :, :].copy()
    test_data[:, :, :, 1:1 + d] = test[:, :, :, :].copy()
    # test_data[:, :, :, 1 + d:] = -test[:, :, :, :].copy()
    # train_data[train_data < 0] = 0
    # test_data[test_data < 0] = 0

    return train_data, test_data


def convolution_overlapping(train, test, components, size):
    # generate sample data and label, change 60000 -> other number (number of images to learn PCA)

    train_shape = train.shape
    test_shape = test.shape
    train_cnt = train_shape[0]
    test_cnt = test_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]
    # use sample to do the DC, AC substraction
    train_window, test_window = window_process_overlapping(train, test, size)
    # pca training

    d = train_window.shape[-1]

    train_dc = (np.mean(train_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(train_cnt, w - (size - 1), h - (size - 1), 1)
    test_dc = (np.mean(test_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(test_cnt, w - (size - 1), h - (size - 1), 1)

    mean = np.mean(train_window, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))
    # PCA weight training

    pca = PCA(n_components=components, svd_solver='full')
    pca.fit(train_window - mean)

    train = pca.transform(train_window - mean).reshape(train_cnt, w - (size - 1), h - (size - 1), -1)
    print("train.shape: {}".format(train.shape))
    mean = np.mean(test_window, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))
    test = pca.transform(test_window - mean).reshape(test_cnt, w - (size - 1), h - (size - 1), -1)
    print("test.shape: {}".format(test.shape))

    shape = train.shape
    w, h, d = shape[1], shape[2], shape[3]

    train_data = np.zeros((train_cnt, w, h, 1 + d))
    test_data = np.zeros((test_cnt, w, h, 1 + d))

    train_data[:, :, :, :1] = train_dc[:, :, :, :]
    test_data[:, :, :, :1] = test_dc[:, :, :, :]
    train_data[:, :, :, 1:1 + d] = train[:, :, :, :].copy()
    # train_data[:, :, :, 1 + d:] = -train[:, :, :, :].copy()
    test_data[:, :, :, 1:1 + d] = test[:, :, :, :].copy()
    # test_data[:, :, :, 1 + d:] = -test[:, :, :, :].copy()
    # train_data[train_data < 0] = 0
    # test_data[test_data < 0] = 0

    return train_data, test_data


def main():

    start_time = time()
    f = gzip.open('./mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    train = np.concatenate((train_set[0], valid_set[0]), 0)
    train_label = np.concatenate((train_set[1], valid_set[1]))
    train_labels = np.concatenate((train_set[1], valid_set[1]))
    test = test_set[0]
    test_label = test_set[1]
    test_labels = test_set[1]

    train_cnt, test_cnt = train.shape[0], test.shape[0]
    train = train.reshape((train_cnt, 28, 28, 1))
    train = np.lib.pad(train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    test = test.reshape((test_cnt, 28, 28, 1))
    test = np.lib.pad(test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))

    retest = np.load("./retest.npy")
    # fig, ax = plt.subplots(1)
    # ax.imshow(test[retest[1], 0:16, 0:16, 0], cmap='Greys')
    # rect = patches.Rectangle((5, 5), 16, 16, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
    for i in range(retest.shape[0]):
        index = retest[i]
        plt.figure(i)
        plt.subplot(331)
        plt.imshow(test[index, 0:16, 0:16, 0], cmap='Greys')
        plt.subplot(332)
        plt.imshow(test[index, 0:16, 8:24, 0], cmap='Greys')
        plt.subplot(333)
        plt.imshow(test[index, 0:16, 16:32, 0], cmap='Greys')
        plt.subplot(334)
        plt.imshow(test[index, 8:24, 0:16, 0], cmap='Greys')
        plt.subplot(335)
        plt.imshow(test[index, 8:24, 8:24, 0], cmap='Greys')
        plt.subplot(336)
        plt.imshow(test[index, 8:24, 16:32, 0], cmap='Greys')
        plt.subplot(337)
        plt.imshow(test[index, 16:32, 0:16, 0], cmap='Greys')
        plt.subplot(338)
        plt.imshow(test[index, 16:32, 8:24, 0], cmap='Greys')
        plt.subplot(339)
        plt.imshow(test[index, 16:32, 16:32, 0], cmap='Greys')
        plt.show()


if __name__ == "__main__":
    main()

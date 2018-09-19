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


def window_process_16_16(train, test, size):

    train_shape = train.shape
    d = train_shape[3]

    train_window = view_as_windows(train, (1, 16, 16, d), step=(1, size, size, d)).reshape(-1, 16, 16, 1)
    test_window = view_as_windows(test, (1, 16, 16, d), step=(1, size, size, d)).reshape(-1, 16, 16, 1)
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

    train_data = np.zeros((train_cnt, w, h, 1 + d * 2))
    test_data = np.zeros((test_cnt, w, h, 1 + d * 2))

    train_data[:, :, :, :1] = train_dc[:, :, :, :]
    test_data[:, :, :, :1] = test_dc[:, :, :, :]
    train_data[:, :, :, 1:1 + d] = train[:, :, :, :].copy()
    train_data[:, :, :, 1 + d:] = -train[:, :, :, :].copy()
    test_data[:, :, :, 1:1 + d] = test[:, :, :, :].copy()
    test_data[:, :, :, 1 + d:] = -test[:, :, :, :].copy()
    train_data[train_data < 0] = 0
    test_data[test_data < 0] = 0

    return train_data, test_data


def clustering_1(train_16, train_label_16, train_8, train_label_8, train_4, train_label_4, test_16, test_8, test_4, index):

    train_16 = train_16
    train_8 = train_8
    train_4 = train_4
    test_16 = test_16
    test_8 = test_8
    test_4 = test_4
    train_label_16 = train_label_16
    train_label_8 = train_label_8
    train_label_4 = train_label_4
    n_clusters = 2
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    train_16_label = kmeans.fit_predict(train_16)
    train_8_label = kmeans.predict(train_8)
    train_4_label = kmeans.predict(train_4)
    test_16_label = kmeans.predict(test_16)
    test_8_label = kmeans.predict(test_8)
    test_4_label = kmeans.predict(test_4)
    train_16_cluster = []
    train_8_cluster = []
    train_4_cluster = []
    test_16_cluster = []
    test_8_cluster = []
    test_4_cluster = []
    train_label_16_cluster = []
    train_label_8_cluster = []
    train_label_4_cluster = []
    zip_train_16 = zip(train_16, train_16_label)
    zip_train_8 = zip(train_8, train_8_label)
    zip_train_4 = zip(train_4, train_4_label)
    zip_test_16 = zip(test_16, test_16_label)
    zip_test_8 = zip(test_8, test_8_label)
    zip_test_4 = zip(test_4, test_4_label)
    zip_train_label_16 = zip(train_label_16, train_16_label)
    zip_train_label_8 = zip(train_label_8, train_8_label)
    zip_train_label_4 = zip(train_label_4, train_4_label)
    for i in range(n_clusters):
        train_16_cluster.append(np.array([dat for dat, k in zip_train_16 if k == i]))
        train_8_cluster.append(np.array([dat for dat, k in zip_train_8 if k == i]))
        train_4_cluster.append(np.array([dat for dat, k in zip_train_4 if k == i]))
        test_16_cluster.append(np.array([dat for dat, k in zip_test_16 if k == i]))
        test_8_cluster.append(np.array([dat for dat, k in zip_test_8 if k == i]))
        test_4_cluster.append(np.array([dat for dat, k in zip_test_4 if k == i]))
        train_label_16_cluster.append(np.array([dat for dat, k in zip_train_label_16 if k == i]))
        train_label_8_cluster.append(np.array([dat for dat, k in zip_train_label_8 if k == i]))
        train_label_4_cluster.append(np.array([dat for dat, k in zip_train_label_4 if k == i]))
    print('')
    print("train_16_cluster.shape: [0]: {}, [1]: {}".format(train_16_cluster[0].shape, train_16_cluster[1].shape))
    print("train_8_cluster.shape: [0]: {}, [1]: {}".format(train_8_cluster[0].shape, train_8_cluster[1].shape))
    print("train_4_cluster.shape: [0]: {}, [1]: {}".format(train_4_cluster[0].shape, train_4_cluster[1].shape))
    print("test_16_cluster.shape: [0]: {}, [1]: {}".format(test_16_cluster[0].shape, test_16_cluster[1].shape))
    print("test_8_cluster.shape: [0]: {}, [1]: {}".format(test_8_cluster[0].shape, test_8_cluster[1].shape))
    print("test_4_cluster.shape: [0]: {}, [1]: {}".format(test_4_cluster[0].shape, test_4_cluster[1].shape))
    print("train_label_16_cluster.shape: [0]: {}, [1]: {}".format(train_label_16_cluster[0].shape, train_label_16_cluster[1].shape))
    print("train_label_8_cluster.shape: [0]: {}, [1]: {}".format(train_label_8_cluster[0].shape, train_label_8_cluster[1].shape))
    print("train_label_4_cluster.shape: [0]: {}, [1]: {}".format(train_label_4_cluster[0].shape, train_label_4_cluster[1].shape))
    print('')

    return train_16_cluster[index], train_8_cluster[index], train_4_cluster[index], \
        test_16_cluster[index], test_8_cluster[index], test_4_cluster[index], \
        train_label_16_cluster[index], train_label_8_cluster[index], train_label_4_cluster[index]


def clustering_2(train_8, train_label_8, train_4, train_label_4, test_8, test_4, index):

    train_8 = train_8
    train_4 = train_4
    test_8 = test_8
    test_4 = test_4
    train_label_8 = train_label_8
    train_label_4 = train_label_4
    n_clusters = 2
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    train_8_label = kmeans.fit_predict(train_8)
    train_4_label = kmeans.predict(train_4)
    test_8_label = kmeans.predict(test_8)
    test_4_label = kmeans.predict(test_4)
    train_8_cluster = []
    train_4_cluster = []
    test_8_cluster = []
    test_4_cluster = []
    train_label_8_cluster = []
    train_label_4_cluster = []
    zip_train_8 = zip(train_8, train_8_label)
    zip_train_4 = zip(train_4, train_4_label)
    zip_test_8 = zip(test_8, test_8_label)
    zip_test_4 = zip(test_4, test_4_label)
    zip_train_label_8 = zip(train_label_8, train_8_label)
    zip_train_label_4 = zip(train_label_4, train_4_label)
    for i in range(n_clusters):
        train_8_cluster.append(np.array([dat for dat, k in zip_train_8 if k == i]))
        train_4_cluster.append(np.array([dat for dat, k in zip_train_4 if k == i]))
        test_8_cluster.append(np.array([dat for dat, k in zip_test_8 if k == i]))
        test_4_cluster.append(np.array([dat for dat, k in zip_test_4 if k == i]))
        train_label_8_cluster.append(np.array([dat for dat, k in zip_train_label_8 if k == i]))
        train_label_4_cluster.append(np.array([dat for dat, k in zip_train_label_4 if k == i]))
    print('')
    print("train_8_cluster.shape: [0]: {}, [1]: {}".format(train_8_cluster[0].shape, train_8_cluster[1].shape))
    print("train_4_cluster.shape: [0]: {}, [1]: {}".format(train_4_cluster[0].shape, train_4_cluster[1].shape))
    print("test_8_cluster.shape: [0]: {}, [1]: {}".format(test_8_cluster[0].shape, test_8_cluster[1].shape))
    print("test_4_cluster.shape: [0]: {}, [1]: {}".format(test_4_cluster[0].shape, test_4_cluster[1].shape))
    print("train_label_8_cluster.shape: [0]: {}, [1]: {}".format(train_label_8_cluster[0].shape, train_label_8_cluster[1].shape))
    print("train_label_4_cluster.shape: [0]: {}, [1]: {}".format(train_label_4_cluster[0].shape, train_label_4_cluster[1].shape))
    print('')

    return train_8_cluster[index], train_4_cluster[index], \
        test_8_cluster[index], test_4_cluster[index], \
        train_label_8_cluster[index], train_label_4_cluster[index]


def clustering_3(train_4, train_label_4, test_4, index):

    train_4 = train_4
    test_4 = test_4
    train_label_4 = train_label_4
    n_clusters = 2
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    train_4_label = kmeans.fit_predict(train_4)
    test_4_label = kmeans.predict(test_4)
    train_4_cluster = []
    test_4_cluster = []
    train_label_4_cluster = []
    zip_train_4 = zip(train_4, train_4_label)
    zip_test_4 = zip(test_4, test_4_label)
    zip_train_label_4 = zip(train_label_4, train_4_label)
    for i in range(n_clusters):
        train_4_cluster.append(np.array([dat for dat, k in zip_train_4 if k == i]))
        test_4_cluster.append(np.array([dat for dat, k in zip_test_4 if k == i]))
        train_label_4_cluster.append(np.array([dat for dat, k in zip_train_label_4 if k == i]))
    print('')
    print("train_4_cluster.shape: [0]: {}, [1]: {}".format(train_4_cluster[0].shape, train_4_cluster[1].shape))
    print("test_4_cluster.shape: [0]: {}, [1]: {}".format(test_4_cluster[0].shape, test_4_cluster[1].shape))
    print("train_label_4_cluster.shape: [0]: {}, [1]: {}".format(train_label_4_cluster[0].shape, train_label_4_cluster[1].shape))
    print('')

    return train_4_cluster[index], test_4_cluster[index], train_label_4_cluster[index]


def binary_tree_1(n):

    tree = []
    if n & 2 == 0:
        tree.append(0)
    else:
        tree.append(1)

    if n & 1 == 0:
        tree.append(0)
    else:
        tree.append(1)

    return tree


def binary_tree_2(n):

    tree = []
    if n & 4 == 0:
        tree.append(0)
    else:
        tree.append(1)

    if n & 2 == 0:
        tree.append(0)
    else:
        tree.append(1)

    if n & 1 == 0:
        tree.append(0)
    else:
        tree.append(1)

    return tree


def binary_tree_3(n):

    tree = []
    if n & 32 == 0:
        tree.append(0)
    else:
        tree.append(1)

    if n & 16 == 0:
        tree.append(0)
    else:
        tree.append(1)

    if n & 8 == 0:
        tree.append(0)
    else:
        tree.append(1)

    if n & 4 == 0:
        tree.append(0)
    else:
        tree.append(1)

    if n & 2 == 0:
        tree.append(0)
    else:
        tree.append(1)

    if n & 1 == 0:
        tree.append(0)
    else:
        tree.append(1)

    return tree


def main():

    start_time = time()
    f = gzip.open('./mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    train = np.concatenate((train_set[0], valid_set[0]), 0)
    train_label = np.concatenate((train_set[1], valid_set[1]))
    test = test_set[0]
    test_label = test_set[1]

    train_cnt, test_cnt = train.shape[0], test.shape[0]
    train = train.reshape((train_cnt, 28, 28, 1))
    train = np.lib.pad(train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    test = test.reshape((test_cnt, 28, 28, 1))
    test = np.lib.pad(test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)

    train_16_root, test_16_root = window_process_16_16(train, test, 16)
    print("train_16_root.shape: {}".format(train_16_root.shape))
    print("test_16_root.shape: {}".format(test_16_root.shape))
    train_16_root = train_16_root.reshape(-1, 16 * 16)
    test_16_root = test_16_root.reshape(-1, 16 * 16)
    print("train_16_root.shape: {}".format(train_16_root.shape))
    print("test_16_root.shape: {}".format(test_16_root.shape))
    train_label_16_root = np.repeat(train_label, 4)
    print("train_label_16_root.shape: {}".format(train_label_16_root.shape))

    train_8_root, test_8_root = window_process_16_16(train, test, 8)
    print("train_8_root.shape: {}".format(train_8_root.shape))
    print("test_8_root.shape: {}".format(test_8_root.shape))
    train_8_root = train_8_root.reshape(-1, 16 * 16)
    test_8_root = test_8_root.reshape(-1, 16 * 16)
    print("train_8_root.shape: {}".format(train_8_root.shape))
    print("test_8_root.shape: {}".format(test_8_root.shape))
    train_label_8_root = np.repeat(train_label, 9)
    print("train_label_8_root.shape: {}".format(train_label_8_root.shape))

    train_4_root, test_4_root = window_process_16_16(train, test, 4)
    print("train_4_root.shape: {}".format(train_4_root.shape))
    print("test_4_root.shape: {}".format(test_4_root.shape))
    train_4_root = train_4_root.reshape(-1, 16 * 16)
    test_4_root = test_4_root.reshape(-1, 16 * 16)
    print("train_4_root.shape: {}".format(train_4_root.shape))
    print("test_4_root.shape: {}".format(test_4_root.shape))
    train_label_4_root = np.repeat(train_label, 25)
    print("train_label_4_root.shape: {}".format(train_label_4_root.shape))

    entropy_10 = []
    count = []
    probability_all = [0] * 10
    probability_20 = [0] * 10
    probability_15 = [0] * 10
    probability_10 = [0] * 10
    for a in range(4):
        train_16_1 = train_16_root
        train_8_1 = train_8_root
        train_4_1 = train_4_root
        test_16_1 = test_16_root
        test_8_1 = test_8_root
        test_4_1 = test_4_root
        train_label_16_1 = train_label_16_root
        train_label_8_1 = train_label_8_root
        train_label_4_1 = train_label_4_root
        tree_a = binary_tree_1(a)
        for i in range(2):
            train_16_1, train_8_1, train_4_1, test_16_1, test_8_1, test_4_1, train_label_16_1, train_label_8_1, train_label_4_1 = clustering_1(train_16_1, train_label_16_1, train_8_1, train_label_8_1, train_4_1, train_label_4_1, test_16_1, test_8_1, test_4_1, tree_a[i])
        for b in range(8):
            train_8_2 = train_8_1
            train_4_2 = train_4_1
            test_8_2 = test_8_1
            test_4_2 = test_4_1
            train_label_8_2 = train_label_8_1
            train_label_4_2 = train_label_4_1
            tree_b = binary_tree_2(b)
            for j in range(3):
                train_8_2, train_4_2, test_8_2, test_4_2, train_label_8_2, train_label_4_2 = clustering_2(train_8_2, train_label_8_2, train_4_2, train_label_4_2, test_8_2, test_4_2, tree_b[j])
            for c in range(64):
                train_4_3 = train_4_2
                test_4_3 = test_4_2
                train_label_4_3 = train_label_4_2
                tree_c = binary_tree_3(c)
                for k in range(6):
                    train_4_3, test_4_3, train_label_4_3 = clustering_3(train_4_3, train_label_4_3, test_4_3, tree_c[k])
                    if train_4_3.shape[0] <= 3000:
                        break
                train_10 = []
                for p in range(10):
                    train_10.append(float(list(train_label_4_3).count(p)) / float(train_label_4_3.shape[0]))
                entropy_10.append(entropy(train_10))
                count.append(train_4_3.shape[0])
                if train_4_3.shape[0] <= 3000:
                    for q in range(10):
                        probability_all[q] += train_10[q]
                if entropy(train_10) <= 2.0 and train_4_3.shape[0] <= 3000:
                    for q in range(10):
                        probability_20[q] += train_10[q]
                if entropy(train_10) <= 1.5 and train_4_3.shape[0] <= 3000:
                    for q in range(10):
                        probability_15[q] += train_10[q]
                if entropy(train_10) <= 1.0 and train_4_3.shape[0] <= 3000:
                    for q in range(10):
                        probability_10[q] += train_10[q]

    plt.figure(1)
    plt.plot(entropy_10)
    plt.title("Entropy of nodes")
    plt.xlabel("index of nodes")

    plt.figure(2)
    plt.plot(count)
    plt.title("Number of blocks")
    plt.xlabel("index of nodes")

    plt.figure(3)
    plt.plot(probability_all)
    plt.title("Probability sums for nodes without entropy restriction")
    plt.xlabel("index of 10 classes")

    plt.figure(4)
    plt.plot(probability_20)
    plt.title("Probability sums for salient nodes with entropy less than 2.0")
    plt.xlabel("index of 10 classes")

    plt.figure(5)
    plt.plot(probability_15)
    plt.title("Probability sums for salient nodes with entropy less than 1.5")
    plt.xlabel("index of 10 classes")

    plt.figure(6)
    plt.plot(probability_10)
    plt.title("Probability sums for salient nodes with entropy less than 1.0")
    plt.xlabel("index of 10 classes")
    plt.show()

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}
    print ('The total time for classification: %(minute)d minute(s) %(second)d second(s)' % time_total)
    print ('')


if __name__ == "__main__":
    main()

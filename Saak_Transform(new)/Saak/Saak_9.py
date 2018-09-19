
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

    print("start average pooling")
    train, test = window_process(train, test)
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))
    train = np.amax(train, axis=1).reshape(train_cnt, 16, 16, 1)
    test = np.amax(test, axis=1).reshape(test_cnt, 16, 16, 1)
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))

    train_8_8, test_8_8 = window_process(train, test)
    print("train_8_8.shape: {}".format(train_8_8.shape))
    print("test_8_8.shape: {}".format(test_8_8.shape))
    train_8_8 = np.amax(train_8_8, axis=1).reshape(train_cnt, -1)
    test_8_8 = np.amax(test_8_8, axis=1).reshape(test_cnt, -1)
    print("train_8_8.shape: {}".format(train_8_8.shape))
    print("test_8_8.shape: {}".format(test_8_8.shape))

    train, test = window_process_8_8(train, test, 4)
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))
    train = train.reshape(train_cnt, 8 * 8, -1)
    test = test.reshape(test_cnt, 8 * 8, -1)
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))

    test_probability = np.zeros([10, test_cnt, 2])
    for x in range(10):
        if x == 9:
            train_block = train_8_8
            test_block = test_8_8
        else:
            train_block = train[:, :, x]
            test_block = test[:, :, x]
        print("train_block.shape: {}".format(train_block.shape))
        print("test_block.shape: {}".format(test_block.shape))

        train_order = np.array(range(train_cnt))
        test_order = np.array(range(test_cnt))
        print("train_order.shape: {}".format(train_order.shape))
        print("test_order.shape: {}".format(test_order.shape))

        train_label = train_labels

        '''
        Stage 1
        '''
        means = np.mean(train_block, axis=0).reshape(1, -1)
        print("means.shape: {}".format(means.shape))
        std = np.std(train_block, axis=0).reshape(1, -1)
        print("std.shape: {}".format(std.shape))
        means_std = np.concatenate((means, std), axis=0)
        print("means_std.shape: {}".format(means_std.shape))

        n_clusters = 2
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, init=means_std, random_state=0, n_init=1)
        train_cluster_label = kmeans.fit_predict(train_block)
        test_cluster_label = kmeans.predict(test_block)

        train_cluster_root = []
        test_cluster_root = []
        train_label_cluster_root = []
        train_order_cluster_root = []
        test_order_cluster_root = []
        zip_train_cluster = zip(train_block, train_cluster_label)
        zip_test_cluster = zip(test_block, test_cluster_label)
        zip_train_label_cluster = zip(train_label, train_cluster_label)
        zip_train_order_cluster = zip(train_order, train_cluster_label)
        zip_test_order_cluster = zip(test_order, test_cluster_label)
        for i in range(n_clusters):
            train_cluster_root.append(np.array([dat for dat, k in zip_train_cluster if k == i]))
            test_cluster_root.append(np.array([dat for dat, k in zip_test_cluster if k == i]))
            train_label_cluster_root.append(np.array([label for label, t in zip_train_label_cluster if t == i]))
            train_order_cluster_root.append(np.array([order for order, m in zip_train_order_cluster if m == i]))
            test_order_cluster_root.append(np.array([order for order, m in zip_test_order_cluster if m == i]))
        print(train_cluster_root[0].shape)
        print(train_cluster_root[1].shape)
        print(test_cluster_root[0].shape)
        print(test_cluster_root[1].shape)
        print(train_label_cluster_root[0].shape)
        print(train_label_cluster_root[1].shape)
        print(train_order_cluster_root[0].shape)
        print(train_order_cluster_root[1].shape)
        print(test_order_cluster_root[0].shape)
        print(test_order_cluster_root[1].shape)

        # train_coefficient = np.empty([train_cnt, 92])
        # test_coefficient = np.empty([test_cnt, 92])
        # entropy_0 = []
        # entropy_1 = []
        for n in range(16):

            tree = []
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
            '''
            Stage 2
            '''
            train_block = train_cluster_root[tree[0]]
            test_block = test_cluster_root[tree[0]]
            train_label = train_label_cluster_root[tree[0]]
            train_order = train_order_cluster_root[tree[0]]
            test_order = test_order_cluster_root[tree[0]]
            means = np.mean(train_block, axis=0).reshape(1, -1)
            print("means.shape: {}".format(means.shape))
            std = np.std(train_block, axis=0).reshape(1, -1)
            print("std.shape: {}".format(std.shape))
            means_std = np.concatenate((means, std), axis=0)
            print("means_std.shape: {}".format(means_std.shape))

            n_clusters = 2
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, init=means_std, random_state=0, n_init=1)
            train_cluster_label = kmeans.fit_predict(train_block)
            test_cluster_label = kmeans.predict(test_block)

            train_cluster = []
            test_cluster = []
            train_label_cluster = []
            train_order_cluster = []
            test_order_cluster = []
            zip_train_cluster = zip(train_block, train_cluster_label)
            zip_test_cluster = zip(test_block, test_cluster_label)
            zip_train_label_cluster = zip(train_label, train_cluster_label)
            zip_train_order_cluster = zip(train_order, train_cluster_label)
            zip_test_order_cluster = zip(test_order, test_cluster_label)
            for i in range(n_clusters):
                train_cluster.append(np.array([dat for dat, k in zip_train_cluster if k == i]))
                test_cluster.append(np.array([dat for dat, k in zip_test_cluster if k == i]))
                train_label_cluster.append(np.array([label for label, t in zip_train_label_cluster if t == i]))
                train_order_cluster.append(np.array([order for order, m in zip_train_order_cluster if m == i]))
                test_order_cluster.append(np.array([order for order, m in zip_test_order_cluster if m == i]))
            print(train_cluster[0].shape)
            print(train_cluster[1].shape)
            print(test_cluster[0].shape)
            print(test_cluster[1].shape)
            print(train_label_cluster[0].shape)
            print(train_label_cluster[1].shape)
            print(train_order_cluster[0].shape)
            print(train_order_cluster[1].shape)
            print(test_order_cluster[0].shape)
            print(test_order_cluster[1].shape)

            '''
            Stage 3
            '''
            train_block = train_cluster[tree[1]]
            test_block = test_cluster[tree[1]]
            train_label = train_label_cluster[tree[1]]
            train_order = train_order_cluster[tree[1]]
            test_order = test_order_cluster[tree[1]]
            means = np.mean(train_block, axis=0).reshape(1, -1)
            print("means.shape: {}".format(means.shape))
            std = np.std(train_block, axis=0).reshape(1, -1)
            print("std.shape: {}".format(std.shape))
            means_std = np.concatenate((means, std), axis=0)
            print("means_std.shape: {}".format(means_std.shape))

            n_clusters = 2
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, init=means_std, random_state=0, n_init=1)
            train_cluster_label = kmeans.fit_predict(train_block)
            test_cluster_label = kmeans.predict(test_block)

            train_cluster = []
            test_cluster = []
            train_label_cluster = []
            train_order_cluster = []
            test_order_cluster = []
            zip_train_cluster = zip(train_block, train_cluster_label)
            zip_test_cluster = zip(test_block, test_cluster_label)
            zip_train_label_cluster = zip(train_label, train_cluster_label)
            zip_train_order_cluster = zip(train_order, train_cluster_label)
            zip_test_order_cluster = zip(test_order, test_cluster_label)
            for i in range(n_clusters):
                train_cluster.append(np.array([dat for dat, k in zip_train_cluster if k == i]))
                test_cluster.append(np.array([dat for dat, k in zip_test_cluster if k == i]))
                train_label_cluster.append(np.array([label for label, t in zip_train_label_cluster if t == i]))
                train_order_cluster.append(np.array([order for order, m in zip_train_order_cluster if m == i]))
                test_order_cluster.append(np.array([order for order, m in zip_test_order_cluster if m == i]))
            print(train_cluster[0].shape)
            print(train_cluster[1].shape)
            print(test_cluster[0].shape)
            print(test_cluster[1].shape)
            print(train_label_cluster[0].shape)
            print(train_label_cluster[1].shape)
            print(train_order_cluster[0].shape)
            print(train_order_cluster[1].shape)
            print(test_order_cluster[0].shape)
            print(test_order_cluster[1].shape)

            '''
            Stage 4
            '''
            train_block = train_cluster[tree[2]]
            test_block = test_cluster[tree[2]]
            train_label = train_label_cluster[tree[2]]
            train_order = train_order_cluster[tree[2]]
            test_order = test_order_cluster[tree[2]]
            means = np.mean(train_block, axis=0).reshape(1, -1)
            print("means.shape: {}".format(means.shape))
            std = np.std(train_block, axis=0).reshape(1, -1)
            print("std.shape: {}".format(std.shape))
            means_std = np.concatenate((means, std), axis=0)
            print("means_std.shape: {}".format(means_std.shape))

            n_clusters = 2
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, init=means_std, random_state=0, n_init=1)
            train_cluster_label = kmeans.fit_predict(train_block)
            test_cluster_label = kmeans.predict(test_block)

            train_cluster = []
            test_cluster = []
            train_label_cluster = []
            train_order_cluster = []
            test_order_cluster = []
            zip_train_cluster = zip(train_block, train_cluster_label)
            zip_test_cluster = zip(test_block, test_cluster_label)
            zip_train_label_cluster = zip(train_label, train_cluster_label)
            zip_train_order_cluster = zip(train_order, train_cluster_label)
            zip_test_order_cluster = zip(test_order, test_cluster_label)
            for i in range(n_clusters):
                train_cluster.append(np.array([dat for dat, k in zip_train_cluster if k == i]))
                test_cluster.append(np.array([dat for dat, k in zip_test_cluster if k == i]))
                train_label_cluster.append(np.array([label for label, t in zip_train_label_cluster if t == i]))
                train_order_cluster.append(np.array([order for order, m in zip_train_order_cluster if m == i]))
                test_order_cluster.append(np.array([order for order, m in zip_test_order_cluster if m == i]))
            print(train_cluster[0].shape)
            print(train_cluster[1].shape)
            print(test_cluster[0].shape)
            print(test_cluster[1].shape)
            print(train_label_cluster[0].shape)
            print(train_label_cluster[1].shape)
            print(train_order_cluster[0].shape)
            print(train_order_cluster[1].shape)
            print(test_order_cluster[0].shape)
            print(test_order_cluster[1].shape)

            '''
            Stage 5
            '''
            train_block = train_cluster[tree[3]]
            test_block = test_cluster[tree[3]]
            train_label = train_label_cluster[tree[3]]
            train_order = train_order_cluster[tree[3]]
            test_order = test_order_cluster[tree[3]]
            means = np.mean(train_block, axis=0).reshape(1, -1)
            print("means.shape: {}".format(means.shape))
            std = np.std(train_block, axis=0).reshape(1, -1)
            print("std.shape: {}".format(std.shape))
            means_std = np.concatenate((means, std), axis=0)
            print("means_std.shape: {}".format(means_std.shape))

            n_clusters = 2
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, init=means_std, random_state=0, n_init=1)
            train_cluster_label = kmeans.fit_predict(train_block)
            test_cluster_label = kmeans.predict(test_block)

            train_cluster = []
            test_cluster = []
            train_label_cluster = []
            train_order_cluster = []
            test_order_cluster = []
            zip_train_cluster = zip(train_block, train_cluster_label)
            zip_test_cluster = zip(test_block, test_cluster_label)
            zip_train_label_cluster = zip(train_label, train_cluster_label)
            zip_train_order_cluster = zip(train_order, train_cluster_label)
            zip_test_order_cluster = zip(test_order, test_cluster_label)
            for i in range(n_clusters):
                train_cluster.append(np.array([dat for dat, k in zip_train_cluster if k == i]))
                test_cluster.append(np.array([dat for dat, k in zip_test_cluster if k == i]))
                train_label_cluster.append(np.array([label for label, t in zip_train_label_cluster if t == i]))
                train_order_cluster.append(np.array([order for order, m in zip_train_order_cluster if m == i]))
                test_order_cluster.append(np.array([order for order, m in zip_test_order_cluster if m == i]))
            print(train_cluster[0].shape)
            print(train_cluster[1].shape)
            print(test_cluster[0].shape)
            print(test_cluster[1].shape)
            print(train_label_cluster[0].shape)
            print(train_label_cluster[1].shape)
            print(train_order_cluster[0].shape)
            print(train_order_cluster[1].shape)
            print(test_order_cluster[0].shape)
            print(test_order_cluster[1].shape)

            train_0 = []
            train_1 = []
            for i in range(10):
                if list(train_label_cluster[0]).count(i) != 0:
                    train_0.append(float(list(train_label_cluster[0]).count(i)) / float(train_label_cluster[0].shape[0]))
                if list(train_label_cluster[1]).count(i) != 0:
                    train_1.append(float(list(train_label_cluster[1]).count(i)) / float(train_label_cluster[1].shape[0]))
            entropy_0 = entropy(train_0)
            entropy_1 = entropy(train_1)
    #         entropy_0.append(entropy(train_0))
    #         entropy_1.append(entropy(train_1))
    #     plt.figure(x)
    #     plt.plot(entropy_0, color='r', label='0')
    #     plt.plot(entropy_1, color='k', label='1')
    #     plt.legend()
    # plt.show()
            if train_order_cluster[0].shape[0] != 0 and test_order_cluster[0].shape[0] != 0:
                if entropy_0 < 100.0:
                    train_saak_0 = train_cluster[0].reshape(-1, 8, 8, 1)
                    test_saak_0 = test_cluster[0].reshape(-1, 8, 8, 1)
                    train_label_0 = train_label_cluster[0]
                    # train_order_0 = train_order_cluster[0]
                    test_order_0 = test_order_cluster[0]
                    if np.unique(train_label_0).shape[0] == 1:
                        test_predict_0 = train_label_0[0]
                        test_probability[x, test_order_0, 0] = test_predict_0
                        test_probability[x, test_order_0, 1] = 1.0
                    else:
                        train_saak_0, test_saak_0 = convolution(train_saak_0, test_saak_0, 3)
                        print("train_saak_0.shape: {}".format(train_saak_0.shape))
                        print("test_saak_0.shape: {}".format(test_saak_0.shape))
                        train_coefficient_0 = Unsign(train_saak_0)
                        test_coefficient_0 = Unsign(test_saak_0)
                        train_saak_0, test_saak_0 = convolution(train_saak_0, test_saak_0, 4)
                        print("train_saak_0.shape: {}".format(train_saak_0.shape))
                        print("test_saak_0.shape: {}".format(test_saak_0.shape))
                        train_coefficient_0 = np.concatenate((train_coefficient_0, Unsign(train_saak_0)), axis=1)
                        test_coefficient_0 = np.concatenate((test_coefficient_0, Unsign(test_saak_0)), axis=1)
                        train_saak_0, test_saak_0 = convolution(train_saak_0, test_saak_0, 7)
                        print("train_saak_0.shape: {}".format(train_saak_0.shape))
                        print("test_saak_0.shape: {}".format(test_saak_0.shape))
                        train_coefficient_0 = np.concatenate((train_coefficient_0, Unsign(train_saak_0)), axis=1)
                        test_coefficient_0 = np.concatenate((test_coefficient_0, Unsign(test_saak_0)), axis=1)
                        classifier_0 = svm.SVC(probability=True)
                        classifier_0.fit(train_coefficient_0, train_label_0)
                        test_proba_0 = classifier_0.predict_proba(test_coefficient_0)
                        # test_proba_0 = np.multiply(test_proba_0, train_0)
                        test_predict_0 = classifier_0.predict(test_coefficient_0)
                        test_probability[x, test_order_0, 0] = test_predict_0
                        test_probability[x, test_order_0, 1] = np.amax(test_proba_0, axis=1)

            if train_order_cluster[1].shape[0] != 0 and test_order_cluster[1].shape[0] != 0:
                if entropy_1 < 100.0:
                    train_saak_1 = train_cluster[1].reshape(-1, 8, 8, 1)
                    test_saak_1 = test_cluster[1].reshape(-1, 8, 8, 1)
                    train_label_1 = train_label_cluster[1]
                    # train_order_1 = train_order_cluster[1]
                    test_order_1 = test_order_cluster[1]
                    if np.unique(train_label_1).shape[0] == 1:
                        test_predict_1 = train_label_1[0]
                        test_probability[x, test_order_1, 0] = test_predict_1
                        test_probability[x, test_order_1, 1] = 1.0
                    else:
                        train_saak_1, test_saak_1 = convolution(train_saak_1, test_saak_1, 3)
                        print("train_saak_1.shape: {}".format(train_saak_1.shape))
                        print("test_saak_1.shape: {}".format(test_saak_1.shape))
                        train_coefficient_1 = Unsign(train_saak_1)
                        test_coefficient_1 = Unsign(test_saak_1)
                        train_saak_1, test_saak_1 = convolution(train_saak_1, test_saak_1, 4)
                        print("train_saak_1.shape: {}".format(train_saak_1.shape))
                        print("test_saak_1.shape: {}".format(test_saak_1.shape))
                        train_coefficient_1 = np.concatenate((train_coefficient_1, Unsign(train_saak_1)), axis=1)
                        test_coefficient_1 = np.concatenate((test_coefficient_1, Unsign(test_saak_1)), axis=1)
                        train_saak_1, test_saak_1 = convolution(train_saak_1, test_saak_1, 7)
                        print("train_saak_1.shape: {}".format(train_saak_1.shape))
                        print("test_saak_1.shape: {}".format(test_saak_1.shape))
                        train_coefficient_1 = np.concatenate((train_coefficient_1, Unsign(train_saak_1)), axis=1)
                        test_coefficient_1 = np.concatenate((test_coefficient_1, Unsign(test_saak_1)), axis=1)
                        classifier_1 = svm.SVC(probability=True)
                        classifier_1.fit(train_coefficient_1, train_label_1)
                        test_proba_1 = classifier_1.predict_proba(test_coefficient_1)
                        # test_proba_1 = np.multiply(test_proba_1, train_1)
                        test_predict_1 = classifier_1.predict(test_coefficient_1)
                        test_probability[x, test_order_1, 0] = test_predict_1
                        test_probability[x, test_order_1, 1] = np.amax(test_proba_1, axis=1)

    print("test_probability.shape: {}".format(test_probability.shape))

    test_predict = np.empty(test_cnt)
    for i in range(test_cnt):
        test_proba_10 = test_probability[:, i, 1]
        test_pred_10 = test_probability[:, i, 0]
        test_predict[i] = test_pred_10[list(test_proba_10).index(np.amax(test_proba_10))]

    accuracy_test = np.sum(test_predict == test_labels)

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}
    print("The accuracty of test images is: {}".format(accuracy_test))
    print ('The total time for classification: %(minute)d minute(s) %(second)d second(s)' % time_total)
    print ('')


if __name__ == "__main__":
    main()

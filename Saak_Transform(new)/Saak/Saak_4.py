
import numpy as np
import cPickle
import gzip
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from time import time
from sklearn import svm
from sklearn.feature_selection import f_classif
from sklearn.cluster import MiniBatchKMeans


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
    test = test_set[0]
    test_label = test_set[1]

    train_cnt, test_cnt = train.shape[0], test.shape[0]
    train = train.reshape((train_cnt, 28, 28, 1))
    train = np.lib.pad(train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    test = test.reshape((test_cnt, 28, 28, 1))
    test = np.lib.pad(test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)

    print('start training')
    train, test = window_process_8_8(train, test, 4)
    train = train.reshape(train_cnt, 8 * 8, -1)
    test = test.reshape(test_cnt, 8 * 8, -1)
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))

    sum_var = np.zeros(train.shape[2])
    for i in range(6000):
        for j in range(train.shape[2]):
            sum_var[j] += np.var(train[i, :, j])
        print(i)
    sum_var_mean = sum_var / 6000
    print(sum_var_mean.shape)
    index = np.argsort(-sum_var_mean)[:int(train.shape[2] * 0.9)]
    print(index.shape)
    train = train[:, :, index]
    test = test[:, :, index]
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))

    train_block = np.empty([train.shape[0] * train.shape[2], train.shape[1]])
    test_block = np.empty([test.shape[0] * test.shape[2], test.shape[1]])

    count = 0
    for i in range(train.shape[0]):
        for j in range(train.shape[2]):
            train_block[count, :] = train[i, :, j]
            count += 1
    count = 0
    for i in range(test.shape[0]):
        for j in range(test.shape[2]):
            test_block[count, :] = test[i, :, j]
            count += 1

    n_clusters = 16
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    train_cluster_label = kmeans.fit_predict(train_block)
    test_cluster_label = kmeans.predict(test_block)

    train_zip = zip(train_block, train_cluster_label)
    test_zip = zip(test_block, test_cluster_label)
    train_cluster = []
    test_cluster = []
    for i in range(n_clusters):
        train_cluster.append(np.array([dat for dat, label in train_zip if label == i]))
        test_cluster.append(np.array([dat for dat, label in test_zip if label == i]))
        print("train_cluster.shape: {}".format(train_cluster[i].shape))
        print("test_cluster.shape: {}".format(test_cluster[i].shape))

    train_coefficients_cluster = []
    test_coefficients_cluster = []
    for i in range(n_clusters):
        train_n = train_cluster[i].reshape(-1, 8, 8, 1)
        test_n = test_cluster[i].reshape(-1, 8, 8, 1)

        train_n, test_n = convolution(train_n, test_n, 3)
        train_n_coefficients = Unsign(train_n)
        test_n_coefficients = Unsign(test_n)

        train_n, test_n = convolution(train_n, test_n, 4)
        train_n_coefficients = np.concatenate((train_n_coefficients, Unsign(train_n)), axis=1)
        test_n_coefficients = np.concatenate((test_n_coefficients, Unsign(test_n)), axis=1)

        train_n, test_n = convolution(train_n, test_n, 7)
        train_n_coefficients = np.concatenate((train_n_coefficients, Unsign(train_n)), axis=1)
        test_n_coefficients = np.concatenate((test_n_coefficients, Unsign(test_n)), axis=1)

        print("train_n_coefficients.shape: {}".format(train_n_coefficients.shape))
        print("test_n_coefficients.shape: {}".format(test_n_coefficients.shape))

        train_coefficients_cluster.append(train_n_coefficients)
        test_coefficients_cluster.append(test_n_coefficients)

    count = 0
    train_count = [0] * n_clusters
    train_coefficients = np.empty([train_block.shape[0], train_coefficients_cluster[0].shape[1]])
    print("train_coefficients.shape: {}".format(train_coefficients.shape))
    for i in train_cluster_label:
        train_coefficients[count, :] = train_coefficients_cluster[i][train_count[i], :]
        count += 1
        train_count[i] += 1

    count = 0
    test_count = [0] * n_clusters
    test_coefficients = np.empty([test_block.shape[0], test_coefficients_cluster[0].shape[1]])
    print("test_coefficients.shape: {}".format(test_coefficients.shape))
    for i in test_cluster_label:
        test_coefficients[count, :] = test_coefficients_cluster[i][test_count[i], :]
        count += 1
        test_count[i] += 1

    train_data = train_coefficients.reshape(train_cnt, -1)
    test_data = test_coefficients.reshape(test_cnt, -1)
    print("train_data.shape: {}".format(train_data.shape))
    print("test_data.shape: {}".format(test_data.shape))

    np.save("./train_data.npy", train_data)
    np.save("./test_data.npy", test_data)

    train_data = np.load("train_data.npy")
    test_data = np.load("test_data.npy")

    """
    @ F-test
    """
    Eva = evac_ftest(train_data, train_label)
    idx = Eva > np.sort(Eva)[::-1][int(np.count_nonzero(Eva) * 0.75) - 1]
    train_coefficients_f_test = train_data[:, idx]
    test_coefficients_f_test = test_data[:, idx]

    """
    @ PCA to 64
    """
    pca = PCA(svd_solver='full')
    pca.fit(train_coefficients_f_test)
    pca_k = pca.components_
    n_components = 128
    W = pca_k[:n_components, :]
    train_coefficients_pca = np.dot(train_coefficients_f_test, np.transpose(W))
    test_coefficients_pca = np.dot(test_coefficients_f_test, np.transpose(W))

    print ('Numpy training saak coefficients shape: {}'.format(train_data.shape))
    print ('Numpy training F-test coefficients shape: {}'.format(train_coefficients_f_test.shape))
    print ('Numpy training PCA coefficients shape: {}'.format(train_coefficients_pca.shape))
    print ('Numpy testing saak coefficients shape: {}'.format(test_data.shape))
    print ('Numpy testing F-test coefficients shape: {}'.format(test_coefficients_f_test.shape))
    print ('Numpy testing PCA coefficients shape: {}'.format(test_coefficients_pca.shape))

    """
    @ SVM classifier
    # """
    classifier = svm.SVC()
    classifier.fit(train_coefficients_pca, train_label)
    accuracy_train = classifier.score(train_coefficients_pca, train_label)
    accuracy_test = classifier.score(test_coefficients_pca, test_label)

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}

    print ("The accuracy of training set is {:.4f}".format(accuracy_train))
    print ("The accuracy of testing set is {:.4f}".format(accuracy_test))
    print ('The total time for classification: %(minute)d minute(s) %(second)d second(s)' % time_total)
    print ('')


if __name__ == "__main__":
    main()


import numpy as np
import cPickle
import gzip
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from time import time
from sklearn import svm
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier


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


def print_shape(train, test):
    print "train's shape is %s" % str(train.shape)
    print "test's shape is %s" % str(test.shape)


def window_process(train, test):
    train_shape = train.shape
    test_shape = test.shape

    train_cnt, test_cnt = train_shape[0], test_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]

    train_window = view_as_windows(train, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(train_cnt * w / 2 * h / 2, -1)
    test_window = view_as_windows(test, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(test_cnt * w / 2 * h / 2, -1)
    print("train_window.shape: {}".format(train_window.shape))
    print("test_window.shape: {}".format(test_window.shape))

    return train_window, test_window


def window_process_overlapping(train, test):
    train_shape = train.shape
    test_shape = test.shape

    train_cnt, test_cnt = train_shape[0], test_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]

    train_window = view_as_windows(train, (1, 2, 2, d), step=(1, 1, 1, d)).reshape(train_cnt * (w - 1) * (h - 1), -1)
    test_window = view_as_windows(test, (1, 2, 2, d), step=(1, 1, 1, d)).reshape(test_cnt * (w - 1) * (h - 1), -1)
    print("train_window.shape: {}".format(train_window.shape))
    print("test_window.shape: {}".format(test_window.shape))

    return train_window, test_window


def convolution(train, test, components):
    # generate sample data and label, change 60000 -> other number (number of images to learn PCA)

    train_shape = train.shape
    test_shape = test.shape
    train_cnt, test_cnt = train_shape[0], test_shape[0]
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
    # print("train.shape: {}".format(train.shape))
    # train = train[:, :f_num].reshape(train_cnt, w - 1, h - 1, -1)
    print("train.shape: {}".format(train.shape))
    mean = np.mean(test_window, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))
    test = pca.transform(test_window - mean).reshape(test_cnt, w / 2, h / 2, -1)
    # print("test.shape: {}".format(test.shape))
    # test = test[:, :f_num].reshape(test_cnt, w - 1, h - 1, -1)
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


def convolution_overlapping(train, test, components):
    # generate sample data and label, change 60000 -> other number (number of images to learn PCA)

    train_shape = train.shape
    test_shape = test.shape
    train_cnt, test_cnt = train_shape[0], test_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]
    # use sample to do the DC, AC substraction
    train_window, test_window = window_process_overlapping(train, test)
    # pca training

    d = train_window.shape[-1]

    # train_dc = (np.mean(train_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(train_cnt, w - 1, h - 1, 1)
    # test_dc = (np.mean(test_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(test_cnt, w - 1, h - 1, 1)

    mean = np.mean(train_window, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))
    # PCA weight training

    pca = PCA(n_components=components, svd_solver='full')
    pca.fit(train_window - mean)

    train = pca.transform(train_window - mean).reshape(train_cnt, w - 1, h - 1, -1)
    print("train.shape: {}".format(train.shape))
    mean = np.mean(test_window, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))
    test = pca.transform(test_window - mean).reshape(test_cnt, w - 1, h - 1, -1)
    print("test.shape: {}".format(test.shape))

    # shape = train.shape
    # w, h, d = shape[1], shape[2], shape[3]

    # train_data = np.zeros((train_cnt, w, h, d))
    # test_data = np.zeros((test_cnt, w, h, d))

    # train_data[:, :, :, :1] = train_dc[:, :, :, :]
    # test_data[:, :, :, :1] = test_dc[:, :, :, :]
    # train_data[:, :, :, :d] = train[:, :, :, :].copy()
    # test_data[:, :, :, :d] = test[:, :, :, :].copy()

    return train, test


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
    print("Saak transform stage one: ")
    train, test = convolution(train, test, 3)
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))

    print("Saak transform stage two: ")
    train_2, test_2 = convolution_overlapping(train, test, 4)
    train, test = convolution(train, test, 4)
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))
    print("train_2.shape: {}".format(train_2.shape))
    print("test_2.shape: {}".format(test_2.shape))

    train_coefficients_2 = np.empty([train_2.shape[0], train_2.shape[1], train_2.shape[2], 2])
    test_coefficients_2 = np.empty([test_2.shape[0], test_2.shape[1], test_2.shape[2], 2])
    print("train_coefficients_2.shape: {}".format(train_coefficients_2.shape))
    print("test_coefficients_2.shape: {}".format(test_coefficients_2.shape))
    for i in range(train_2.shape[0]):
        for j in range(train_2.shape[1]):
            for k in range(train_2.shape[2]):
                temp = np.argsort(-abs(train_2[i, j, k, :]))[0]
                train_coefficients_2[i, j, k, 0] = temp
                train_coefficients_2[i, j, k, 1] = train_2[i, j, k, temp]

    for i in range(test_2.shape[0]):
        for j in range(test_2.shape[1]):
            for k in range(test_2.shape[2]):
                temper = np.argsort(-abs(test_2[i, j, k, :]))[0]
                test_coefficients_2[i, j, k, 0] = temper
                test_coefficients_2[i, j, k, 1] = test_2[i, j, k, temper]

    train_coefficients_2 = train_coefficients_2.reshape(train_coefficients_2.shape[0], -1)
    test_coefficients_2 = test_coefficients_2.reshape(test_coefficients_2.shape[0], -1)
    print("train_coefficients_2.shape: {}".format(train_coefficients_2.shape))
    print("test_coefficients_2.shape: {}".format(test_coefficients_2.shape))

    print("Saak transform stage three: ")
    train_3, test_3 = convolution_overlapping(train, test, 7)
    train, test = convolution(train, test, 7)
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))
    print("train_3.shape: {}".format(train_3.shape))
    print("test_3.shape: {}".format(test_3.shape))

    train_coefficients_3 = np.empty([train_3.shape[0], train_3.shape[1], train_3.shape[2], 2])
    test_coefficients_3 = np.empty([test_3.shape[0], test_3.shape[1], test_3.shape[2], 2])
    print("train_coefficients_3.shape: {}".format(train_coefficients_3.shape))
    print("test_coefficients_3.shape: {}".format(test_coefficients_3.shape))
    for i in range(train_3.shape[0]):
        for j in range(train_3.shape[1]):
            for k in range(train_3.shape[2]):
                temp = np.argsort(-abs(train_3[i, j, k, :]))[0]
                train_coefficients_3[i, j, k, 0] = temp
                train_coefficients_3[i, j, k, 1] = train_3[i, j, k, temp]

    for i in range(test_3.shape[0]):
        for j in range(test_3.shape[1]):
            for k in range(test_3.shape[2]):
                temper = np.argsort(-abs(test_3[i, j, k, :]))[0]
                test_coefficients_3[i, j, k, 0] = temper
                test_coefficients_3[i, j, k, 1] = test_3[i, j, k, temper]

    train_coefficients_3 = train_coefficients_3.reshape(train_coefficients_3.shape[0], -1)
    test_coefficients_3 = test_coefficients_3.reshape(test_coefficients_3.shape[0], -1)
    print("train_coefficients_3.shape: {}".format(train_coefficients_3.shape))
    print("test_coefficients_3.shape: {}".format(test_coefficients_3.shape))

    print("Saak transform stage four: ")
    train, test = convolution_overlapping(train, test, 6)
    print("train.shape: {}".format(train.shape))
    print("test.shape: {}".format(test.shape))

    train_coefficients_4 = np.empty([train.shape[0], train.shape[1], train.shape[2], 2])
    test_coefficients_4 = np.empty([test.shape[0], test.shape[1], test.shape[2], 2])
    print("train_coefficients_4.shape: {}".format(train_coefficients_4.shape))
    print("test_coefficients_4.shape: {}".format(test_coefficients_4.shape))
    for i in range(train.shape[0]):
        for j in range(train.shape[1]):
            for k in range(train.shape[2]):
                temp = np.argsort(-abs(train[i, j, k, :]))[0]
                train_coefficients_4[i, j, k, 0] = temp
                train_coefficients_4[i, j, k, 1] = train[i, j, k, temp]

    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            for k in range(test.shape[2]):
                temper = np.argsort(-abs(test[i, j, k, :]))[0]
                test_coefficients_4[i, j, k, 0] = temper
                test_coefficients_4[i, j, k, 1] = test[i, j, k, temper]

    train_coefficients_4 = train_coefficients_4.reshape(train_coefficients_4.shape[0], -1)
    test_coefficients_4 = test_coefficients_4.reshape(test_coefficients_4.shape[0], -1)
    print("train_coefficients_4.shape: {}".format(train_coefficients_4.shape))
    print("test_coefficients_4.shape: {}".format(test_coefficients_4.shape))

    train_coefficients = np.concatenate((train_coefficients_2, train_coefficients_3), axis=1)
    train_coefficients = np.concatenate((train_coefficients, train_coefficients_4), axis=1)
    test_coefficients = np.concatenate((test_coefficients_2, test_coefficients_3), axis=1)
    test_coefficients = np.concatenate((test_coefficients, test_coefficients_4), axis=1)
    print("train_coefficients.shape: {}".format(train_coefficients.shape))
    print("test_coefficients.shape: {}".format(test_coefficients.shape))

    """
    @ SVM classifier
    """
    print("Start svm training")
    classifier = RandomForestClassifier(n_estimators=500)
    classifier.fit(train_coefficients, train_label)
    accuracy_train = classifier.score(train_coefficients, train_label)
    accuracy_test = classifier.score(test_coefficients, test_label)

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}

    print ("The accuracy of training set is {:.4f}".format(accuracy_train))
    print ("The accuracy of testing set is {:.4f}".format(accuracy_test))
    print ('The total time for classification: %(minute)d minute(s) %(second)d second(s)' % time_total)
    print ('')


if __name__ == "__main__":
    main()

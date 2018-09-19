import numpy as np
import cPickle
import gzip
from time import time
from sklearn import svm
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from scipy.stats import entropy


def Unsign(train_data):
    filternum = (train_data.shape[3] - 1) / 2
    ta1 = np.concatenate((train_data[:, :, :, :1], train_data[:, :, :, 1:filternum + 1] - train_data[:, :, :, filternum + 1:]), axis=3)
    return ta1.reshape(ta1.shape[0], -1)


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

    return train, test


def main():
    start_time = time()
    f = gzip.open('./mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    test_label = test_set[1]

    train_all = np.concatenate((train_set[0], valid_set[0]), 0)
    train_label_all = np.concatenate((train_set[1], valid_set[1]))
    test_all = test_set[0]

    train = train_all  # [train_label_all == class_id]
    test = test_all
    train_label = train_label_all  # [train_label_all == class_id]
    train_cnt, test_cnt = train.shape[0], test.shape[0]
    print train_cnt, test_cnt
    train = train.reshape((train_cnt, 28, 28, 1))
    train = np.lib.pad(train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    test = test.reshape((test_cnt, 28, 28, 1))
    test = np.lib.pad(test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    print('start training')
    zip_data = zip(train, train_label)
    train_classifier = np.load("./train_classifier.npy")
    zip_classifier = zip(train_classifier, train_label)
    dataset = []
    classifier = []
    for i in range(10):
        dataset.append(np.array([dat for dat, k in zip_data if k == i]))
        classifier.append(np.array([clf for clf, t in zip_classifier if t == i]))

    dataset_select = []
    for j in range(10):
        # max_index = np.argsort(-classifier[j][:, j])
        min_index = np.argsort(classifier[j][:, j])
        index = min_index[:2000]
        # index = np.concatenate((max_index[:500], min_index[:500]))
        dataset_select.append(dataset[j][index, :])

    for i in range(10):
        for j in range(10):
            train_n = np.array(dataset_select[i])
            test_n = np.array(dataset[j])

            if j == 0:
                train_n, test_n = convolution(train_n, test_n, 3)
                train_n, test_n = convolution(train_n, test_n, 4)
                train_n, test_n = convolution_overlapping(train_n, test_n, 7)
                temp = np.empty([test_n.shape[0], test_n.shape[1], test_n.shape[2], 2])
                for a in range(test_n.shape[0]):
                    for b in range(test_n.shape[1]):
                        for c in range(test_n.shape[2]):
                            x = np.argsort(-abs(test_n[a, b, c, :]))[0]
                            temp[a, b, c, 0] = x
                            temp[a, b, c, 1] = test_n[a, b, c, x]
                temp = temp.reshape(temp.shape[0], -1)

                train_dataset = np.array(dataset_select[i])
                test_dataset = test[:]
                train_dataset, test_dataset = convolution(train_dataset, test_dataset, 3)
                train_dataset, test_dataset = convolution(train_dataset, test_dataset, 4)
                train_dataset, test_dataset = convolution_overlapping(train_dataset, test_dataset, 7)
                temper = np.empty([test_dataset.shape[0], test_dataset.shape[1], test_dataset.shape[2], 2])
                for a in range(test_dataset.shape[0]):
                    for b in range(test_dataset.shape[1]):
                        for c in range(test_dataset.shape[2]):
                            x = np.argsort(-abs(test_dataset[a, b, c, :]))[0]
                            temper[a, b, c, 0] = x
                            temper[a, b, c, 1] = test_dataset[a, b, c, x]
                temper = temper.reshape(temper.shape[0], -1)
                np.save("./coefficients/test" + str(i) + ".npy", temper)
            else:
                train_n, test_n = convolution(train_n, test_n, 3)
                train_n, test_n = convolution(train_n, test_n, 4)
                train_n, test_n = convolution_overlapping(train_n, test_n, 7)
                temp = np.empty([test_n.shape[0], test_n.shape[1], test_n.shape[2], 2])
                for a in range(test_n.shape[0]):
                    for b in range(test_n.shape[1]):
                        for c in range(test_n.shape[2]):
                            x = np.argsort(-abs(test_n[a, b, c, :]))[0]
                            temp[a, b, c, 0] = x
                            temp[a, b, c, 1] = test_n[a, b, c, x]
                temp = temp.reshape(temp.shape[0], -1)
            np.save("./coefficients/train" + str(i) + str(j) + ".npy", temp)

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}
    print ('The total time for classification: %(minute)d minute(s) %(second)d second(s)' % time_total)


if __name__ == "__main__":
    main()

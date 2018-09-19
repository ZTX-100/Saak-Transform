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


def window_process_2(train, test):
    train_shape = train.shape
    test_shape = test.shape
    print("train_shape: {}".format(train_shape))
    print("test_shape: {}".format(test_shape))

    train_cnt, test_cnt = train_shape[0], test_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]

    train_window = view_as_windows(train, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(train_cnt * w / 2 * h / 2, -1)
    test_window = view_as_windows(test, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(test_cnt * w / 2 * h / 2, -1)
    print("train_window: {}".format(train_window.shape))
    print("test_window: {}".format(test_window.shape))

    return train_window, test_window


def convolution_2(train, test, stage):
    train_shape = train.shape
    test_shape = test.shape
    train_cnt, test_cnt = train_shape[0], test_shape[0]
    print('train count: {}'.format(train_cnt))
    print('test count: {}'.format(test_cnt))
    w, h, d = train_shape[1], train_shape[2], train_shape[3]
    # use sample to do the DC, AC substraction
    train_window, test_window = window_process_2(train, test)
    # train_filter, train_label = patch_filter(train_window, train_label)
    # pca training

    d = train_window.shape[-1]
    train_dc = (np.mean(train_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(train_cnt, w / 2, h / 2, 1)
    test_dc = (np.mean(test_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(test_cnt, w / 2, h / 2, 1)
    print("train_dc.shape: {}".format(train_dc.shape))
    print("test_dc.shape: {}".format(test_dc.shape))

    mean = np.mean(train_window, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))

    # PCA weight training
    # components_PCA = [3, 4, 7, 6, 8]
    components_PCA = [3, 15, 51, 209, 568]
    f_num = components_PCA[stage - 1]
    pca = PCA(n_components=d, svd_solver='full', random_state=0)
    pca.fit(train_window - mean)
    train = pca.transform(train_window - mean)
    print(train.shape)
    train = train[:, :f_num].reshape(train_cnt, w / 2, h / 2, -1)
    print(train.shape)
    mean = np.mean(test_window, axis=1).reshape(-1, 1)
    print(mean.shape)
    test = pca.transform(test_window - mean)
    print(test.shape)
    test = test[:, :f_num].reshape(test_cnt, w / 2, h / 2, -1)
    print(test.shape)

    shape = train.shape
    w, h, d = shape[1], shape[2], shape[3]

    train_data = np.zeros((train_cnt, w, h, 1 + d * 2))
    test_data = np.zeros((test_cnt, w, h, 1 + d * 2))

    train_data[:, :, :, :1] = train_dc[:, :, :, :]
    test_data[:, :, :, :1] = test_dc[:, :, :, :]
    train_data[:, :, :, 1:d + 1] = train[:, :, :, :].copy()
    train_data[:, :, :, d + 1:] = -train[:, :, :, :].copy()
    test_data[:, :, :, 1:d + 1] = test[:, :, :, :].copy()
    test_data[:, :, :, d + 1:] = -test[:, :, :, :].copy()
    train_data[train_data < 0] = 0
    test_data[test_data < 0] = 0

    return train_data, test_data, pca


def convolution_2_pca(train, test, stage, p_c_a):
    train_shape = train.shape
    test_shape = test.shape
    train_cnt, test_cnt = train_shape[0], test_shape[0]
    print('train count: {}'.format(train_cnt))
    print('test count: {}'.format(test_cnt))
    w, h, d = train_shape[1], train_shape[2], train_shape[3]
    # use sample to do the DC, AC substraction
    train_window, test_window = window_process_2(train, test)
    # train_filter, train_label = patch_filter(train_window, train_label)
    # pca training

    d = train_window.shape[-1]
    train_dc = (np.mean(train_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(train_cnt, w / 2, h / 2, 1)
    test_dc = (np.mean(test_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(test_cnt, w / 2, h / 2, 1)
    print("train_dc.shape: {}".format(train_dc.shape))
    print("test_dc.shape: {}".format(test_dc.shape))

    mean = np.mean(train_window, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))

    # PCA weight training
    # components_PCA = [3, 4, 7, 6, 8]
    components_PCA = [3, 15, 51, 209, 568]
    f_num = components_PCA[stage - 1]
    # pca = PCA(n_components=d, svd_solver='full', random_state=0)
    # pca.fit(train_window - mean)
    pca = p_c_a
    train = pca.transform(train_window - mean)
    print(train.shape)
    train = train[:, :f_num].reshape(train_cnt, w / 2, h / 2, -1)
    print(train.shape)
    mean = np.mean(test_window, axis=1).reshape(-1, 1)
    print(mean.shape)
    test = pca.transform(test_window - mean)
    print(test.shape)
    test = test[:, :f_num].reshape(test_cnt, w / 2, h / 2, -1)
    print(test.shape)

    shape = train.shape
    w, h, d = shape[1], shape[2], shape[3]

    train_data = np.zeros((train_cnt, w, h, 1 + d * 2))
    test_data = np.zeros((test_cnt, w, h, 1 + d * 2))

    train_data[:, :, :, :1] = train_dc[:, :, :, :]
    test_data[:, :, :, :1] = test_dc[:, :, :, :]
    train_data[:, :, :, 1:d + 1] = train[:, :, :, :].copy()
    train_data[:, :, :, d + 1:] = -train[:, :, :, :].copy()
    test_data[:, :, :, 1:d + 1] = test[:, :, :, :].copy()
    test_data[:, :, :, d + 1:] = -test[:, :, :, :].copy()
    train_data[train_data < 0] = 0
    test_data[test_data < 0] = 0

    return train_data, test_data


def main():
    start_time = time()
    f = gzip.open('./mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    test_label = test_set[1]

    train_all = np.concatenate((train_set[0], valid_set[0]), 0)
    train_label_all = np.concatenate((train_set[1], valid_set[1]))
    test_all = test_set[0]

    train = train_all
    test = test_all
    train_label = train_label_all
    train_cnt, test_cnt = train.shape[0], test.shape[0]
    print train_cnt, test_cnt
    train = train.reshape((train_cnt, 28, 28, 1))
    train = np.lib.pad(train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    test = test.reshape((test_cnt, 28, 28, 1))
    test = np.lib.pad(test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    print('start training')
    stages = ['first', 'second', 'third', 'fourth', 'fifth']
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
        max_index = np.argsort(-classifier[j][:, j])
        min_index = np.argsort(classifier[j][:, j])
        # index = max_index[:2000]
        index = np.concatenate((max_index[:1000], min_index[:1000]))
        dataset_select.append(dataset[j][index, :])

    for i in range(10):
        pca = []
        for j in range(10):
            train_n = np.array(dataset_select[i])
            test_n = np.array(dataset_select[j])

            if j == 0:
                for k in range(5):
                    stage = k + 1
                    print("The {} stage: ".format(stages[k]))
                    train_n, test_n, p_c_a = convolution_2(train_n, test_n, stage)
                    pca.append(p_c_a)
                    temp = Unsign(test_n)
                    # if k == 0:
                    #     temp = Unsign(test_n)
                    # else:
                    #     temp = np.concatenate((temp, Unsign(test_n)), axis=1)

                train_dataset = np.array(dataset_select[i])
                test_dataset = test[:]
                for k in range(5):
                    stage = k + 1
                    print("The {} stage: ".format(stages[k]))
                    train_dataset, test_dataset = convolution_2_pca(train_dataset, test_dataset, stage, pca[k])
                    temper = Unsign(test_dataset)
                    # if k == 0:
                    #     temper = Unsign(test_dataset)
                    # else:
                    #     temper = np.concatenate((temper, Unsign(test_dataset)), axis=1)
                np.save("./coefficients/test" + str(i) + ".npy", temper)
            else:
                for k in range(5):
                    stage = k + 1
                    print("The {} stage: ".format(stages[k]))
                    train_n, test_n = convolution_2_pca(train_n, test_n, stage, pca[k])
                    temp = Unsign(test_n)
                    # if k == 0:
                    #     temp = Unsign(test_n)
                    # else:
                    #     temp = np.concatenate((temp, Unsign(test_n)), axis=1)
            np.save("./coefficients/train" + str(i) + str(j) + ".npy", temp)

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}
    print ('The total time for classification: %(minute)d minute(s) %(second)d second(s)' % time_total)


if __name__ == "__main__":
    main()


import numpy as np
import cPickle
import gzip
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from time import time
from sklearn import svm
from sklearn.feature_selection import f_classif


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


def window_process_1(train):
    train_shape = train.shape
    # test_shape = test.shape

    train_cnt = train_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]

    train_window = view_as_windows(train, (1, 2, 2, d), step=(1, 1, 1, d)).reshape(train_cnt * (w - 1) * (h - 1), -1)
    # test_window = view_as_windows(test, (1, 2, 2, d), step=(1, 1, 1, d)).reshape(test_cnt * (w - 1) * (h - 1), -1)
    print("train_window.shape: {}".format(train_window.shape))
    # print("test_window.shape: {}".format(test_window.shape))

    return train_window


def window_process_2(train):
    train_shape = train.shape
    # test_shape = test.shape

    train_cnt = train_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]

    train_window = view_as_windows(train, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(train_cnt * w / 2 * h / 2, -1)
    # test_window = view_as_windows(test, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(test_cnt * w / 2 * h / 2, -1)
    print("train_window.shape: {}".format(train_window.shape))
    # print("test_window.shape: {}".format(test_window.shape))

    return train_window


def convolution(train, stage):
    # generate sample data and label, change 60000 -> other number (number of images to learn PCA)

    train_shape = train.shape
    # test_shape = test.shape
    train_cnt = train_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]
    # use sample to do the DC, AC substraction
    train_window_1 = window_process_1(train)
    train_window_2 = window_process_2(train)
    # pca training

    d = train_window_2.shape[-1]

    train_dc_1 = (np.mean(train_window_1, axis=1) * (d**0.5)).reshape(-1, 1).reshape(train_cnt, w - 1, h - 1, 1)
    # test_dc_1 = (np.mean(test_window_1, axis=1) * (d**0.5)).reshape(-1, 1).reshape(test_cnt, w - 1, h - 1, 1)

    train_dc_2 = (np.mean(train_window_2, axis=1) * (d**0.5)).reshape(-1, 1).reshape(train_cnt, w / 2, h / 2, 1)
    # PCA weight training

    component_pca = [3, 4, 7, 6, 8]
    f_num = component_pca[stage - 1]

    mean = np.mean(train_window_1, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))
    pca_1 = PCA(n_components=f_num, svd_solver='full')
    pca_1.fit(train_window_1 - mean)
    train_1 = pca_1.transform(train_window_1 - mean).reshape(train_cnt, w - 1, h - 1, -1)
    print("train_1.shape: {}".format(train_1.shape))
    # mean = np.mean(test_window_1, axis=1).reshape(-1, 1)
    # print("mean.shape: {}".format(mean.shape))
    # test_1 = pca_1.transform(test_window_1 - mean).reshape(test_cnt, w - 1, h - 1, -1)
    # print("test_1.shape: {}".format(test_1.shape))

    mean = np.mean(train_window_2, axis=1).reshape(-1, 1)
    print("mean.shape: {}".format(mean.shape))
    pca_2 = PCA(n_components=f_num, svd_solver='full')
    pca_2.fit(train_window_2 - mean)
    train_2 = pca_2.transform(train_window_2 - mean).reshape(train_cnt, w / 2, h / 2, -1)
    print("train_2.shape: {}".format(train_2.shape))
    # mean = np.mean(test_window_2, axis=1).reshape(-1, 1)
    # print("mean.shape: {}".format(mean.shape))
    # test_2 = pca_2.transform(test_window_2 - mean).reshape(test_cnt, w / 2, h / 2, -1)
    # print("test_2.shape: {}".format(test_2.shape))

    shape = train_2.shape
    w, h, d = shape[1], shape[2], shape[3]

    train_data_2 = np.zeros((train_cnt, w, h, 1 + d * 2))
    # test_data_2 = np.zeros((test_cnt, w, h, 1 + d * 2))

    train_data_2[:, :, :, :1] = train_dc_2[:, :, :, :]
    # test_data_2[:, :, :, :1] = test_dc_2[:, :, :, :]
    train_data_2[:, :, :, 1:1 + d] = train_2[:, :, :, :].copy()
    train_data_2[:, :, :, 1 + d:] = -train_2[:, :, :, :].copy()
    # test_data_2[:, :, :, 1:1 + d] = test_2[:, :, :, :].copy()
    # test_data_2[:, :, :, 1 + d:] = -test_2[:, :, :, :].copy()
    train_data_2[train_data_2 < 0] = 0
    # test_data_2[test_data_2 < 0] = 0

    shape = train_1.shape
    w, h, d = shape[1], shape[2], shape[3]

    train_data_1 = np.zeros((train_cnt, w, h, 1 + d * 2))
    # test_data_1 = np.zeros((test_cnt, w, h, 1 + d * 2))

    train_data_1[:, :, :, :1] = train_dc_1[:, :, :, :]
    # test_data_1[:, :, :, :1] = test_dc_1[:, :, :, :]
    train_data_1[:, :, :, 1:1 + d] = train_1[:, :, :, :].copy()
    train_data_1[:, :, :, 1 + d:] = -train_1[:, :, :, :].copy()
    # test_data_1[:, :, :, 1:1 + d] = test_1[:, :, :, :].copy()
    # test_data_1[:, :, :, 1 + d:] = -test_1[:, :, :, :].copy()
    train_data_1[train_data_1 < 0] = 0
    # test_data_1[test_data_1 < 0] = 0

    return train_data_1, train_data_2


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
    for stage in range(1, 6):
        train_1, train = convolution(train, stage)
        test_1, test = convolution(test, stage)
        print_shape(train, test)
        # save features of each stage (augmented features, when do classfication, you need to converge)
        if stage == 1:
            train_data = Unsign(train_1)
            test_data = Unsign(test_1)
        else:
            train_data = np.concatenate((train_data, Unsign(train_1)), 1)
            test_data = np.concatenate((test_data, Unsign(test_1)), 1)
        print("train_data.shape: {}".format(train_data.shape))
        print("test_data.shape: {}".format(test_data.shape))

    """
    @ F-test
    """
    Eva = evac_ftest(train_data, train_label)
    idx = Eva > np.sort(Eva)[::-1][int(np.count_nonzero(Eva) * 0.25) - 1]
    train_coefficients_f_test = train_data[:, idx]
    test_coefficients_f_test = test_data[:, idx]

    """
    @ PCA to 64
    """
    pca = PCA(svd_solver='full')
    pca.fit(train_coefficients_f_test)
    pca_k = pca.components_
    n_components = 64
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

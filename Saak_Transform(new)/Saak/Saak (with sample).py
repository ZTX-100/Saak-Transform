
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


def generate_sample(imgs, label, img_cnt, class_cnt):
    cnt, w, h, d = imgs.shape
    each_class = img_cnt / class_cnt
    sample_data = np.zeros((img_cnt, w, h, d))
    sample_label = np.zeros(img_cnt)
    taken_img = 0
    dictionary = {}
    for i in range(class_cnt):
        dictionary[i] = each_class
    for index, img in enumerate(imgs):
        if taken_img == img_cnt:
            break
        if (dictionary[label[index]] > 0):
            sample_data[taken_img] = img
            sample_label[taken_img] = label[index]
            taken_img += 1
            dictionary[label[index]] -= 1
    return sample_data, sample_label


def print_shape(train, test):
    print "train's shape is %s" % str(train.shape)
    print "test's shape is %s" % str(test.shape)


def patch_filter(sample, patch_threshold=10000):
    sample_var = np.std(sample, axis=1)
    max_std = np.max(sample_var)
    sample_filter = sample[sample_var > max_std / patch_threshold]
    return sample_filter


def window_process(sample, train, test):
    sample_shape = sample.shape
    train_shape = train.shape
    test_shape = test.shape

    sample_cnt, train_cnt, test_cnt = sample_shape[0], train_shape[0], test_shape[0]
    w, h, d = sample_shape[1], sample_shape[2], sample_shape[3]

    sample_window = view_as_windows(sample, (1, 2, 2, d), step=(1, 1, 1, d)).reshape(sample_cnt * (w - 1) * (h - 1), -1)
    train_window = view_as_windows(train, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(train_cnt * w / 2 * h / 2, -1)
    test_window = view_as_windows(test, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(test_cnt * w / 2 * h / 2, -1)

    return sample_window, train_window, test_window


def convolution(train, test, train_label, stage):
    # generate sample data and label, change 60000 -> other number (number of images to learn PCA)
    sample, sample_label = generate_sample(train, train_label, 60000, 10)

    sample_shape = sample.shape
    train_shape = train.shape
    test_shape = test.shape
    train_cnt, test_cnt = train_shape[0], test_shape[0]
    w, h, d = sample_shape[1], sample_shape[2], sample_shape[3]
    # use sample to do the DC, AC substraction
    sample_window, train_window, test_window = window_process(sample, train, test)
    print('before filtering training sample size: %d' % (sample_window.shape[0]))
    sample_filter = patch_filter(sample_window)
    print('PCA training sample_filter size: %d' % (sample_filter.shape[0]))
    # pca training

    d = sample_window.shape[-1]

    train_dc = (np.mean(train_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(train_cnt, w / 2, h / 2, 1)
    test_dc = (np.mean(test_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(test_cnt, w / 2, h / 2, 1)

    mean = np.mean(sample_filter, axis=1).reshape(-1, 1)

    # PCA weight training

    component_pca = [3, 4, 7, 6, 8]
    f_num = component_pca[stage - 1]

    # pca = PCA(n_components=f_num, svd_solver='full')
    pca = PCA(n_components=f_num)
    pca.fit(sample_filter - mean)

    mean = np.mean(train_window, axis=1).reshape(-1, 1)
    train = pca.transform(train_window - mean).reshape(train_cnt, w / 2, w / 2, -1)
    # train = train[:, :f_num].reshape(train_cnt, (w - 1), (h - 1), -1)
    mean = np.mean(test_window, axis=1).reshape(-1, 1)
    test = pca.transform(test_window - mean).reshape(test_cnt, w / 2, h / 2, -1)
    # test = test[:, :f_num].reshape(test_cnt, (w - 1), (h - 1), -1)

    shape = train.shape
    w, h, d = shape[1], shape[2], shape[3]

    train_data = np.zeros((train_cnt, w, h, 1 + d * 2))
    test_data = np.zeros((test_cnt, w, h, 1 + d * 2))
    # train_data = np.zeros((train_cnt, w, h, 1 + d))
    # test_data = np.zeros((test_cnt, w, h, 1 + d))

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
    for stage in range(1, 6, 1):
        train, test = convolution(train, test, train_label, stage)
        print_shape(train, test)
        # save features of each stage (augmented features, when do classfication, you need to converge)
        if stage == 1:
            train_data = Unsign(train)
            test_data = Unsign(test)
            # train_data = train.reshape(train.shape[0], -1)
            # test_data = test.reshape(test.shape[0], -1)
        else:
            train_data = np.concatenate((train_data, Unsign(train)), 1)
            test_data = np.concatenate((test_data, Unsign(test)), 1)
            # train_data = np.concatenate((train_data, train.reshape(train.shape[0], -1)), 1)
            # test_data = np.concatenate((test_data, test.reshape(test.shape[0], -1)), 1)

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


import numpy as np
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
import cPickle
import gzip
from time import time


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


def patch_filter(sample, sample_label, patch_threshold=10000):
    sample_var = np.std(sample, axis=1)
    max_std = np.max(sample_var)
    print np.max(sample_var)
    sample_filter = sample[sample_var > max_std / patch_threshold]
    sample_label = sample_label[sample_var > max_std / patch_threshold]
    return sample_filter, sample_label


def window_process(sample, train, test, sample_label, train_label, test_label):
    sample_shape = sample.shape
    train_shape = train.shape
    test_shape = test.shape

    sample_cnt, train_cnt, test_cnt = sample_shape[0], train_shape[0], test_shape[0]
    w, h, d = sample_shape[1], sample_shape[2], sample_shape[3]

    sample_label = np.repeat(sample_label, (w - 1) * (h - 1))
    train_label = np.repeat(train_label, w / 2 * h / 2)
    test_label = np.repeat(test_label, w / 2 * h / 2)

    sample_window = view_as_windows(sample, (1, 2, 2, d), step=(1, 1, 1, d)).reshape(sample_cnt * (w - 1) * (h - 1), -1)
    train_window = view_as_windows(train, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(train_cnt * w / 2 * h / 2, -1)
    test_window = view_as_windows(test, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(test_cnt * w / 2 * h / 2, -1)

    return sample_window, train_window, test_window, sample_label, train_label, test_label


def convolution(train, test, train_label, test_label, k, stage, per=None):
    # generate sample data and label, change 60000 -> other number (number of images to learn PCA)
    sample, sample_label = generate_sample(train, train_label, 60000, 10)

    sample_shape = sample.shape
    train_shape = train.shape
    test_shape = test.shape
    sample_cnt, train_cnt, test_cnt = sample_shape[0], train_shape[0], test_shape[0]
    w, h, d = sample_shape[1], sample_shape[2], sample_shape[3]
    # use sample to do the DC, AC substraction
    sample_window, train_window, test_window, sample_label, train_label, test_label = window_process(sample, train, test, sample_label, train_label, test_label)
    print 'before filtering training sample size: %d' % (sample_window.shape[0])
    sample_filter, sample_label = patch_filter(sample_window, sample_label)
    print 'PCA training sample size: %d' % (sample_filter.shape[0])
    # pca training

    d = sample_window.shape[-1]

    train_dc = (np.mean(train_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(train_cnt, w / 2, h / 2, 1)
    test_dc = (np.mean(test_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(test_cnt, w / 2, h / 2, 1)

    mean = np.mean(sample_filter, axis=1).reshape(-1, 1)

    # PCA weight training
    pca = PCA(n_components=d, svd_solver='full')
    pca.fit(sample_filter - mean)

    print pca.explained_variance_ratio_[:50]
    if stage == 1:
        f_num = d - 1
    else:
        Energy = np.cumsum(pca.explained_variance_ratio_)
        idx = pca.explained_variance_ratio_ > 0.03
        f_num = np.count_nonzero(idx)
        print f_num, Energy[f_num]

    mean = np.mean(train_window, axis=1).reshape(-1, 1)
    train = pca.transform(train_window - mean)
    train = train[:, :f_num].reshape(train_cnt, w / 2, h / 2, -1)
    mean = np.mean(test_window, axis=1).reshape(-1, 1)
    test = pca.transform(test_window - mean)
    test = test[:, :f_num].reshape(test_cnt, w / 2, h / 2, -1)

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
    train = np.concatenate((train_set[0], valid_set[0]), 0)
    train_label = np.concatenate((train_set[1], valid_set[1]))
    test = test_set[0]
    test_label = test_set[1]

    train_cnt, test_cnt = train.shape[0], test.shape[0]
    train = train.reshape((train_cnt, 28, 28, 1))
    train = np.lib.pad(train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    test = test.reshape((test_cnt, 28, 28, 1))
    test = np.lib.pad(test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    train_data = np.zeros((train_cnt, 0))
    test_data = np.zeros((test_cnt, 0))

    print('start training')
    stage = 1
    Percentage = [1, 1, 1, 1, 1]  # [3 15 51 209 568]
    for k in [-1, -1, -1, -1, -1]:
        print 'k value is :%d' % k
        train, test = convolution(train, test, train_label, test_label, k, stage, Percentage[stage - 1])
        print_shape(train, test)
        # save features of each stage (augmented features, when do classfication, you need to converge)
        np.save('./feature/train_before_f_test_' + str(stage) + '_v' + '.npy', train)
        np.save('./feature/test_before_f_test_' + str(stage) + '_v' + '.npy', test)
        stage += 1

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}
    print ('The total time for generating Saak coefficients: %(minute)d minute(s) %(second)d second(s)' % time_total)


if __name__ == "__main__":
    main()

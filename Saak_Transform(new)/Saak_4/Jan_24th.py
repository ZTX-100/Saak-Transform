
import numpy as np
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
import cPickle
import gzip
from time import time


def print_shape(train, test):
    print "train's shape is %s" % str(train.shape)
    print "test's shape is %s" % str(test.shape)


def patch_filter(sample, patch_threshold=10000):
    print("sample.shape: {}".format(sample.shape))
    sample_var = np.std(sample, axis=1)
    print("sample_var.shape: {}".format(sample_var.shape))
    max_std = np.max(sample_var)
    print np.max(sample_var)
    sample_filter = sample[sample_var > max_std / patch_threshold]
    return sample_filter


# def window_process(train, test, train_label, test_label):
#     train_shape = train.shape
#     test_shape = test.shape
#     print("train_shape: {}".format(train_shape))
#     print("test_shape: {}".format(test_shape))

#     train_cnt, test_cnt = train_shape[0], test_shape[0]
#     w, h, d = train_shape[1], train_shape[2], train_shape[3]

#     train_label = np.repeat(train_label, w / 2 * h / 2)
#     test_label = np.repeat(test_label, w / 2 * h / 2)
#     print("train_labels.shape: {}".format(train_label.shape))
#     print("test_labels.shape: {}".format(test_label.shape))

#     train_window = view_as_windows(train, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(train_cnt * w / 2 * h / 2, -1)
#     test_window = view_as_windows(test, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(test_cnt * w / 2 * h / 2, -1)
#     print("train_window: {}".format(train_window.shape))
#     print("test_window: {}".format(test_window.shape))

#     return train_window, test_window, train_label, test_label


# def convolution(train, test, train_label, test_label, stage, per=None):
#     train_shape = train.shape
#     test_shape = test.shape
#     train_cnt, test_cnt = train_shape[0], test_shape[0]
#     print('train count: {}'.format(train_cnt))
#     print('test count: {}'.format(test_cnt))
#     w, h, d = train_shape[1], train_shape[2], train_shape[3]
#     # use sample to do the DC, AC substraction
#     train_window, test_window, train_label, test_label = window_process(train, test, train_label, test_label)
#     train_filter, train_label = patch_filter(train_window, train_label)
#     # pca training

#     d = train_window.shape[-1]
#     train_dc = (np.mean(train_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(train_cnt, w / 2, h / 2, 1)
#     test_dc = (np.mean(test_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(test_cnt, w / 2, h / 2, 1)
#     print("train_dc.shape: {}".format(train_dc.shape))
#     print("test_dc.shape: {}".format(test_dc.shape))

#     mean = np.mean(train_window, axis=1).reshape(-1, 1)
#     print("mean.shape: {}".format(mean.shape))

#     # PCA weight training
#     pca = PCA(n_components=d, svd_solver='full')
#     pca.fit(train_window - mean)
#     # print("pca.explained_variance_ratio_: ", pca.explained_variance_ratio_[:50])
#     if stage == 1:
#         f_num = d - 1
#     else:
#         Energy = np.cumsum(pca.explained_variance_ratio_)
#         # f_num = np.count_nonzero(Energy < 0.995)
#         idx = pca.explained_variance_ratio_ > 0.03
#         f_num = np.count_nonzero(idx)
#         print f_num, Energy[f_num]
#     train = pca.transform(train_window - mean)
#     print(train.shape)
#     train = train[:, :f_num].reshape(train_cnt, w / 2, h / 2, -1)
#     print(train.shape)
#     mean = np.mean(test_window, axis=1).reshape(-1, 1)
#     print(mean.shape)
#     test = pca.transform(test_window - mean)
#     print(test.shape)
#     test = test[:, :f_num].reshape(test_cnt, w / 2, h / 2, -1)
#     print(test.shape)

#     shape = train.shape
#     w, h, d = shape[1], shape[2], shape[3]

#     train_data = np.zeros((train_cnt, w, h, 1 + d * 2))
#     test_data = np.zeros((test_cnt, w, h, 1 + d * 2))

#     train_data[:, :, :, :1] = train_dc[:, :, :, :]
#     test_data[:, :, :, :1] = test_dc[:, :, :, :]
#     train_data[:, :, :, 1:d + 1] = train[:, :, :, :].copy()
#     train_data[:, :, :, d + 1:] = -train[:, :, :, :].copy()
#     test_data[:, :, :, 1:d + 1] = test[:, :, :, :].copy()
#     test_data[:, :, :, d + 1:] = -test[:, :, :, :].copy()
#     train_data[train_data < 0] = 0
#     test_data[test_data < 0] = 0

#     return train_data, test_data


def window_process(sample):

    sample_shape = sample.shape
    print("sample_shape: {}".format(sample_shape))
    sample_cnt = sample_shape[0]
    w, h, d = sample_shape[1], sample_shape[2], sample_shape[3]

    sample_window = view_as_windows(sample, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(sample_cnt * w / 2 * h / 2, -1)
    print("sample_window: {}".format(sample_window.shape))

    return sample_window


def convolution(sample, stage):
    sample = np.array(sample)
    sample_shape = sample.shape
    sample_cnt = sample_shape[0]
    print('sample count: {}'.format(sample_cnt))
    w, h, d = sample_shape[1], sample_shape[2], sample_shape[3]
    # use sample to do the DC, AC substraction
    sample_window = window_process(sample)
    # sample_filter = patch_filter(sample_window)
    # pca training

    d = sample_window.shape[-1]
    sample_dc = (np.mean(sample_window, axis=1) * (d**0.5)).reshape(-1, 1).reshape(sample_cnt, w / 2, h / 2, 1)
    print("sample_dc.shape: {}".format(sample_dc.shape))

    mean = np.mean(sample_window, axis=1).reshape(-1, 1)
    # mean = np.mean(sample_window, axis=0)
    print("mean.shape: {}".format(mean.shape))

    # PCA weight training
    components_PCA = [3, 4, 7, 4, 1]
    f_num = components_PCA[stage - 1]
    pca = PCA(n_components=d, svd_solver='full')
    pca.fit(sample_window - mean)
    # print("pca.explained_variance_ratio_: ", pca.explained_variance_ratio_[:50])
    # if stage == 1:
    #     f_num = d - 1
    # else:
    #     Energy = np.cumsum(pca.explained_variance_ratio_)
    #     # f_num = np.count_nonzero(Energy < 0.995)
    #     idx = pca.explained_variance_ratio_ > 0.03
    #     f_num = np.count_nonzero(idx)
    #     print f_num, Energy[f_num]

    sample = pca.transform(sample_window - mean)
    print(sample.shape)
    sample = sample[:, :f_num].reshape(sample_cnt, w / 2, h / 2, -1)
    print(sample.shape)

    shape = sample.shape
    w, h, d = shape[1], shape[2], shape[3]

    sample_data = np.zeros((sample_cnt, w, h, 1 + d * 2))

    sample_data[:, :, :, :1] = sample_dc[:, :, :, :]
    sample_data[:, :, :, 1:d + 1] = sample[:, :, :, :].copy()
    sample_data[:, :, :, d + 1:] = -sample[:, :, :, :].copy()
    sample_data[sample_data < 0] = 0

    return sample_data


def cluster(sample, sample_label, stage):

    zip_data = zip(sample, sample_label)
    dataset = []
    for i in range(10):
        dataset.append([dat for dat, k in zip_data if k == i])

    data_n = []
    # dataset = np.empty([datasets.shape[0], data.shape[1], data.shape[2], data.shape[3]])
    # dataset[0, :, :, :] = data
    for i in range(10):
        data_n.append(convolution(np.array(dataset[i]), stage))
    data_0 = np.array(data_n[0])
    data = np.empty([sample.shape[0], data_0.shape[1], data_0.shape[2], data_0.shape[3]])
    i = [0] * 10
    m = 0
    for k in sample_label:
        data[m, :, :, :] = np.array(data_n[k][i[k]])
        i[k] += 1
        m += 1
    return data


def onebyone(sample, stage):
    sample_0 = convolution(np.expand_dims(sample[0], axis=0), stage)
    dataset = np.empty([sample.shape[0], sample_0.shape[1], sample_0.shape[2], sample_0.shape[3]])
    dataset[0] = sample_0
    for i in range(1, sample.shape[0]):
        dataset[i] = convolution(np.expand_dims(sample[i], axis=0), stage)

    return dataset


def main():

    start_time = time()
    f = gzip.open('./mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_all = np.concatenate((train_set[0], valid_set[0]), 0)
    train_label_all = np.concatenate((train_set[1], valid_set[1]))
    test_all = test_set[0]
    test_label = test_set[1]
    train = train_all  # [train_label_all == class_id]
    test = test_all
    train_label = train_label_all  # [train_label_all == class_id]
    train_cnt, test_cnt = train.shape[0], test.shape[0]
    print train_cnt, test_cnt
    train = train.reshape((train_cnt, 28, 28, 1))
    train = np.lib.pad(train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    train_cluster = train[:, :, :, :]
    test = test.reshape((test_cnt, 28, 28, 1))
    test = np.lib.pad(test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    print('start training')
    stages = ['first', 'second', 'third', 'fourth', 'fifth']
    for k in range(5):
        stage = k + 1
        print("The {} stage: ".format(stages[k]))
        train_cluster = cluster(train_cluster, train_label, stage)
        train = convolution(train, stage)
        test = onebyone(test, stage)
        print_shape(train, test)
        # save features of each stage (augmented features, when do classfication, you need to converge)
        np.save('./feature/train_before_f_test_' + str(stage) + '_v' + '.npy', train)
        np.save('./feature/train_cluster_before_f_test_' + str(stage) + '_v' + '.npy', train_cluster)
        np.save('./feature/test_before_f_test_' + str(stage) + '_v' + '.npy', test)
        print('')

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}
    print ('The total time for generating Saak coefficients: %(minute)d minute(s) %(second)d second(s)' % time_total)


if __name__ == "__main__":
    main()

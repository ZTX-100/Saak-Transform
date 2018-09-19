from sklearn.decomposition import PCA
import numpy as np
import cPickle
import gzip
from sklearn import svm
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from time import time


# (dc,ac,-ac) -> (dc,ac)


def Unsign(train_data):
    filternum = (train_data.shape[3] - 1) / 2
    ta1 = np.concatenate((train_data[:, :, :, :1], train_data[:, :, :, 1:filternum + 1] - train_data[:, :, :, filternum + 1:]), axis=3)
    return ta1.reshape(ta1.shape[0], -1)


def main():

    start_time = time()
    f = gzip.open('./mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    train_label = np.concatenate((train_set[1], valid_set[1]))
    test_label = test_set[1]
    # load features
    train_data = np.load('./feature/train_before_f_test_' + '1_v' + '.npy')
    print (train_data.shape)
    ta = Unsign(train_data)
    print(np.count_nonzero(ta))
    train_data = np.load('./feature/train_before_f_test_' + '2_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_data)), 1)
    print (ta.shape)
    print(np.count_nonzero(ta))
    train_data = np.load('./feature/train_before_f_test_' + '3_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_data)), 1)
    print(np.count_nonzero(ta))
    print (ta.shape)
    train_data = np.load('./feature/train_before_f_test_' + '4_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_data)), 1)
    print(np.count_nonzero(ta))
    print (ta.shape)
    train_data = np.load('./feature/train_before_f_test_' + '5_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_data)), 1)
    print(np.count_nonzero(ta))
    print (ta.shape)
    num_traindata = 60000
    train_data = (ta[:num_traindata])
    print("train_data.shape: {}".format(train_data.shape))

    # load features
    train_cluster = np.load('./feature/train_cluster_before_f_test_' + '1_v' + '.npy')
    print (train_cluster.shape)
    ta = Unsign(train_cluster)
    print(np.count_nonzero(ta))
    train_cluster = np.load('./feature/train_cluster_before_f_test_' + '2_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_cluster)), 1)
    print (ta.shape)
    print(np.count_nonzero(ta))
    train_cluster = np.load('./feature/train_cluster_before_f_test_' + '3_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_cluster)), 1)
    print(np.count_nonzero(ta))
    print (ta.shape)
    train_cluster = np.load('./feature/train_cluster_before_f_test_' + '4_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_cluster)), 1)
    print(np.count_nonzero(ta))
    print (ta.shape)
    train_cluster = np.load('./feature/train_cluster_before_f_test_' + '5_v' + '.npy')
    ta = np.concatenate((ta, Unsign(train_cluster)), 1)
    print(np.count_nonzero(ta))
    print (ta.shape)
    train_cluster = (ta[:num_traindata])
    print("train_cluster.shape: {}".format(train_cluster.shape))

    # idx = train_data < 0
    # train_data[idx] = -train_data[idx]

    train_label = train_label[:num_traindata]
    test_data = np.load('./feature/test_before_f_test_' + '1_v' + '.npy')
    ta = Unsign(test_data)
    print (ta.shape)
    test_data = np.load('./feature/test_before_f_test_' + '2_v' + '.npy')
    ta = np.concatenate((ta, Unsign(test_data)), 1)
    print (ta.shape)
    test_data = np.load('./feature/test_before_f_test_' + '3_v' + '.npy')
    ta = np.concatenate((ta, Unsign(test_data)), 1)
    print (ta.shape)
    test_data = np.load('./feature/test_before_f_test_' + '4_v' + '.npy')
    ta = np.concatenate((ta, Unsign(test_data)), 1)
    print (ta.shape)
    test_data = np.load('./feature/test_before_f_test_' + '5_v' + '.npy')
    ta = np.concatenate((ta, Unsign(test_data)), 1)
    print (ta.shape)
    test_data = (ta)
    print("test_data.shape: {}".format(test_data.shape))
    # idx = test_data < 0
    # test_data[idx] = test_data[idx]
    del ta

    # feature selection
    selector = SelectPercentile(f_classif, percentile=75)
    selector.fit(train_data, train_label)
    train_coefficients_f_test = selector.transform(train_data)
    test_coefficients_f_test = selector.transform(test_data)

    selector_cluster = SelectPercentile(f_classif, percentile=75)
    selector_cluster.fit(train_cluster, train_label)
    train_cluster_coefficients_f_test = selector_cluster.transform(train_cluster)
    # test_coefficients_f_test = selector.transform(test_data)

    """
    @ PCA to 64
    """
    pca = PCA(n_components=64)
    pca.fit(train_coefficients_f_test)
    train_coefficients_pca = pca.transform(train_coefficients_f_test)
    test_coefficients_pca = pca.transform(test_coefficients_f_test)

    pca_cluster = PCA(n_components=64)
    pca_cluster.fit(train_cluster_coefficients_f_test)
    train_cluster_coefficients_pca = pca_cluster.transform(train_cluster_coefficients_f_test)
    # test_coefficients_pca = pca.transform(test_coefficients_f_test)

    zip_data = zip(train_cluster_coefficients_pca, train_label)
    classifier_cluster = []
    for i in range(10):
        dataset = [dat for dat, k in zip_data if k == i]
        data_label = [m for m in train_label if m == i]
        clf = svm.SVC(probability=True)
        clf.fit(dataset, data_label)
        classifier_cluster.append(clf)

    print ('Numpy training saak coefficients shape: {}'.format(train_data.shape))
    print ('Numpy training F-test coefficients shape: {}'.format(train_coefficients_f_test.shape))
    print ('Numpy training PCA coefficients shape: {}'.format(train_coefficients_pca.shape))
    print ('Numpy testing saak coefficients shape: {}'.format(test_data.shape))
    print ('Numpy testing F-test coefficients shape: {}'.format(test_coefficients_f_test.shape))
    print ('Numpy testing PCA coefficients shape: {}'.format(test_coefficients_pca.shape))

    """
    @ SVM classifier
    """
    classifier = svm.SVC(probability=True)
    classifier.fit(train_coefficients_pca, train_label)
    test_probability = classifier.predict_proba(test_coefficients_pca)
    test_probability = list(test_probability)
    accuracy_test = 0
    for i in range(10000):
        max_1 = max(test_probability[i])
        max_index_1 = test_probability[i].index(max_1)
        max_2 = max(test_probability[i].remove(max_1))
        test_probability.insert(max_index_1, max_1)
        max_index_2 = test_probability[i].index(max_2)
        if max_1 / max_2 > 10:
            accuracy_test += max_index_1 == test_label[i]
        else:
            if classifier_cluster[max_index_1].predict_proba(test_coefficients_pca[i]) > classifier_cluster[max_index_2].predict_proba(test_coefficients_pca[i]):
                accuracy_test += max_index_1 == test_label[i]
            else:
                accuracy_test += max_index_2 == test_label[i]

            # accuracy_train = classifier.score(train_coefficients_pca, train_label)
            # accuracy_test = classifier.score(test_coefficients_pca, test_label)

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}

    # print ("The accuracy of training set is {:.4f}".format(accuracy_train))
    print ("The accuracy of testing set is {:.4f}".format(accuracy_test))
    print ('The total time for classification: %(minute)d minute(s) %(second)d second(s)' % time_total)
    print ('')


if __name__ == "__main__":
    main()

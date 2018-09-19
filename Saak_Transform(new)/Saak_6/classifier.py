from sklearn.decomposition import PCA
import numpy as np
import cPickle
import gzip
from sklearn import svm
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from time import time
import matplotlib.pyplot as plt


# (dc,ac,-ac) -> (dc,ac)


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


def pairwise(lst):
    a = []
    length = len(lst)
    for i in range(length - 1):
        a.append([lst[i], lst[i + 1]])

    return a


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

    # # train_data = train_data[:, 1:]
    # # test_data = test_data[:, 1:]
    # print("train_data.shape: {}".format(train_data.shape))
    # print("test_data.shape: {}".format(test_data.shape))

    # zip_data = zip(train_data, train_label)
    # dataset = []

    # max_dataset = []
    # var_dataset = []
    # for i in range(10):
    #     temp = [dat for dat, k in zip_data if k == i]
    #     dataset.append(temp)
    #     # max_dataset.append(np.mean(abs(np.array(temp)), axis=0))
    #     max_dataset.append(np.mean(temp, axis=0))
    #     # var_dataset.append(np.var(np.power(temp, 2), axis=0))
    #     # max_dataset.append(np.amax(temp, axis=0))
    #     # var_dataset.append(np.var(temp, axis=0))

    # max_dataset = np.array(max_dataset)
    # var_dataset = np.array(var_dataset)
    # print("max_dataset.shape: {}".format(max_dataset.shape))
    # print("var_dataset.shape: {}".format(var_dataset.shape))

    # n_dim = train_data.shape[1]
    # score = []
    # label_check = []
    # list_max_1 = []
    # list_max_2 = []
    # print("The dimensional of the data: {}".format(n_dim))
    # for i in range(n_dim):
    #     max_temp = max_dataset[:, i]
    #     max_temp_order = sorted(max_temp, reverse=True)
    #     max_temp_diff = [x - y for x, y in pairwise(max_temp_order)]
    #     max_diff = max(max_temp_diff)
    #     max_temp_diff.remove(max_diff)
    #     mean_diff = np.mean(max_temp_diff)
    #     score.append(max_diff - mean_diff)

    # score = np.array(score)
    # # list_max_1 = np.array(list_max_1)
    # # list_max_2 = np.array(list_max_2)
    # print("score.shape: {}".format(score.shape))
    # index_new = np.argsort(-score)[:800]
    # # list_max_1 = list(list_max_1[index_sorted_score])
    # # list_max_2 = list(list_max_2[index_sorted_score])

    # # for i in range(10):
    # #     print("max_1: label: {}, number: {}".format(i, list_max_1.count(i)))
    # #     print("max_2: label: {}, number: {}".format(i, list_max_2.count(i)))

    # # plt.plot(range(64), score[index_sorted_score], 'o')
    # # plt.title("The highest 64 scores")
    # # plt.show()
    # # selector = SelectPercentile(f_classif)
    # # selector.fit(train_data, train_label)
    # # score_ftest = selector.scores_
    # # print(score_ftest.shape)
    # # index_ftest = list(np.argsort(-score_ftest)[:400])
    # # for index in index_new:
    # #     if index not in index_ftest:
    # #         index_ftest.append(index)

    # # index_ftest = np.array(index_ftest)
    # # print(index_ftest.shape)

    # train_svm = train_data[:, index_new]
    # test_svm = test_data[:, index_new]
    # print("train_svm.shape: {}".format(train_svm.shape))
    # print("test_svm.shape: {}".format(test_svm.shape))

    # pca = PCA(n_components=64)
    # pca.fit(train_svm)
    # train_svm = pca.transform(train_svm)
    # test_svm = pca.transform(test_svm)
    # print("train_svm.shape: {}".format(train_svm.shape))
    # print("test_svm.shape: {}".format(test_svm.shape))

    # classifier = svm.SVC()
    # classifier.fit(train_svm, train_label)
    # accuracy_train = classifier.score(train_svm, train_label)
    # accuracy_test = classifier.score(test_svm, test_label)
    # # test_classifier = classifier.predict_proba(test_svm)
    # # np.save("./test_classifier.npy", test_classifier)

    # end_time = time()
    # minutes, seconds = divmod(end_time - start_time, 60)
    # time_total = {'minute': minutes, 'second': seconds}

    # print ("The accuracy of training set is {:.4f}".format(accuracy_train))
    # print ("The accuracy of testing set is {:.4f}".format(accuracy_test))
    # print ('The total time for classification: %(minute)d minute(s) %(second)d second(s)' % time_total)
    # print ('')

    # selector = SelectPercentile(f_classif, percentile=75)
    # selector.fit(train_data, train_label)
    # scores = np.array(selector.scores_)
    # pvalues = np.array(selector.pvalues_)
    # scores[pvalues > 0.05] = 0
    # scores[np.isnan(scores)] = 0
    # index = np.argsort(-scores)
    # train_coefficients_f_test = train_data[:, index[:int(np.count_nonzero(scores) * 0.75) - 1]]
    # test_coefficients_f_test = test_data[:, index[:int(np.count_nonzero(scores) * 0.75) - 1]]

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
    # train_classifier = classifier.predict_proba(train_coefficients_pca)
    # test_classifier = classifier.predict_proba(test_coefficients_pca)
    # np.save("./train_classifier.npy", train_classifier)
    # np.save("./test_classifier.npy", test_classifier)

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}

    print ("The accuracy of training set is {:.4f}".format(accuracy_train))
    print ("The accuracy of testing set is {:.4f}".format(accuracy_test))
    print ('The total time for classification: %(minute)d minute(s) %(second)d second(s)' % time_total)
    print ('')


if __name__ == "__main__":
    main()

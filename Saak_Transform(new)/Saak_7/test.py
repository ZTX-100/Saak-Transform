import numpy as np
import cPickle
import gzip
from time import time
from sklearn import svm
from sklearn.decomposition import PCA
from scipy.stats import entropy
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


def main():
    start_time = time()
    f = gzip.open('./mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    test_label = test_set[1]

    test_probability = np.load('./test_classifier_1' + '.npy')
    print("test_probability.shape: {}".format(test_probability.shape))
    test_probability = list(test_probability)
    accuracy_test = 0
    clf = []
    confusion_labels = []
    pca = []
    index = []
    count = 0
    correct = 0
    wrong = 0
    right = 0
    for i in range(10000):
        print(i)
        test_list = test_probability[i]
        max_1 = -(np.sort(-test_list)[0])
        max_2 = -(np.sort(-test_list)[1])
        max_index_1 = np.argsort(-test_list)[0]
        max_index_2 = np.argsort(-test_list)[1]

        if max_1 / max_2 > 2:
            accuracy_test += max_index_1 == test_label[i]

        else:
            right += max_index_1 == test_label[i]
            count += 1
            element = [min(max_index_1, max_index_2), max(max_index_1, max_index_2)]
            test_1 = np.load("./coefficients/test" + str(max_index_1) + ".npy")
            test_2 = np.load("./coefficients/test" + str(max_index_2) + ".npy")
            test_1 = test_1.reshape(test_1.shape[0], -1)
            test_2 = test_2.reshape(test_2.shape[0], -1)
            if element not in confusion_labels:
                score_1 = []
                score_2 = []
                data_11 = np.load("./coefficients/train" + str(max_index_1) + str(max_index_1) + ".npy")
                data_12 = np.load("./coefficients/train" + str(max_index_1) + str(max_index_2) + ".npy")
                data_11 = data_11.reshape(data_11.shape[0], -1)
                data_12 = data_12.reshape(data_12.shape[0], -1)
                data_1 = np.vstack((data_11, data_12))
                bins = np.arange(np.amin(data_1), np.amax(data_1), 0.1)
                for m in range(data_11.shape[1]):
                    hist_11, edges_11 = np.histogram(data_11[:, m], bins=bins, density=True)
                    hist_12, edges_12 = np.histogram(data_12[:, m], bins=bins, density=True)
                    score_1.append(abs(entropy(hist_11) - entropy(hist_12)))
                score_1 = np.array(score_1)
                count_1 = sum(score_1 > 0)
                print("count_1: {}".format(count_1))
                index_1 = np.argsort(-score_1)[:count_1]
                data_1 = data_1[:, index_1]
                pca_1 = PCA(n_components=64)
                data_1 = pca_1.fit_transform(data_1)

                data_21 = np.load("./coefficients/train" + str(max_index_2) + str(max_index_1) + ".npy")
                data_22 = np.load("./coefficients/train" + str(max_index_2) + str(max_index_2) + ".npy")
                data_21 = data_21.reshape(data_21.shape[0], -1)
                data_22 = data_22.reshape(data_22.shape[0], -1)
                data_2 = np.vstack((data_21, data_22))
                bins = np.arange(np.amin(data_2), np.amax(data_2), 0.1)
                for m in range(data_21.shape[1]):
                    hist_21, edges_21 = np.histogram(data_21[:, m], bins=bins, density=True)
                    hist_22, edges_22 = np.histogram(data_22[:, m], bins=bins, density=True)
                    score_2.append(abs(entropy(hist_21) - entropy(hist_22)))
                score_2 = np.array(score_2)
                count_2 = sum(score_2 > 0)
                print("count_2: {}".format(count_2))
                index_2 = np.argsort(-score_2)[:count_2]
                data_2 = data_2[:, index_2]
                pca_2 = PCA(n_components=64)
                data_2 = pca_2.fit_transform(data_2)

                classifier_1 = svm.SVC(probability=True)
                classifier_2 = svm.SVC(probability=True)
                classifier_1.fit(data_1, np.concatenate((max_index_1 * np.ones(data_11.shape[0]), max_index_2 * np.ones(data_12.shape[0]))))
                classifier_2.fit(data_2, np.concatenate((max_index_1 * np.ones(data_21.shape[0]), max_index_2 * np.ones(data_22.shape[0]))))
                confusion_labels.append(element)
                clf.append([classifier_1, classifier_2])
                pca.append([pca_1, pca_2])
                index.append([index_1, index_2])
            else:
                classifier_1 = clf[confusion_labels.index(element)][0]
                classifier_2 = clf[confusion_labels.index(element)][1]
                pca_1 = pca[confusion_labels.index(element)][0]
                pca_2 = pca[confusion_labels.index(element)][1]
                index_1 = index[confusion_labels.index(element)][0]
                index_2 = index[confusion_labels.index(element)][1]

            test_pca_1 = test_1[i, index_1].reshape(1, -1)
            test_pca_2 = test_2[i, index_2].reshape(1, -1)
            test_pca_1 = pca_1.transform(test_pca_1)
            test_pca_2 = pca_2.transform(test_pca_2)
            # test_list_1 = np.squeeze(classifier_1.predict_proba(test_pca_1))
            # test_list_2 = np.squeeze(classifier_2.predict_proba(test_pca_2))
            if classifier_1.predict(test_pca_1) == classifier_2.predict(test_pca_2):
                accuracy_test += classifier_1.predict(test_pca_1) == test_label[i]
                if classifier_1.predict(test_pca_1) == test_label[i] and max_index_1 != test_label[i]:
                    correct += 1
                if classifier_1.predict(test_pca_1) != test_label[i] and max_index_1 == test_label[i]:
                    wrong += 1
            else:
                accuracy_test += max_index_1 == test_label[i]

            # if max(test_list_1) >= max(test_list_2):
            #     accuracy_test += classifier_1.predict(test_pca_1) == test_label[i]
            #     if classifier_1.predict(test_pca_1) == test_label[i] and max_index_1 != test_label[i]:
            #         correct += 1
            #     if classifier_1.predict(test_pca_1) != test_label[i] and max_index_1 == test_label[i]:
            #         wrong += 1

            # else:
            #     accuracy_test += classifier_2.predict(test_pca_2) == test_label[i]
            #     if classifier_2.predict(test_pca_2) == test_label[i] and max_index_1 != test_label[i]:
            #         correct += 1
            #     if classifier_2.predict(test_pca_2) != test_label[i] and max_index_1 == test_label[i]:
            #         wrong += 1

            # if max_index_1 > max_index_2:
            #     if test_list_1[1] >= test_list_2[0]:
            #         accuracy_test += max_index_1 == test_label[i]
            #     else:
            #         accuracy_test += max_index_2 == test_label[i]
            # else:
            #     if test_list_1[0] >= test_list_2[1]:
            #         accuracy_test += max_index_1 == test_label[i]
            #     else:
            #         accuracy_test += max_index_2 == test_label[i]

    end_time = time()
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}
    print("The number of SVM: {}".format(len(confusion_labels)))
    print("The number images which are needed to be retested: {}".format(count))
    print("The number of right images among retested images: {}".format(right))
    print("Original wrong but now correct: {}".format(correct))
    print("Original correct but now wrong: {}".format(wrong))
    print ("The accuracy of testing set is {}".format(accuracy_test))
    print ('The total time for classification: %(minute)d minute(s) %(second)d second(s)' % time_total)
    print ('')


if __name__ == "__main__":
    main()

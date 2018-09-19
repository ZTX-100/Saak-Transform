from sklearn.decomposition import PCA
import numpy as np
import cPickle
import gzip
from sklearn import svm
# import matplotlib.pyplot as plt
# from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
# from sklearn.ensemble import RandomForestClassifier

f = gzip.open('./mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_label = np.concatenate((train_set[1], valid_set[1]))
test_label = test_set[1]

# (dc,ac,-ac) -> (dc,ac)


def Unsign(train_data):
    filternum = (train_data.shape[3] - 1) / 2
    ta1 = np.concatenate((train_data[:, :, :, :1], train_data[:, :, :, 1:filternum + 1] - train_data[:, :, :, filternum + 1:]), axis=3)
    return ta1.reshape(ta1.shape[0], -1)


# for i in range(10):
num = 0
class_id = 'class' + '7' + '_'
# load features
train_data = np.load('./feature/train_before_f_test_' + class_id + '1_v' + str(num) + '.npy')
print (train_data.shape)
ta = Unsign(train_data)
print(np.count_nonzero(ta))
train_data = np.load('./feature/train_before_f_test_' + class_id + '2_v' + str(num) + '.npy')
ta = np.concatenate((ta, Unsign(train_data)), 1)
print (ta.shape)
print(np.count_nonzero(ta))
train_data = np.load('./feature/train_before_f_test_' + class_id + '3_v' + str(num) + '.npy')
ta = np.concatenate((ta, Unsign(train_data)), 1)
print(np.count_nonzero(ta))
print (ta.shape)
train_data = np.load('./feature/train_before_f_test_' + class_id + '4_v' + str(num) + '.npy')
ta = np.concatenate((ta, Unsign(train_data)), 1)
print(np.count_nonzero(ta))
print (ta.shape)
train_data = np.load('./feature/train_before_f_test_' + class_id + '5_v' + str(num) + '.npy')
ta = np.concatenate((ta, Unsign(train_data)), 1)
print(np.count_nonzero(ta))
print (ta.shape)
num_traindata = 60000
train_data = (ta[:num_traindata])
print("train_data.shape: {}".format(train_data.shape))
# idx = train_data < 0
# train_data[idx] = -train_data[idx]
train_label = train_label[:num_traindata]
test_data = np.load('./feature/test_before_f_test_' + class_id + '1_v' + str(num) + '.npy')
ta = Unsign(test_data)
print (ta.shape)
test_data = np.load('./feature/test_before_f_test_' + class_id + '2_v' + str(num) + '.npy')
ta = np.concatenate((ta, Unsign(test_data)), 1)
print (ta.shape)
test_data = np.load('./feature/test_before_f_test_' + class_id + '3_v' + str(num) + '.npy')
ta = np.concatenate((ta, Unsign(test_data)), 1)
print (ta.shape)
test_data = np.load('./feature/test_before_f_test_' + class_id + '4_v' + str(num) + '.npy')
ta = np.concatenate((ta, Unsign(test_data)), 1)
print (ta.shape)
test_data = np.load('./feature/test_before_f_test_' + class_id + '5_v' + str(num) + '.npy')
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

"""
@ PCA to 64
"""
pca = PCA(n_components=64)
pca.fit(train_coefficients_f_test)
train_coefficients_pca = pca.transform(train_coefficients_f_test)
test_coefficients_pca = pca.transform(test_coefficients_f_test)

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
accuracy_train = classifier.score(train_coefficients_pca, train_label)
accuracy_test = classifier.score(test_coefficients_pca, test_label)
# print("label is: {}".format(i))
print (classifier.predict_proba(test_coefficients_pca))
print ("The accuracy of training set is {:.4f}".format(accuracy_train))
print ("The accuracy of testing set is {:.4f}".format(accuracy_test))
print ('')


# def evac_ftest(rep2, label):
#     F, p = f_classif(rep2, label)
#     low_conf = p > 0.05
#     F[low_conf] = 0
#     where_are_NaNs = np.isnan(F)
#     F[where_are_NaNs] = 0
#     return F


# def evac_RFS(feature, label):
#     from skfeature.function.sparse_learning_based import RFS
#     from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking

#     Label = construct_label_matrix(label)
#     # obtain the feature weight matrix
#     Weight = RFS.rfs(feature, Label, gamma=0.1)

#     # sort the feature scores in an ascending order according to the feature scores
#     idx = feature_ranking(Weight)
#     return idx


# # obtain the dataset on the selected features
# # selected_features = X[:, idx[0:num_fea]]
# EVA = 'FTest'
# save = False
# if EVA == 'FTest':
#     print 'computer f-test:'
#     if True:
#         Eva = evac_ftest(train_data, train_label)
#         # np.save('./EVA/eva_ftest_v'+str(num)+'.npy',Eva)
#     else:
#         Eva = np.load('./EVA/eva_ftest_v' + str(num) + '.npy')
#     Eva_cnt = Eva.shape[0]
#     print Eva_cnt, np.count_nonzero(Eva)
#     idx = Eva > np.sort(Eva)[::-1][int(np.count_nonzero(Eva) * 0.75) - 1]
#     idx2 = np.argsort(Eva)[::-1]
# if EVA == 'RFS':
#     print 'computer RFS:'
#     num_fea = 200
#     if save == True:
#         idx = evac_RFS(train_data, train_label)
#         np.save('./EVA/eva_RFS_v' + str(num) + '.npy', idx)
#     else:
#         idx = np.load('./EVA/eva_RFS_v' + str(num) + '.npy')
#     idx = idx[0:num_fea]

# if EVA == 'MRMR':
#     print 'computer MRMR:'
#     num_fea = 200
#     from skfeature.function.information_theoretical_based import MRMR
#     if save == True:
#         idx = MRMR.mrmr(train_data, train_label)  # , n_selected_features=num_fea)
#         np.save('./EVA/eva_MRMR_v' + str(num) + '.npy', idx)
#     else:
#         idx = np.load('./EVA/eva_MRMR_v' + str(num) + '.npy')
#     idx = idx[:num_fea]
# if EVA == 'RF':
#     print 'computer ReliefF:'
#     num_fea = 200
#     from skfeature.function.similarity_based import reliefF
#     if save == True:
#         score = reliefF.reliefF(train_data, train_label)
#         # rank features in descending order according to score
#         idx = reliefF.feature_ranking(score)
#         np.save('./EVA/eva_RF_v' + str(num) + '.npy', idx)
#     else:
#         idx = np.load('./EVA/eva_RF_v' + str(num) + '.npy')
#     idx = idx[:num_fea]

# train_data = train_data[:, idx]
# test_data = test_data[:, idx]

# if False:
#     import matplotlib.pyplot as plt

#     for f_id in range(1):
#         f_id = 14
#         plt.figure(f_id)
#         # the histogram of the data
#         for class_id in range(10):
#             x = train_data[train_label == class_id]
#             y, binEdges = np.histogram(x[:, f_id], bins=20)
#             bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
#             plt.plot(bincenters, y, '-', label='class' + str(class_id))

#             # plt.xlabel('Smarts')
#             # plt.ylabel('Probability')
#         plt.grid(True)
#         plt.legend(loc=2)
#         plt.show()

# if False:
#     print 'computer svm:'
#     clf = svm.SVC()
#     clf.fit(train_data, train_label)
#     pre = clf.predict(test_data)
#     print np.count_nonzero(pre == test_label)
# if True:
#     # feature reduction
#     print 'computer pca:'
#     pca = PCA(svd_solver='full')
#     pca.fit(train_data)
#     pca_k = pca.components_
#     for n_components in [64]:
#         W = pca_k[:n_components, :]
#         traindata = np.dot(train_data, np.transpose(W))
#         testdata = np.dot(test_data, np.transpose(W))
#         print 'computer svm:'
#         clf = svm.SVC()
#         clf.fit(traindata, train_label)
#         pre = clf.predict(testdata)
#         print 'reduce dim to %d:' % n_components
#         print np.count_nonzero(pre == test_label)

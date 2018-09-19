# load libs
import torch
from torchvision import transforms
# import matplotlib.pyplot as plt
import numpy as np
from data.datasets import MNIST
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.svm import SVC
from time import time
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN

'''
@ Original 28x28 is rescaled to 32x32 to meet 2^P size
@ batch_size and workders can be increased for faster loading
'''
print(torch.__version__)
batch_size = 1
test_batch_size = 1
kwargs = {}
train_loader = data_utils.DataLoader(MNIST(root='./data', train=True, process=False, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])), batch_size=batch_size, shuffle=True, **kwargs)


test_loader = data_utils.DataLoader(MNIST(root='./data', train=False, process=False, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])), batch_size=test_batch_size, shuffle=True, **kwargs)


'''
@ For demo use, only use first 1000 samples
'''


def create_numpy_dataset_train():
    dataset = []
    labels = []

    for data in train_loader:
        data_numpy = data[0].numpy()
        label_numpy = data[1].numpy()

        data_numpy = np.squeeze(data_numpy)
        label_numpy = np.squeeze(label_numpy)

        dataset.append(data_numpy)
        labels.append(label_numpy)

    dataset = np.array(dataset)
    labels = np.array(labels)

    dataset = np.expand_dims(dataset, axis=1)
    print("Numpy training dataset shape is {}".format(dataset.shape))
    print("Numpy training labels shape is {}".format(labels.shape))

    # return datasets
    return dataset, labels


def create_numpy_dataset_test():
    dataset = []
    labels = []

    for data in test_loader:
        data_numpy = data[0].numpy()
        label_numpy = data[1].numpy()

        data_numpy = np.squeeze(data_numpy)
        label_numpy = np.squeeze(label_numpy)

        dataset.append(data_numpy)
        labels.append(label_numpy)

    dataset = np.array(dataset)
    labels = np.array(labels)

    dataset = np.expand_dims(dataset, axis=1)
    print("Numpy testing dataset shape is {}".format(dataset.shape))
    print("Numpy testing labels shape is {}".format(labels.shape))

    # dataset = 255 - dataset
    # return datasets
    return dataset[:1000, :, :, :], labels

    '''
@ return: augmented anchors
'''


def PCA_and_augment(data_in):
    # data reshape
    data = np.reshape(data_in, (data_in.shape[0], -1))
    print("PCA_and_augment: {}".format(data.shape))
    # mean removal
    mean = np.mean(data, axis=0)
    datas_mean_remov = data - mean
    print("PCA_and_augment meanremove shape: {}".format(datas_mean_remov.shape))

    # PCA, retain all components
    pca = PCA()
    pca.fit(datas_mean_remov)
    comps = pca.components_
    print("pca matrix shape: {}".format(comps.shape))

    # augment, DC component doesn't
    comps_aug = [vec * (-1) for vec in comps[:-1]]
    comps_complete = np.vstack((comps, comps_aug))
    print("PCA_and_augment comps_complete shape: {}".format(comps_complete.shape))
    return comps_complete


'''
@ depth: determine shape, initial: 0
'''


def fit_pca_shape(datasets, depth):
    factor = np.power(2, depth)
    length = 32 / factor
    length = length.astype(np.int64)
    print("fit_pca_shape: length: {}".format(length))
    idx1 = range(0, length, 2)
    idx2 = [i + 2 for i in idx1]
    print("fit_pca_shape: idx1: {}".format(idx1))
    data_lattice = [datasets[:, :, i:j, k:l] for ((i, j), (k, l)) in product(zip(idx1, idx2), zip(idx1, idx2))]
    data_lattice = np.array(data_lattice)
    print("fit_pca_shape: data_lattice.shape: {}".format(data_lattice.shape))

    # shape reshape
    data = np.reshape(data_lattice, (data_lattice.shape[0] * data_lattice.shape[1], data_lattice.shape[2], 2, 2))
    print("fit_pca_shape: reshape: {}".format(data.shape))
    return data


'''
@ Prepare shape changes. 
@ return filters for convolution
@ aug_anchors: [out_num*in_num,4] -> [out_num,in_num,2,2]
'''


def ret_filt_patches(aug_anchors, input_channels):
    shape = aug_anchors.shape[1] / 4
    num = aug_anchors.shape[0]
    shape = int(shape)
    filt = np.reshape(aug_anchors, (num, shape, 4))

    # reshape to kernels, (# output_channels,# input_channels,2,2)
    filters = np.reshape(filt, (num, shape, 2, 2))

    return filters


'''
@ input: kernel and data
@ output: conv+relu result
'''


def conv_and_relu(filters, datasets, stride=2):
    # print(datasets.shape)
    # torch data change
    filters_t = torch.from_numpy(filters)
    datasets_t = torch.from_numpy(datasets)

    # Variables
    filt = Variable(filters_t).type(torch.FloatTensor)
    data = Variable(datasets_t).type(torch.FloatTensor)

    # Convolution
    output = F.conv2d(data, filt, stride=stride)

    # Relu
    relu_output = F.relu(output)

    return relu_output


def saak_coefficients_test(datasets=None, index=[0, 0, 0], depth=0, components_pca=0, input_channels=0):

    datasets = np.array(datasets)
    datasets = np.reshape(datasets, (datasets.shape[0], index[0], index[1], index[2]))

    # change data shape, (14*60000,4)
    data_flatten = fit_pca_shape(datasets, depth)

    # augmented components, first round: (7,4), only augment AC components
    comps_complete = PCA_and_augment(data_flatten)
    comps_complete = comps_complete[:components_pca + 1, :]
    print("one_stage_saak_trans: comps_complete: {}".format(comps_complete.shape))

    # get filter, (7,1,2,2)
    filters = ret_filt_patches(comps_complete, input_channels)
    print("one_stage_saak_trans: filters: {}".format(filters.shape))

    # output (60000,7,14,14)
    relu_output = conv_and_relu(filters, datasets, stride=2)
    data = relu_output.data.numpy()
    print("one_stage_saak_trans: data: {}".format(data.shape))
    data = list(data)
    return data


def meanshift_cluster(data_in):
    data = np.reshape(data_in, (data_in.shape[0], -1))
    # count_choice = np.random.choice(np.arange(data.shape[0]), size=int(data.shape[0] * 0.01), replace=False)
    # count_choice = range(int(data.shape[0] * 0.01))
    # bandwidth = estimate_bandwidth(X=data[count_choice, :], quantile=0.005)
    # print("bandwidth:                                        {}".format(bandwidth))
    meanshift = DBSCAN()
    K = meanshift.fit_predict(data)
    zip_data = zip(data, K)
    data = []
    for i in range(len(np.unique(K))):
        data.append([dat for dat, k in zip_data if k == i])

    return data, K


'''
@ One-stage Saak transform
@ input: datasets [60000,channel,size,size]
'''


def one_stage_saak_trans_test(datasets=None, depth=0, components_pca=0):

    # intial dataset, (60000,1,32,32)
    # channel change: 1->7
    print("one_stage_saak_trans: datasets.shape {}".format(datasets.shape))
    input_channels = datasets.shape[1]
    index = [datasets.shape[1], datasets.shape[2], datasets.shape[3]]
    data_cluster, K = meanshift_cluster(datasets)
    n_clusters = len(np.unique(K))
    print("one_stage_saak_trans: number of clusters: {}".format(n_clusters))
    data_n = []
    for i in range(n_clusters):
        data_n.append(saak_coefficients_test(data_cluster[i], index, depth, components_pca, input_channels))

    data = np.empty([datasets.shape[0], np.array(data_n[0]).shape[1], np.array(data_n[0]).shape[2], np.array(data_n[0]).shape[3]])
    i = [0] * n_clusters
    m = 0
    for k in K:
        data[m, :, :, :] = np.array(data_n[k][i[k]])
        m += 1
        i[k] += 1

    output = list(data)
    return data, output


def one_stage_saak_trans_test(datasets=None, depth=0, components_pca=0):

    # intial dataset, (60000,1,32,32)
    # channel change: 1->7
    print("one_stage_saak_trans: datasets.shape {}".format(datasets.shape))
    input_channels = datasets.shape[1]
    index = [datasets.shape[1], datasets.shape[2], datasets.shape[3]]
    output = saak_coefficients(datasets, index, depth, components_pca, input_channels)

    data = np.array(output)
    return data, output


'''
@ Multi-stage Saak transform
'''


def multi_stage_saak_trans(components_pca):
    # train_filters = []
    train_outputs = []
    # test_filters = []
    test_outputs = []

    train_data, train_labels = create_numpy_dataset_train()
    test_data, test_labels = create_numpy_dataset_test()
    train_dataset = train_data
    test_dataset = test_data
    num = 0
    img_len = train_data.shape[-1]
    while(img_len >= 2):
        num += 1
        img_len /= 2

    for i in range(num):
        print("{} stage of saak transform: ".format(i))
        train_data, train_output = one_stage_saak_trans_test(train_data, depth=i, components_pca=components_pca[i])
        # train_filters.append(train_filt)
        train_outputs.append(train_output)
        test_data, test_output = one_stage_saak_trans_train(test_data, depth=i, components_pca=components_pca[i])
        # test_filters.append(test_filt)
        test_outputs.append(test_output)
        print('')

    return train_dataset, train_labels, test_dataset, test_labels, train_outputs, test_outputs

    """
@ Implement multi-stage Saak trans
"""


def main():
    components_PCA = [3, 4, 7, 6, 8]
    train_size = 60000
    test_size = 1000

    start_time = time()
    train_data, train_labels, test_data, test_labels, train_outputs, test_outputs = multi_stage_saak_trans(components_pca=components_PCA)
    mid_time = time()

    # test_outputs = []
    # for filt in train_filters:
    #     filter_numpy = filt.data.numpy()
    #     relu_output, filt = conv_and_relu(filter_numpy, test_data, stride=2)
    #     test_data = relu_output.data.numpy()
    #     test_outputs.append(relu_output)

    train_coefficient = []
    for output in train_outputs:
        output = np.array(output)
        train_coefficient.append(output.reshape(train_size, -1))
    train_coefficients = np.concatenate((train_coefficient), axis=1)

    test_coefficient = []
    for output in test_outputs:
        output = np.array(output)
        test_coefficient.append(output.reshape(test_size, -1))
    test_coefficients = np.concatenate((test_coefficient), axis=1)

    """
    @ F-test feature selection
    """

    selector = SelectKBest(f_classif, k=1000)
    selector.fit(train_coefficients, train_labels)
    train_coefficients_f_test = selector.transform(train_coefficients)
    test_coefficients_f_test = selector.transform(test_coefficients)

    """
    @ PCA to 64
    """
    pca = PCA(n_components=64)
    pca.fit(train_coefficients_f_test)
    train_coefficients_pca = pca.transform(train_coefficients_f_test)
    test_coefficients_pca = pca.transform(test_coefficients_f_test)

    print ('Numpy training saak coefficients shape: {}'.format(train_coefficients.shape))
    print ('Numpy training F-test coefficients shape: {}'.format(train_coefficients_f_test.shape))
    print ('Numpy training PCA coefficients shape: {}'.format(train_coefficients_pca.shape))
    print ('Numpy testing saak coefficients shape: {}'.format(test_coefficients.shape))
    print ('Numpy testing F-test coefficients shape: {}'.format(test_coefficients_f_test.shape))
    print ('Numpy testing PCA coefficients shape: {}'.format(test_coefficients_pca.shape))

    """
    @ SVM classifier
    """
    classifier = SVC()
    classifier.fit(train_coefficients_pca, train_labels)
    accuracy_train = classifier.score(train_coefficients_pca, train_labels)
    accuracy_test = classifier.score(test_coefficients_pca, test_labels)

    end_time = time()
    time_coefficients = mid_time - start_time
    minutes, seconds = divmod(end_time - start_time, 60)
    time_total = {'minute': minutes, 'second': seconds}

    print("The accuracy of training set is {:.4f}".format(accuracy_train))
    print ("The accuracy of testing set is {:.4f}".format(accuracy_test))
    print ('The time spent for generating saak coefficients is: %d seconds' % time_coefficients)
    print ('The total time: %(minute)d minute(s) %(second)d second(s)' % time_total)
    print ('')


if __name__ == '__main__':
    main()

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
from sklearn.cluster import KMeans
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
    return dataset, labels

    '''
@ return: augmented anchors
'''


def PCA_and_augment(data_in):
    # data reshape
    # data = np.reshape(data_in, (data_in.shape[0], -1))
    # print("PCA_and_augment: {}".format(data.shape))
    # mean removal
    mean = np.mean(data_in, axis=0)
    datas_mean_remov = data_in - mean
    print("PCA_and_augment meanremove shape: {}".format(datas_mean_remov.shape))

    # PCA, retain all components
    pca = PCA()
    pca.fit(datas_mean_remov)
    comps = pca.components_
    print("pca matrix shape: {}".format(comps.shape))

    # augment, DC component doesn't
    comps_aug = [vec * (-1) for vec in comps[:-1]]
    comps_complete = np.vstack((comps, comps_aug))

    # comps_complete = []
    # for vec in comps:
    #     comps_complete.append(vec)
    #     comps_complete.append(vec * (-1))
    # comps_complete = np.array(comps_complete)
    # print("PCA_and_augment comps shape: {}".format(comps.shape))
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
    data_lattice = [data_lattice[:, i, :, :, :] for i in np.arange(data_lattice.shape[1])]
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


# cluster the saak coefficients using k-mean


def kmeans_cluster(data_in):
    data = np.reshape(data_in, (data_in.shape[0], -1))
    # count_choice = np.random.choice(np.arange(data.shape[0]), size=int(data.shape[0] * 0.01), replace=False)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
    K = kmeans.labels_
    zip_data = zip(data, K)
    data_0 = [dat for dat, k in zip_data if k == 0]
    data_1 = [dat for dat, k in zip_data if k == 1]
    data_2 = [dat for dat, k in zip_data if k == 2]
    data_3 = [dat for dat, k in zip_data if k == 3]
    data_0 = np.array(data_0)
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_3 = np.array(data_3)

    return data_0, data_1, data_2, data_3, K


'''
@ One-stage Saak transform
@ input: datasets [60000,channel,size,size]
'''


def one_stage_saak_trans(datasets=None, depth=0, components_pca=0):

    # intial dataset, (60000,1,32,32)
    # channel change: 1->7
    print("one_stage_saak_trans: datasets.shape {}".format(datasets.shape))
    input_channels = datasets.shape[1]

    # change data shape, (14*60000,4)
    data_flatten = fit_pca_shape(datasets, depth)
    data_flatten_0, data_flatten_1, data_flatten_2, data_flatten_3, K = kmeans_cluster(data_flatten)
    # data_flatten = np.reshape(data_flatten, (data_flatten.shape[0], -1))
    data_0 = np.reshape(data_flatten_0, (-1, datasets.shape[1], 2, 2))
    data_1 = np.reshape(data_flatten_1, (-1, datasets.shape[1], 2, 2))
    data_2 = np.reshape(data_flatten_2, (-1, datasets.shape[1], 2, 2))
    data_3 = np.reshape(data_flatten_3, (-1, datasets.shape[1], 2, 2))
    print("PCA_and_augment: data_flatten_0: {}".format(data_flatten_0.shape))
    print("PCA_and_augment: data_flatten_1: {}".format(data_flatten_1.shape))
    print("PCA_and_augment: data_flatten_2: {}".format(data_flatten_2.shape))
    print("PCA_and_augment: data_flatten_3: {}".format(data_flatten_3.shape))
    print("PCA_and_augment: data_0: {}".format(data_0.shape))
    print("PCA_and_augment: data_1: {}".format(data_1.shape))
    print("PCA_and_augment: data_2: {}".format(data_2.shape))
    print("PCA_and_augment: data_3: {}".format(data_3.shape))

    # augmented components, first round: (7,4), only augment AC components
    comps_complete_0 = PCA_and_augment(data_flatten_0)
    comps_complete_1 = PCA_and_augment(data_flatten_1)
    comps_complete_2 = PCA_and_augment(data_flatten_2)
    comps_complete_3 = PCA_and_augment(data_flatten_3)

    comps_complete_0 = comps_complete_0[:components_pca + 1, :]
    comps_complete_1 = comps_complete_1[:components_pca + 1, :]
    comps_complete_2 = comps_complete_2[:components_pca + 1, :]
    comps_complete_3 = comps_complete_3[:components_pca + 1, :]
    # filters = np.expand_dims(comps_complete, axis=1)
    print("one_stage_saak_trans: comps_complete_0: {}".format(comps_complete_0.shape))
    print("one_stage_saak_trans: comps_complete_1: {}".format(comps_complete_1.shape))
    print("one_stage_saak_trans: comps_complete_2: {}".format(comps_complete_2.shape))
    print("one_stage_saak_trans: comps_complete_3: {}".format(comps_complete_3.shape))
    # print("one_stage_saak_trans: filters: {}".format(filters.shape))

    # get filter, (7,1,2,2)
    filters_0 = ret_filt_patches(comps_complete_0, input_channels)
    filters_1 = ret_filt_patches(comps_complete_1, input_channels)
    filters_2 = ret_filt_patches(comps_complete_2, input_channels)
    filters_3 = ret_filt_patches(comps_complete_3, input_channels)
    print("one_stage_saak_trans: filters_0: {}".format(filters_0.shape))
    print("one_stage_saak_trans: filters_1: {}".format(filters_1.shape))
    print("one_stage_saak_trans: filters_2: {}".format(filters_2.shape))
    print("one_stage_saak_trans: filters_3: {}".format(filters_3.shape))

    # output (60000,7,14,14)
    relu_output_0 = conv_and_relu(filters_0, data_0, stride=2)
    relu_output_1 = conv_and_relu(filters_1, data_1, stride=2)
    relu_output_2 = conv_and_relu(filters_2, data_2, stride=2)
    relu_output_3 = conv_and_relu(filters_3, data_3, stride=2)
    data_0 = relu_output_0.data.numpy()
    data_1 = relu_output_1.data.numpy()
    data_2 = relu_output_2.data.numpy()
    data_3 = relu_output_3.data.numpy()
    print(data_0.shape)
    print(data_1.shape)
    print(data_2.shape)
    print(data_3.shape)
    data_0 = np.squeeze(data_0)
    data_1 = np.squeeze(data_1)
    data_2 = np.squeeze(data_2)
    data_3 = np.squeeze(data_3)
    print(data_0.shape)
    print(data_1.shape)
    print(data_2.shape)
    print(data_3.shape)
    data = np.empty([data_flatten.shape[0], data_0.shape[1]])
    i0, i1, i2, i3, m = 0, 0, 0, 0, 0
    for k in K:
        if k == 0:
            data[m, :] = data_0[i0, :]
            m += 1
            i0 += 1
        elif k == 1:
            data[m, :] = data_1[i1, :]
            m += 1
            i1 += 1
        elif k == 2:
            data[m, :] = data_2[i2, :]
            m += 1
            i2 += 1
        else:
            data[m, :] = data_3[i3, :]
            m += 1
            i3 += 1

    data = np.array(data)
    print(data.shape)
    data = np.transpose(data)
    print(data.shape)
    data = np.reshape(data, (data.shape[0], datasets.shape[0], datasets.shape[2] / 2, -1))
    print(data.shape)
    data = [data[:, n, :, :] for n in np.arange(datasets.shape[0])]
    data = np.array(data)
    print(data.shape)
    output = list(data)
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
        train_data, train_output = one_stage_saak_trans(train_data, depth=i, components_pca=components_pca[i])
        # train_filters.append(train_filt)
        train_outputs.append(train_output)
        test_data, test_output = one_stage_saak_trans(test_data, depth=i, components_pca=components_pca[i])
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
    test_size = 10000

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

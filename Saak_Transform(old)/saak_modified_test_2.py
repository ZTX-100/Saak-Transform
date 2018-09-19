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
from sklearn.cluster import MiniBatchKMeans

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


def kmeans_cluster_train(data_in):
    data = np.reshape(data_in, (data_in.shape[0], -1))
    # count_choice = np.random.choice(np.arange(data.shape[0]), size=int(data.shape[0] * 0.01), replace=False)
    kmeans = MiniBatchKMeans(n_clusters=30).fit(data)
    K = kmeans.labels_
    zip_data = zip(data, K)
    data_0 = [dat for dat, k in zip_data if k == 0]
    data_1 = [dat for dat, k in zip_data if k == 1]
    data_2 = [dat for dat, k in zip_data if k == 2]
    data_3 = [dat for dat, k in zip_data if k == 3]
    data_4 = [dat for dat, k in zip_data if k == 4]
    data_5 = [dat for dat, k in zip_data if k == 5]
    data_6 = [dat for dat, k in zip_data if k == 6]
    data_7 = [dat for dat, k in zip_data if k == 7]
    data_8 = [dat for dat, k in zip_data if k == 8]
    data_9 = [dat for dat, k in zip_data if k == 9]
    data_10 = [dat for dat, k in zip_data if k == 10]
    data_11 = [dat for dat, k in zip_data if k == 11]
    data_12 = [dat for dat, k in zip_data if k == 12]
    data_13 = [dat for dat, k in zip_data if k == 13]
    data_14 = [dat for dat, k in zip_data if k == 14]
    data_15 = [dat for dat, k in zip_data if k == 15]
    data_16 = [dat for dat, k in zip_data if k == 16]
    data_17 = [dat for dat, k in zip_data if k == 17]
    data_18 = [dat for dat, k in zip_data if k == 18]
    data_19 = [dat for dat, k in zip_data if k == 19]
    data_20 = [dat for dat, k in zip_data if k == 20]
    data_21 = [dat for dat, k in zip_data if k == 21]
    data_22 = [dat for dat, k in zip_data if k == 22]
    data_23 = [dat for dat, k in zip_data if k == 23]
    data_24 = [dat for dat, k in zip_data if k == 24]
    data_25 = [dat for dat, k in zip_data if k == 25]
    data_26 = [dat for dat, k in zip_data if k == 26]
    data_27 = [dat for dat, k in zip_data if k == 27]
    data_28 = [dat for dat, k in zip_data if k == 28]
    data_29 = [dat for dat, k in zip_data if k == 29]
    data_0 = np.array(data_0)
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_3 = np.array(data_3)
    data_4 = np.array(data_4)
    data_5 = np.array(data_5)
    data_6 = np.array(data_6)
    data_7 = np.array(data_7)
    data_8 = np.array(data_8)
    data_9 = np.array(data_9)
    data_10 = np.array(data_10)
    data_11 = np.array(data_11)
    data_12 = np.array(data_12)
    data_13 = np.array(data_13)
    data_14 = np.array(data_14)
    data_15 = np.array(data_15)
    data_16 = np.array(data_16)
    data_17 = np.array(data_17)
    data_18 = np.array(data_18)
    data_19 = np.array(data_19)
    data_20 = np.array(data_20)
    data_21 = np.array(data_21)
    data_22 = np.array(data_22)
    data_23 = np.array(data_23)
    data_24 = np.array(data_24)
    data_25 = np.array(data_25)
    data_26 = np.array(data_26)
    data_27 = np.array(data_27)
    data_28 = np.array(data_28)
    data_29 = np.array(data_29)

    return data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_15, data_16, data_17, data_18, data_19, data_20, data_21, data_22, data_23, data_24, data_25, data_26, data_27, data_28, data_29, K


'''
@ One-stage Saak transform
@ input: datasets [60000,channel,size,size]
'''


def one_stage_saak_trans_train(datasets=None, depth=0, components_pca=0):

    # intial dataset, (60000,1,32,32)
    # channel change: 1->7
    print("one_stage_saak_trans: datasets.shape {}".format(datasets.shape))
    input_channels = datasets.shape[1]

    datasets_0, datasets_1, datasets_2, datasets_3, datasets_4, datasets_5, datasets_6, datasets_7, datasets_8, datasets_9, datasets_10, datasets_11, datasets_12, datasets_13, datasets_14, datasets_15, datasets_16, datasets_17, datasets_18, datasets_19, datasets_20, datasets_21, datasets_22, datasets_23, datasets_24, datasets_25, datasets_26, datasets_27, datasets_28, datasets_29, K = kmeans_cluster_train(datasets)
    datasets_0 = np.reshape(datasets_0, (datasets_0.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_1 = np.reshape(datasets_1, (datasets_1.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_2 = np.reshape(datasets_2, (datasets_2.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_3 = np.reshape(datasets_3, (datasets_3.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_4 = np.reshape(datasets_4, (datasets_4.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_5 = np.reshape(datasets_5, (datasets_5.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_6 = np.reshape(datasets_6, (datasets_6.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_7 = np.reshape(datasets_7, (datasets_7.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_8 = np.reshape(datasets_8, (datasets_8.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_9 = np.reshape(datasets_9, (datasets_9.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_10 = np.reshape(datasets_10, (datasets_10.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_11 = np.reshape(datasets_11, (datasets_11.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_12 = np.reshape(datasets_12, (datasets_12.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_13 = np.reshape(datasets_13, (datasets_13.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_14 = np.reshape(datasets_14, (datasets_14.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_15 = np.reshape(datasets_15, (datasets_15.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_16 = np.reshape(datasets_16, (datasets_16.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_17 = np.reshape(datasets_17, (datasets_17.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_18 = np.reshape(datasets_18, (datasets_18.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_19 = np.reshape(datasets_19, (datasets_19.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_20 = np.reshape(datasets_20, (datasets_20.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_21 = np.reshape(datasets_21, (datasets_21.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_22 = np.reshape(datasets_22, (datasets_22.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_23 = np.reshape(datasets_23, (datasets_23.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_24 = np.reshape(datasets_24, (datasets_24.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_25 = np.reshape(datasets_25, (datasets_25.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_26 = np.reshape(datasets_26, (datasets_26.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_27 = np.reshape(datasets_27, (datasets_27.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_28 = np.reshape(datasets_28, (datasets_28.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_29 = np.reshape(datasets_29, (datasets_29.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))

    # change data shape, (14*60000,4)
    data_flatten_0 = fit_pca_shape(datasets_0, depth)
    data_flatten_1 = fit_pca_shape(datasets_1, depth)
    data_flatten_2 = fit_pca_shape(datasets_2, depth)
    data_flatten_3 = fit_pca_shape(datasets_3, depth)
    data_flatten_4 = fit_pca_shape(datasets_4, depth)
    data_flatten_5 = fit_pca_shape(datasets_5, depth)
    data_flatten_6 = fit_pca_shape(datasets_6, depth)
    data_flatten_7 = fit_pca_shape(datasets_7, depth)
    data_flatten_8 = fit_pca_shape(datasets_8, depth)
    data_flatten_9 = fit_pca_shape(datasets_9, depth)
    data_flatten_10 = fit_pca_shape(datasets_10, depth)
    data_flatten_11 = fit_pca_shape(datasets_11, depth)
    data_flatten_12 = fit_pca_shape(datasets_12, depth)
    data_flatten_13 = fit_pca_shape(datasets_13, depth)
    data_flatten_14 = fit_pca_shape(datasets_14, depth)
    data_flatten_15 = fit_pca_shape(datasets_15, depth)
    data_flatten_16 = fit_pca_shape(datasets_16, depth)
    data_flatten_17 = fit_pca_shape(datasets_17, depth)
    data_flatten_18 = fit_pca_shape(datasets_18, depth)
    data_flatten_19 = fit_pca_shape(datasets_19, depth)
    data_flatten_20 = fit_pca_shape(datasets_20, depth)
    data_flatten_21 = fit_pca_shape(datasets_21, depth)
    data_flatten_22 = fit_pca_shape(datasets_22, depth)
    data_flatten_23 = fit_pca_shape(datasets_23, depth)
    data_flatten_24 = fit_pca_shape(datasets_24, depth)
    data_flatten_25 = fit_pca_shape(datasets_25, depth)
    data_flatten_26 = fit_pca_shape(datasets_26, depth)
    data_flatten_27 = fit_pca_shape(datasets_27, depth)
    data_flatten_28 = fit_pca_shape(datasets_28, depth)
    data_flatten_29 = fit_pca_shape(datasets_29, depth)

    # augmented components, first round: (7,4), only augment AC components
    comps_complete_0 = PCA_and_augment(data_flatten_0)
    comps_complete_1 = PCA_and_augment(data_flatten_1)
    comps_complete_2 = PCA_and_augment(data_flatten_2)
    comps_complete_3 = PCA_and_augment(data_flatten_3)
    comps_complete_4 = PCA_and_augment(data_flatten_4)
    comps_complete_5 = PCA_and_augment(data_flatten_5)
    comps_complete_6 = PCA_and_augment(data_flatten_6)
    comps_complete_7 = PCA_and_augment(data_flatten_7)
    comps_complete_8 = PCA_and_augment(data_flatten_8)
    comps_complete_9 = PCA_and_augment(data_flatten_9)
    comps_complete_10 = PCA_and_augment(data_flatten_10)
    comps_complete_11 = PCA_and_augment(data_flatten_11)
    comps_complete_12 = PCA_and_augment(data_flatten_12)
    comps_complete_13 = PCA_and_augment(data_flatten_13)
    comps_complete_14 = PCA_and_augment(data_flatten_14)
    comps_complete_15 = PCA_and_augment(data_flatten_15)
    comps_complete_16 = PCA_and_augment(data_flatten_16)
    comps_complete_17 = PCA_and_augment(data_flatten_17)
    comps_complete_18 = PCA_and_augment(data_flatten_18)
    comps_complete_19 = PCA_and_augment(data_flatten_19)
    comps_complete_20 = PCA_and_augment(data_flatten_20)
    comps_complete_21 = PCA_and_augment(data_flatten_21)
    comps_complete_22 = PCA_and_augment(data_flatten_22)
    comps_complete_23 = PCA_and_augment(data_flatten_23)
    comps_complete_24 = PCA_and_augment(data_flatten_24)
    comps_complete_25 = PCA_and_augment(data_flatten_25)
    comps_complete_26 = PCA_and_augment(data_flatten_26)
    comps_complete_27 = PCA_and_augment(data_flatten_27)
    comps_complete_28 = PCA_and_augment(data_flatten_28)
    comps_complete_29 = PCA_and_augment(data_flatten_29)

    comps_complete_0 = comps_complete_0[:components_pca + 1, :]
    comps_complete_1 = comps_complete_1[:components_pca + 1, :]
    comps_complete_2 = comps_complete_2[:components_pca + 1, :]
    comps_complete_3 = comps_complete_3[:components_pca + 1, :]
    comps_complete_4 = comps_complete_4[:components_pca + 1, :]
    comps_complete_5 = comps_complete_5[:components_pca + 1, :]
    comps_complete_6 = comps_complete_6[:components_pca + 1, :]
    comps_complete_7 = comps_complete_7[:components_pca + 1, :]
    comps_complete_8 = comps_complete_8[:components_pca + 1, :]
    comps_complete_9 = comps_complete_9[:components_pca + 1, :]
    comps_complete_10 = comps_complete_10[:components_pca + 1, :]
    comps_complete_11 = comps_complete_11[:components_pca + 1, :]
    comps_complete_12 = comps_complete_12[:components_pca + 1, :]
    comps_complete_13 = comps_complete_13[:components_pca + 1, :]
    comps_complete_14 = comps_complete_14[:components_pca + 1, :]
    comps_complete_15 = comps_complete_15[:components_pca + 1, :]
    comps_complete_16 = comps_complete_16[:components_pca + 1, :]
    comps_complete_17 = comps_complete_17[:components_pca + 1, :]
    comps_complete_18 = comps_complete_18[:components_pca + 1, :]
    comps_complete_19 = comps_complete_19[:components_pca + 1, :]
    comps_complete_20 = comps_complete_20[:components_pca + 1, :]
    comps_complete_21 = comps_complete_21[:components_pca + 1, :]
    comps_complete_22 = comps_complete_22[:components_pca + 1, :]
    comps_complete_23 = comps_complete_23[:components_pca + 1, :]
    comps_complete_24 = comps_complete_24[:components_pca + 1, :]
    comps_complete_25 = comps_complete_25[:components_pca + 1, :]
    comps_complete_26 = comps_complete_26[:components_pca + 1, :]
    comps_complete_27 = comps_complete_27[:components_pca + 1, :]
    comps_complete_28 = comps_complete_28[:components_pca + 1, :]
    comps_complete_29 = comps_complete_29[:components_pca + 1, :]

    print("one_stage_saak_trans: comps_complete_0: {}".format(comps_complete_0.shape))
    print("one_stage_saak_trans: comps_complete_1: {}".format(comps_complete_1.shape))
    print("one_stage_saak_trans: comps_complete_2: {}".format(comps_complete_2.shape))
    print("one_stage_saak_trans: comps_complete_3: {}".format(comps_complete_3.shape))
    print("one_stage_saak_trans: comps_complete_4: {}".format(comps_complete_4.shape))
    print("one_stage_saak_trans: comps_complete_5: {}".format(comps_complete_5.shape))
    print("one_stage_saak_trans: comps_complete_6: {}".format(comps_complete_6.shape))
    print("one_stage_saak_trans: comps_complete_7: {}".format(comps_complete_7.shape))
    print("one_stage_saak_trans: comps_complete_8: {}".format(comps_complete_8.shape))
    print("one_stage_saak_trans: comps_complete_9: {}".format(comps_complete_9.shape))
    print("one_stage_saak_trans: comps_complete_10: {}".format(comps_complete_10.shape))
    print("one_stage_saak_trans: comps_complete_11: {}".format(comps_complete_11.shape))
    print("one_stage_saak_trans: comps_complete_12: {}".format(comps_complete_12.shape))
    print("one_stage_saak_trans: comps_complete_13: {}".format(comps_complete_13.shape))
    print("one_stage_saak_trans: comps_complete_14: {}".format(comps_complete_14.shape))
    print("one_stage_saak_trans: comps_complete_15: {}".format(comps_complete_15.shape))
    print("one_stage_saak_trans: comps_complete_16: {}".format(comps_complete_16.shape))
    print("one_stage_saak_trans: comps_complete_17: {}".format(comps_complete_17.shape))
    print("one_stage_saak_trans: comps_complete_18: {}".format(comps_complete_18.shape))
    print("one_stage_saak_trans: comps_complete_19: {}".format(comps_complete_19.shape))
    print("one_stage_saak_trans: comps_complete_20: {}".format(comps_complete_20.shape))
    print("one_stage_saak_trans: comps_complete_21: {}".format(comps_complete_21.shape))
    print("one_stage_saak_trans: comps_complete_22: {}".format(comps_complete_22.shape))
    print("one_stage_saak_trans: comps_complete_23: {}".format(comps_complete_23.shape))
    print("one_stage_saak_trans: comps_complete_24: {}".format(comps_complete_24.shape))
    print("one_stage_saak_trans: comps_complete_25: {}".format(comps_complete_25.shape))
    print("one_stage_saak_trans: comps_complete_26: {}".format(comps_complete_26.shape))
    print("one_stage_saak_trans: comps_complete_27: {}".format(comps_complete_27.shape))
    print("one_stage_saak_trans: comps_complete_28: {}".format(comps_complete_28.shape))
    print("one_stage_saak_trans: comps_complete_29: {}".format(comps_complete_29.shape))

    # get filter, (7,1,2,2)
    filters_0 = ret_filt_patches(comps_complete_0, input_channels)
    filters_1 = ret_filt_patches(comps_complete_1, input_channels)
    filters_2 = ret_filt_patches(comps_complete_2, input_channels)
    filters_3 = ret_filt_patches(comps_complete_3, input_channels)
    filters_4 = ret_filt_patches(comps_complete_4, input_channels)
    filters_5 = ret_filt_patches(comps_complete_5, input_channels)
    filters_6 = ret_filt_patches(comps_complete_6, input_channels)
    filters_7 = ret_filt_patches(comps_complete_7, input_channels)
    filters_8 = ret_filt_patches(comps_complete_8, input_channels)
    filters_9 = ret_filt_patches(comps_complete_9, input_channels)
    filters_10 = ret_filt_patches(comps_complete_10, input_channels)
    filters_11 = ret_filt_patches(comps_complete_11, input_channels)
    filters_12 = ret_filt_patches(comps_complete_12, input_channels)
    filters_13 = ret_filt_patches(comps_complete_13, input_channels)
    filters_14 = ret_filt_patches(comps_complete_14, input_channels)
    filters_15 = ret_filt_patches(comps_complete_15, input_channels)
    filters_16 = ret_filt_patches(comps_complete_16, input_channels)
    filters_17 = ret_filt_patches(comps_complete_17, input_channels)
    filters_18 = ret_filt_patches(comps_complete_18, input_channels)
    filters_19 = ret_filt_patches(comps_complete_19, input_channels)
    filters_20 = ret_filt_patches(comps_complete_20, input_channels)
    filters_21 = ret_filt_patches(comps_complete_21, input_channels)
    filters_22 = ret_filt_patches(comps_complete_22, input_channels)
    filters_23 = ret_filt_patches(comps_complete_23, input_channels)
    filters_24 = ret_filt_patches(comps_complete_24, input_channels)
    filters_25 = ret_filt_patches(comps_complete_25, input_channels)
    filters_26 = ret_filt_patches(comps_complete_26, input_channels)
    filters_27 = ret_filt_patches(comps_complete_27, input_channels)
    filters_28 = ret_filt_patches(comps_complete_28, input_channels)
    filters_29 = ret_filt_patches(comps_complete_29, input_channels)
    print("one_stage_saak_trans: filters_0: {}".format(filters_0.shape))
    print("one_stage_saak_trans: filters_1: {}".format(filters_1.shape))
    print("one_stage_saak_trans: filters_2: {}".format(filters_2.shape))
    print("one_stage_saak_trans: filters_3: {}".format(filters_3.shape))
    print("one_stage_saak_trans: filters_4: {}".format(filters_4.shape))
    print("one_stage_saak_trans: filters_5: {}".format(filters_5.shape))
    print("one_stage_saak_trans: filters_6: {}".format(filters_6.shape))
    print("one_stage_saak_trans: filters_7: {}".format(filters_7.shape))
    print("one_stage_saak_trans: filters_8: {}".format(filters_8.shape))
    print("one_stage_saak_trans: filters_9: {}".format(filters_9.shape))
    print("one_stage_saak_trans: filters_10: {}".format(filters_10.shape))
    print("one_stage_saak_trans: filters_11: {}".format(filters_11.shape))
    print("one_stage_saak_trans: filters_12: {}".format(filters_12.shape))
    print("one_stage_saak_trans: filters_13: {}".format(filters_13.shape))
    print("one_stage_saak_trans: filters_14: {}".format(filters_14.shape))
    print("one_stage_saak_trans: filters_15: {}".format(filters_15.shape))
    print("one_stage_saak_trans: filters_16: {}".format(filters_16.shape))
    print("one_stage_saak_trans: filters_17: {}".format(filters_17.shape))
    print("one_stage_saak_trans: filters_18: {}".format(filters_18.shape))
    print("one_stage_saak_trans: filters_19: {}".format(filters_19.shape))
    print("one_stage_saak_trans: filters_20: {}".format(filters_20.shape))
    print("one_stage_saak_trans: filters_21: {}".format(filters_21.shape))
    print("one_stage_saak_trans: filters_22: {}".format(filters_22.shape))
    print("one_stage_saak_trans: filters_23: {}".format(filters_23.shape))
    print("one_stage_saak_trans: filters_24: {}".format(filters_24.shape))
    print("one_stage_saak_trans: filters_25: {}".format(filters_25.shape))
    print("one_stage_saak_trans: filters_26: {}".format(filters_26.shape))
    print("one_stage_saak_trans: filters_27: {}".format(filters_27.shape))
    print("one_stage_saak_trans: filters_28: {}".format(filters_28.shape))
    print("one_stage_saak_trans: filters_29: {}".format(filters_29.shape))

    # output (60000,7,14,14)
    relu_output_0 = conv_and_relu(filters_0, datasets_0, stride=2)
    relu_output_1 = conv_and_relu(filters_1, datasets_1, stride=2)
    relu_output_2 = conv_and_relu(filters_2, datasets_2, stride=2)
    relu_output_3 = conv_and_relu(filters_3, datasets_3, stride=2)
    relu_output_4 = conv_and_relu(filters_4, datasets_4, stride=2)
    relu_output_5 = conv_and_relu(filters_5, datasets_5, stride=2)
    relu_output_6 = conv_and_relu(filters_6, datasets_6, stride=2)
    relu_output_7 = conv_and_relu(filters_7, datasets_7, stride=2)
    relu_output_8 = conv_and_relu(filters_8, datasets_8, stride=2)
    relu_output_9 = conv_and_relu(filters_9, datasets_9, stride=2)
    relu_output_10 = conv_and_relu(filters_10, datasets_10, stride=2)
    relu_output_11 = conv_and_relu(filters_11, datasets_11, stride=2)
    relu_output_12 = conv_and_relu(filters_12, datasets_12, stride=2)
    relu_output_13 = conv_and_relu(filters_13, datasets_13, stride=2)
    relu_output_14 = conv_and_relu(filters_14, datasets_14, stride=2)
    relu_output_15 = conv_and_relu(filters_15, datasets_15, stride=2)
    relu_output_16 = conv_and_relu(filters_16, datasets_16, stride=2)
    relu_output_17 = conv_and_relu(filters_17, datasets_17, stride=2)
    relu_output_18 = conv_and_relu(filters_18, datasets_18, stride=2)
    relu_output_19 = conv_and_relu(filters_19, datasets_19, stride=2)
    relu_output_20 = conv_and_relu(filters_20, datasets_20, stride=2)
    relu_output_21 = conv_and_relu(filters_21, datasets_21, stride=2)
    relu_output_22 = conv_and_relu(filters_22, datasets_22, stride=2)
    relu_output_23 = conv_and_relu(filters_23, datasets_23, stride=2)
    relu_output_24 = conv_and_relu(filters_24, datasets_24, stride=2)
    relu_output_25 = conv_and_relu(filters_25, datasets_25, stride=2)
    relu_output_26 = conv_and_relu(filters_26, datasets_26, stride=2)
    relu_output_27 = conv_and_relu(filters_27, datasets_27, stride=2)
    relu_output_28 = conv_and_relu(filters_28, datasets_28, stride=2)
    relu_output_29 = conv_and_relu(filters_29, datasets_29, stride=2)

    data_0 = relu_output_0.data.numpy()
    data_1 = relu_output_1.data.numpy()
    data_2 = relu_output_2.data.numpy()
    data_3 = relu_output_3.data.numpy()
    data_4 = relu_output_4.data.numpy()
    data_5 = relu_output_5.data.numpy()
    data_6 = relu_output_6.data.numpy()
    data_7 = relu_output_7.data.numpy()
    data_8 = relu_output_8.data.numpy()
    data_9 = relu_output_9.data.numpy()
    data_10 = relu_output_10.data.numpy()
    data_11 = relu_output_11.data.numpy()
    data_12 = relu_output_12.data.numpy()
    data_13 = relu_output_13.data.numpy()
    data_14 = relu_output_14.data.numpy()
    data_15 = relu_output_15.data.numpy()
    data_16 = relu_output_16.data.numpy()
    data_17 = relu_output_17.data.numpy()
    data_18 = relu_output_18.data.numpy()
    data_19 = relu_output_19.data.numpy()
    data_20 = relu_output_20.data.numpy()
    data_21 = relu_output_21.data.numpy()
    data_22 = relu_output_22.data.numpy()
    data_23 = relu_output_23.data.numpy()
    data_24 = relu_output_24.data.numpy()
    data_25 = relu_output_25.data.numpy()
    data_26 = relu_output_26.data.numpy()
    data_27 = relu_output_27.data.numpy()
    data_28 = relu_output_28.data.numpy()
    data_29 = relu_output_29.data.numpy()
    print("one_stage_saak_trans: data_0: {}".format(data_0.shape))
    print("one_stage_saak_trans: data_1: {}".format(data_1.shape))
    print("one_stage_saak_trans: data_2: {}".format(data_2.shape))
    print("one_stage_saak_trans: data_3: {}".format(data_3.shape))
    print("one_stage_saak_trans: data_4: {}".format(data_4.shape))
    print("one_stage_saak_trans: data_5: {}".format(data_5.shape))
    print("one_stage_saak_trans: data_6: {}".format(data_6.shape))
    print("one_stage_saak_trans: data_7: {}".format(data_7.shape))
    print("one_stage_saak_trans: data_8: {}".format(data_8.shape))
    print("one_stage_saak_trans: data_9: {}".format(data_9.shape))
    print("one_stage_saak_trans: data_10: {}".format(data_10.shape))
    print("one_stage_saak_trans: data_11: {}".format(data_11.shape))
    print("one_stage_saak_trans: data_12: {}".format(data_12.shape))
    print("one_stage_saak_trans: data_13: {}".format(data_13.shape))
    print("one_stage_saak_trans: data_14: {}".format(data_14.shape))
    print("one_stage_saak_trans: data_15: {}".format(data_15.shape))
    print("one_stage_saak_trans: data_16: {}".format(data_16.shape))
    print("one_stage_saak_trans: data_17: {}".format(data_17.shape))
    print("one_stage_saak_trans: data_18: {}".format(data_18.shape))
    print("one_stage_saak_trans: data_19: {}".format(data_19.shape))
    print("one_stage_saak_trans: data_20: {}".format(data_20.shape))
    print("one_stage_saak_trans: data_21: {}".format(data_21.shape))
    print("one_stage_saak_trans: data_22: {}".format(data_22.shape))
    print("one_stage_saak_trans: data_23: {}".format(data_23.shape))
    print("one_stage_saak_trans: data_24: {}".format(data_24.shape))
    print("one_stage_saak_trans: data_25: {}".format(data_25.shape))
    print("one_stage_saak_trans: data_26: {}".format(data_26.shape))
    print("one_stage_saak_trans: data_27: {}".format(data_27.shape))
    print("one_stage_saak_trans: data_28: {}".format(data_28.shape))
    print("one_stage_saak_trans: data_29: {}".format(data_29.shape))

    data = np.empty([datasets.shape[0], data_0.shape[1], data_0.shape[2], data_0.shape[3]])
    i = [0] * 30
    m = 0
    for k in K:
        if k == 0:
            data[m, :, :, :] = data_0[i[0], :, :, :]
            m += 1
            i[0] += 1
        if k == 1:
            data[m, :, :, :] = data_1[i[1], :, :, :]
            m += 1
            i[1] += 1
        if k == 2:
            data[m, :, :, :] = data_2[i[2], :, :, :]
            m += 1
            i[2] += 1
        if k == 3:
            data[m, :, :, :] = data_3[i[3], :, :, :]
            m += 1
            i[3] += 1
        if k == 4:
            data[m, :, :, :] = data_4[i[4], :, :, :]
            m += 1
            i[4] += 1
        if k == 5:
            data[m, :, :, :] = data_5[i[5], :, :, :]
            m += 1
            i[5] += 1
        if k == 6:
            data[m, :, :, :] = data_6[i[6], :, :, :]
            m += 1
            i[6] += 1
        if k == 7:
            data[m, :, :, :] = data_7[i[7], :, :, :]
            m += 1
            i[7] += 1
        if k == 8:
            data[m, :, :, :] = data_8[i[8], :, :, :]
            m += 1
            i[8] += 1
        if k == 9:
            data[m, :, :, :] = data_9[i[9], :, :, :]
            m += 1
            i[9] += 1
        if k == 10:
            data[m, :, :, :] = data_10[i[10], :, :, :]
            m += 1
            i[10] += 1
        if k == 11:
            data[m, :, :, :] = data_11[i[11], :, :, :]
            m += 1
            i[11] += 1
        if k == 12:
            data[m, :, :, :] = data_12[i[12], :, :, :]
            m += 1
            i[12] += 1
        if k == 13:
            data[m, :, :, :] = data_13[i[13], :, :, :]
            m += 1
            i[13] += 1
        if k == 14:
            data[m, :, :, :] = data_14[i[14], :, :, :]
            m += 1
            i[14] += 1
        if k == 15:
            data[m, :, :, :] = data_15[i[15], :, :, :]
            m += 1
            i[15] += 1
        if k == 16:
            data[m, :, :, :] = data_16[i[16], :, :, :]
            m += 1
            i[16] += 1
        if k == 17:
            data[m, :, :, :] = data_17[i[17], :, :, :]
            m += 1
            i[17] += 1
        if k == 18:
            data[m, :, :, :] = data_18[i[18], :, :, :]
            m += 1
            i[18] += 1
        if k == 19:
            data[m, :, :, :] = data_19[i[19], :, :, :]
            m += 1
            i[19] += 1
        if k == 20:
            data[m, :, :, :] = data_20[i[20], :, :, :]
            m += 1
            i[20] += 1
        if k == 21:
            data[m, :, :, :] = data_21[i[21], :, :, :]
            m += 1
            i[21] += 1
        if k == 22:
            data[m, :, :, :] = data_22[i[22], :, :, :]
            m += 1
            i[22] += 1
        if k == 23:
            data[m, :, :, :] = data_23[i[23], :, :, :]
            m += 1
            i[23] += 1
        if k == 24:
            data[m, :, :, :] = data_24[i[24], :, :, :]
            m += 1
            i[24] += 1
        if k == 25:
            data[m, :, :, :] = data_25[i[25], :, :, :]
            m += 1
            i[25] += 1
        if k == 26:
            data[m, :, :, :] = data_26[i[26], :, :, :]
            m += 1
            i[26] += 1
        if k == 27:
            data[m, :, :, :] = data_27[i[27], :, :, :]
            m += 1
            i[27] += 1
        if k == 28:
            data[m, :, :, :] = data_28[i[28], :, :, :]
            m += 1
            i[28] += 1
        if k == 29:
            data[m, :, :, :] = data_29[i[29], :, :, :]
            m += 1
            i[29] += 1
    output = list(data)
    return data, output


def kmeans_cluster_test(data_in):
    data = np.reshape(data_in, (data_in.shape[0], -1))
    # count_choice = np.random.choice(np.arange(data.shape[0]), size=int(data.shape[0] * 0.01), replace=False)
    kmeans = MiniBatchKMeans(n_clusters=8).fit(data)
    K = kmeans.labels_
    zip_data = zip(data, K)
    data_0 = [dat for dat, k in zip_data if k == 0]
    data_1 = [dat for dat, k in zip_data if k == 1]
    data_2 = [dat for dat, k in zip_data if k == 2]
    data_3 = [dat for dat, k in zip_data if k == 3]
    data_4 = [dat for dat, k in zip_data if k == 4]
    data_5 = [dat for dat, k in zip_data if k == 5]
    data_6 = [dat for dat, k in zip_data if k == 6]
    data_7 = [dat for dat, k in zip_data if k == 7]
    # data_8 = [dat for dat, k in zip_data if k == 8]
    # data_9 = [dat for dat, k in zip_data if k == 9]
    # data_10 = [dat for dat, k in zip_data if k == 10]
    # data_11 = [dat for dat, k in zip_data if k == 11]
    # data_12 = [dat for dat, k in zip_data if k == 12]
    # data_13 = [dat for dat, k in zip_data if k == 13]
    # data_14 = [dat for dat, k in zip_data if k == 14]
    # data_15 = [dat for dat, k in zip_data if k == 15]
    # data_16 = [dat for dat, k in zip_data if k == 16]
    # data_17 = [dat for dat, k in zip_data if k == 17]
    # data_18 = [dat for dat, k in zip_data if k == 18]
    # data_19 = [dat for dat, k in zip_data if k == 19]
    data_0 = np.array(data_0)
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_3 = np.array(data_3)
    data_4 = np.array(data_4)
    data_5 = np.array(data_5)
    data_6 = np.array(data_6)
    data_7 = np.array(data_7)
    # data_8 = np.array(data_8)
    # data_9 = np.array(data_9)
    # data_10 = np.array(data_10)
    # data_11 = np.array(data_11)
    # data_12 = np.array(data_12)
    # data_13 = np.array(data_13)
    # data_14 = np.array(data_14)
    # data_15 = np.array(data_15)
    # data_16 = np.array(data_16)
    # data_17 = np.array(data_17)
    # data_18 = np.array(data_18)
    # data_19 = np.array(data_19)

    return data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, K


def one_stage_saak_trans_test(datasets=None, depth=0, components_pca=0):

    # intial dataset, (60000,1,32,32)
    # channel change: 1->7
    print("one_stage_saak_trans: datasets.shape {}".format(datasets.shape))
    input_channels = datasets.shape[1]

    datasets_0, datasets_1, datasets_2, datasets_3, datasets_4, datasets_5, datasets_6, datasets_7, K = kmeans_cluster_test(datasets)
    datasets_0 = np.reshape(datasets_0, (datasets_0.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_1 = np.reshape(datasets_1, (datasets_1.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_2 = np.reshape(datasets_2, (datasets_2.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_3 = np.reshape(datasets_3, (datasets_3.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_4 = np.reshape(datasets_4, (datasets_4.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_5 = np.reshape(datasets_5, (datasets_5.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_6 = np.reshape(datasets_6, (datasets_6.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    datasets_7 = np.reshape(datasets_7, (datasets_7.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_8 = np.reshape(datasets_8, (datasets_8.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_9 = np.reshape(datasets_9, (datasets_9.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_10 = np.reshape(datasets_10, (datasets_10.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_11 = np.reshape(datasets_11, (datasets_11.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_12 = np.reshape(datasets_12, (datasets_12.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_13 = np.reshape(datasets_13, (datasets_13.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_14 = np.reshape(datasets_14, (datasets_14.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_15 = np.reshape(datasets_15, (datasets_15.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_16 = np.reshape(datasets_16, (datasets_16.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_17 = np.reshape(datasets_17, (datasets_17.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_18 = np.reshape(datasets_18, (datasets_18.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))
    # datasets_19 = np.reshape(datasets_19, (datasets_19.shape[0], datasets.shape[1], datasets.shape[2], datasets.shape[3]))

    # change data shape, (14*60000,4)
    data_flatten_0 = fit_pca_shape(datasets_0, depth)
    data_flatten_1 = fit_pca_shape(datasets_1, depth)
    data_flatten_2 = fit_pca_shape(datasets_2, depth)
    data_flatten_3 = fit_pca_shape(datasets_3, depth)
    data_flatten_4 = fit_pca_shape(datasets_4, depth)
    data_flatten_5 = fit_pca_shape(datasets_5, depth)
    data_flatten_6 = fit_pca_shape(datasets_6, depth)
    data_flatten_7 = fit_pca_shape(datasets_7, depth)
    # data_flatten_8 = fit_pca_shape(datasets_8, depth)
    # data_flatten_9 = fit_pca_shape(datasets_9, depth)
    # data_flatten_10 = fit_pca_shape(datasets_10, depth)
    # data_flatten_11 = fit_pca_shape(datasets_11, depth)
    # data_flatten_12 = fit_pca_shape(datasets_12, depth)
    # data_flatten_13 = fit_pca_shape(datasets_13, depth)
    # data_flatten_14 = fit_pca_shape(datasets_14, depth)
    # data_flatten_15 = fit_pca_shape(datasets_15, depth)
    # data_flatten_16 = fit_pca_shape(datasets_16, depth)
    # data_flatten_17 = fit_pca_shape(datasets_17, depth)
    # data_flatten_18 = fit_pca_shape(datasets_18, depth)
    # data_flatten_19 = fit_pca_shape(datasets_19, depth)

    # augmented components, first round: (7,4), only augment AC components
    comps_complete_0 = PCA_and_augment(data_flatten_0)
    comps_complete_1 = PCA_and_augment(data_flatten_1)
    comps_complete_2 = PCA_and_augment(data_flatten_2)
    comps_complete_3 = PCA_and_augment(data_flatten_3)
    comps_complete_4 = PCA_and_augment(data_flatten_4)
    comps_complete_5 = PCA_and_augment(data_flatten_5)
    comps_complete_6 = PCA_and_augment(data_flatten_6)
    comps_complete_7 = PCA_and_augment(data_flatten_7)
    # comps_complete_8 = PCA_and_augment(data_flatten_8)
    # comps_complete_9 = PCA_and_augment(data_flatten_9)
    # comps_complete_10 = PCA_and_augment(data_flatten_10)
    # comps_complete_11 = PCA_and_augment(data_flatten_11)
    # comps_complete_12 = PCA_and_augment(data_flatten_12)
    # comps_complete_13 = PCA_and_augment(data_flatten_13)
    # comps_complete_14 = PCA_and_augment(data_flatten_14)
    # comps_complete_15 = PCA_and_augment(data_flatten_15)
    # comps_complete_16 = PCA_and_augment(data_flatten_16)
    # comps_complete_17 = PCA_and_augment(data_flatten_17)
    # comps_complete_18 = PCA_and_augment(data_flatten_18)
    # comps_complete_19 = PCA_and_augment(data_flatten_19)

    comps_complete_0 = comps_complete_0[:components_pca + 1, :]
    comps_complete_1 = comps_complete_1[:components_pca + 1, :]
    comps_complete_2 = comps_complete_2[:components_pca + 1, :]
    comps_complete_3 = comps_complete_3[:components_pca + 1, :]
    comps_complete_4 = comps_complete_4[:components_pca + 1, :]
    comps_complete_5 = comps_complete_5[:components_pca + 1, :]
    comps_complete_6 = comps_complete_6[:components_pca + 1, :]
    comps_complete_7 = comps_complete_7[:components_pca + 1, :]
    # comps_complete_8 = comps_complete_8[:components_pca + 1, :]
    # comps_complete_9 = comps_complete_9[:components_pca + 1, :]
    # comps_complete_10 = comps_complete_10[:components_pca + 1, :]
    # comps_complete_11 = comps_complete_11[:components_pca + 1, :]
    # comps_complete_12 = comps_complete_12[:components_pca + 1, :]
    # comps_complete_13 = comps_complete_13[:components_pca + 1, :]
    # comps_complete_14 = comps_complete_14[:components_pca + 1, :]
    # comps_complete_15 = comps_complete_15[:components_pca + 1, :]
    # comps_complete_16 = comps_complete_16[:components_pca + 1, :]
    # comps_complete_17 = comps_complete_17[:components_pca + 1, :]
    # comps_complete_18 = comps_complete_18[:components_pca + 1, :]
    # comps_complete_19 = comps_complete_19[:components_pca + 1, :]

    print("one_stage_saak_trans: comps_complete_0: {}".format(comps_complete_0.shape))
    print("one_stage_saak_trans: comps_complete_1: {}".format(comps_complete_1.shape))
    print("one_stage_saak_trans: comps_complete_2: {}".format(comps_complete_2.shape))
    print("one_stage_saak_trans: comps_complete_3: {}".format(comps_complete_3.shape))
    print("one_stage_saak_trans: comps_complete_4: {}".format(comps_complete_4.shape))
    print("one_stage_saak_trans: comps_complete_5: {}".format(comps_complete_5.shape))
    print("one_stage_saak_trans: comps_complete_6: {}".format(comps_complete_6.shape))
    print("one_stage_saak_trans: comps_complete_7: {}".format(comps_complete_7.shape))
    # print("one_stage_saak_trans: comps_complete_8: {}".format(comps_complete_8.shape))
    # print("one_stage_saak_trans: comps_complete_9: {}".format(comps_complete_9.shape))
    # print("one_stage_saak_trans: comps_complete_10: {}".format(comps_complete_10.shape))
    # print("one_stage_saak_trans: comps_complete_11: {}".format(comps_complete_11.shape))
    # print("one_stage_saak_trans: comps_complete_12: {}".format(comps_complete_12.shape))
    # print("one_stage_saak_trans: comps_complete_13: {}".format(comps_complete_13.shape))
    # print("one_stage_saak_trans: comps_complete_14: {}".format(comps_complete_14.shape))
    # print("one_stage_saak_trans: comps_complete_15: {}".format(comps_complete_15.shape))
    # print("one_stage_saak_trans: comps_complete_16: {}".format(comps_complete_16.shape))
    # print("one_stage_saak_trans: comps_complete_17: {}".format(comps_complete_17.shape))
    # print("one_stage_saak_trans: comps_complete_18: {}".format(comps_complete_18.shape))
    # print("one_stage_saak_trans: comps_complete_19: {}".format(comps_complete_19.shape))

    # get filter, (7,1,2,2)
    filters_0 = ret_filt_patches(comps_complete_0, input_channels)
    filters_1 = ret_filt_patches(comps_complete_1, input_channels)
    filters_2 = ret_filt_patches(comps_complete_2, input_channels)
    filters_3 = ret_filt_patches(comps_complete_3, input_channels)
    filters_4 = ret_filt_patches(comps_complete_4, input_channels)
    filters_5 = ret_filt_patches(comps_complete_5, input_channels)
    filters_6 = ret_filt_patches(comps_complete_6, input_channels)
    filters_7 = ret_filt_patches(comps_complete_7, input_channels)
    # filters_8 = ret_filt_patches(comps_complete_8, input_channels)
    # filters_9 = ret_filt_patches(comps_complete_9, input_channels)
    # filters_10 = ret_filt_patches(comps_complete_10, input_channels)
    # filters_11 = ret_filt_patches(comps_complete_11, input_channels)
    # filters_12 = ret_filt_patches(comps_complete_12, input_channels)
    # filters_13 = ret_filt_patches(comps_complete_13, input_channels)
    # filters_14 = ret_filt_patches(comps_complete_14, input_channels)
    # filters_15 = ret_filt_patches(comps_complete_15, input_channels)
    # filters_16 = ret_filt_patches(comps_complete_16, input_channels)
    # filters_17 = ret_filt_patches(comps_complete_17, input_channels)
    # filters_18 = ret_filt_patches(comps_complete_18, input_channels)
    # filters_19 = ret_filt_patches(comps_complete_19, input_channels)
    print("one_stage_saak_trans: filters_0: {}".format(filters_0.shape))
    print("one_stage_saak_trans: filters_1: {}".format(filters_1.shape))
    print("one_stage_saak_trans: filters_2: {}".format(filters_2.shape))
    print("one_stage_saak_trans: filters_3: {}".format(filters_3.shape))
    print("one_stage_saak_trans: filters_4: {}".format(filters_4.shape))
    print("one_stage_saak_trans: filters_5: {}".format(filters_5.shape))
    print("one_stage_saak_trans: filters_6: {}".format(filters_6.shape))
    print("one_stage_saak_trans: filters_7: {}".format(filters_7.shape))
    # print("one_stage_saak_trans: filters_8: {}".format(filters_8.shape))
    # print("one_stage_saak_trans: filters_9: {}".format(filters_9.shape))
    # print("one_stage_saak_trans: filters_10: {}".format(filters_10.shape))
    # print("one_stage_saak_trans: filters_11: {}".format(filters_11.shape))
    # print("one_stage_saak_trans: filters_12: {}".format(filters_12.shape))
    # print("one_stage_saak_trans: filters_13: {}".format(filters_13.shape))
    # print("one_stage_saak_trans: filters_14: {}".format(filters_14.shape))
    # print("one_stage_saak_trans: filters_15: {}".format(filters_15.shape))
    # print("one_stage_saak_trans: filters_16: {}".format(filters_16.shape))
    # print("one_stage_saak_trans: filters_17: {}".format(filters_17.shape))
    # print("one_stage_saak_trans: filters_18: {}".format(filters_18.shape))
    # print("one_stage_saak_trans: filters_19: {}".format(filters_19.shape))

    # output (60000,7,14,14)
    relu_output_0 = conv_and_relu(filters_0, datasets_0, stride=2)
    relu_output_1 = conv_and_relu(filters_1, datasets_1, stride=2)
    relu_output_2 = conv_and_relu(filters_2, datasets_2, stride=2)
    relu_output_3 = conv_and_relu(filters_3, datasets_3, stride=2)
    relu_output_4 = conv_and_relu(filters_4, datasets_4, stride=2)
    relu_output_5 = conv_and_relu(filters_5, datasets_5, stride=2)
    relu_output_6 = conv_and_relu(filters_6, datasets_6, stride=2)
    relu_output_7 = conv_and_relu(filters_7, datasets_7, stride=2)
    # relu_output_8 = conv_and_relu(filters_8, datasets_8, stride=2)
    # relu_output_9 = conv_and_relu(filters_9, datasets_9, stride=2)
    # relu_output_10 = conv_and_relu(filters_10, datasets_10, stride=2)
    # relu_output_11 = conv_and_relu(filters_11, datasets_11, stride=2)
    # relu_output_12 = conv_and_relu(filters_12, datasets_12, stride=2)
    # relu_output_13 = conv_and_relu(filters_13, datasets_13, stride=2)
    # relu_output_14 = conv_and_relu(filters_14, datasets_14, stride=2)
    # relu_output_15 = conv_and_relu(filters_15, datasets_15, stride=2)
    # relu_output_16 = conv_and_relu(filters_16, datasets_16, stride=2)
    # relu_output_17 = conv_and_relu(filters_17, datasets_17, stride=2)
    # relu_output_18 = conv_and_relu(filters_18, datasets_18, stride=2)
    # relu_output_19 = conv_and_relu(filters_19, datasets_19, stride=2)

    data_0 = relu_output_0.data.numpy()
    data_1 = relu_output_1.data.numpy()
    data_2 = relu_output_2.data.numpy()
    data_3 = relu_output_3.data.numpy()
    data_4 = relu_output_4.data.numpy()
    data_5 = relu_output_5.data.numpy()
    data_6 = relu_output_6.data.numpy()
    data_7 = relu_output_7.data.numpy()
    # data_8 = relu_output_8.data.numpy()
    # data_9 = relu_output_9.data.numpy()
    # data_10 = relu_output_10.data.numpy()
    # data_11 = relu_output_11.data.numpy()
    # data_12 = relu_output_12.data.numpy()
    # data_13 = relu_output_13.data.numpy()
    # data_14 = relu_output_14.data.numpy()
    # data_15 = relu_output_15.data.numpy()
    # data_16 = relu_output_16.data.numpy()
    # data_17 = relu_output_17.data.numpy()
    # data_18 = relu_output_18.data.numpy()
    # data_19 = relu_output_19.data.numpy()
    print("one_stage_saak_trans: data_0: {}".format(data_0.shape))
    print("one_stage_saak_trans: data_1: {}".format(data_1.shape))
    print("one_stage_saak_trans: data_2: {}".format(data_2.shape))
    print("one_stage_saak_trans: data_3: {}".format(data_3.shape))
    print("one_stage_saak_trans: data_4: {}".format(data_4.shape))
    print("one_stage_saak_trans: data_5: {}".format(data_5.shape))
    print("one_stage_saak_trans: data_6: {}".format(data_6.shape))
    print("one_stage_saak_trans: data_7: {}".format(data_7.shape))
    # print("one_stage_saak_trans: data_8: {}".format(data_8.shape))
    # print("one_stage_saak_trans: data_9: {}".format(data_9.shape))
    # print("one_stage_saak_trans: data_10: {}".format(data_10.shape))
    # print("one_stage_saak_trans: data_11: {}".format(data_11.shape))
    # print("one_stage_saak_trans: data_12: {}".format(data_12.shape))
    # print("one_stage_saak_trans: data_13: {}".format(data_13.shape))
    # print("one_stage_saak_trans: data_14: {}".format(data_14.shape))
    # print("one_stage_saak_trans: data_15: {}".format(data_15.shape))
    # print("one_stage_saak_trans: data_16: {}".format(data_16.shape))
    # print("one_stage_saak_trans: data_17: {}".format(data_17.shape))
    # print("one_stage_saak_trans: data_18: {}".format(data_18.shape))
    # print("one_stage_saak_trans: data_19: {}".format(data_19.shape))

    data = np.empty([datasets.shape[0], data_0.shape[1], data_0.shape[2], data_0.shape[3]])
    i = [0] * 8
    m = 0
    for k in K:
        if k == 0:
            data[m, :, :, :] = data_0[i[0], :, :, :]
            m += 1
            i[0] += 1
        if k == 1:
            data[m, :, :, :] = data_1[i[1], :, :, :]
            m += 1
            i[1] += 1
        if k == 2:
            data[m, :, :, :] = data_2[i[2], :, :, :]
            m += 1
            i[2] += 1
        if k == 3:
            data[m, :, :, :] = data_3[i[3], :, :, :]
            m += 1
            i[3] += 1
        if k == 4:
            data[m, :, :, :] = data_4[i[4], :, :, :]
            m += 1
            i[4] += 1
        if k == 5:
            data[m, :, :, :] = data_5[i[5], :, :, :]
            m += 1
            i[5] += 1
        if k == 6:
            data[m, :, :, :] = data_6[i[6], :, :, :]
            m += 1
            i[6] += 1
        if k == 7:
            data[m, :, :, :] = data_7[i[7], :, :, :]
            m += 1
            i[7] += 1
        # if k == 8:
        #     data[m, :, :, :] = data_8[i[8], :, :, :]
        #     m += 1
        #     i[8] += 1
        # if k == 9:
        #     data[m, :, :, :] = data_9[i[9], :, :, :]
        #     m += 1
        #     i[9] += 1
        # if k == 10:
        #     data[m, :, :, :] = data_10[i[10], :, :, :]
        #     m += 1
        #     i[10] += 1
        # if k == 11:
        #     data[m, :, :, :] = data_11[i[11], :, :, :]
        #     m += 1
        #     i[11] += 1
        # if k == 12:
        #     data[m, :, :, :] = data_12[i[12], :, :, :]
        #     m += 1
        #     i[12] += 1
        # if k == 13:
        #     data[m, :, :, :] = data_13[i[13], :, :, :]
        #     m += 1
        #     i[13] += 1
        # if k == 14:
        #     data[m, :, :, :] = data_14[i[14], :, :, :]
        #     m += 1
        #     i[14] += 1
        # if k == 15:
        #     data[m, :, :, :] = data_15[i[15], :, :, :]
        #     m += 1
        #     i[15] += 1
        # if k == 16:
        #     data[m, :, :, :] = data_16[i[16], :, :, :]
        #     m += 1
        #     i[16] += 1
        # if k == 17:
        #     data[m, :, :, :] = data_17[i[17], :, :, :]
        #     m += 1
        #     i[17] += 1
        # if k == 18:
        #     data[m, :, :, :] = data_18[i[18], :, :, :]
        #     m += 1
        #     i[18] += 1
        # if k == 19:
        #     data[m, :, :, :] = data_19[i[19], :, :, :]
        #     m += 1
        #     i[19] += 1
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
        train_data, train_output = one_stage_saak_trans_train(train_data, depth=i, components_pca=components_pca[i])
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

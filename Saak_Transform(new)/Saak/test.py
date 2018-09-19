import matplotlib.pyplot as plt
import numpy as np
import cPickle
import gzip
from skimage.util.shape import view_as_windows


def window_process(train, test):

    train_shape = train.shape
    test_shape = test.shape

    train_cnt = train_shape[0]
    test_cnt = test_shape[0]
    w, h, d = train_shape[1], train_shape[2], train_shape[3]

    train_window = view_as_windows(train, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(train_cnt * w / 2 * h / 2, -1)
    test_window = view_as_windows(test, (1, 2, 2, d), step=(1, 2, 2, d)).reshape(test_cnt * w / 2 * h / 2, -1)
    print("train_window.shape: {}".format(train_window.shape))
    print("test_window.shape: {}".format(test_window.shape))

    return train_window, test_window


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


# retest = np.load("./retest.npy")
# retest = retest[:10]
# for i in range(retest.shape[0]):
#     plt.figure(i)
#     plt.imshow(test[retest[i], :, :, 0], cmap='Greys')
#     plt.title("This is {}".format(test_label[retest[i]]))
#     plt.show()

# train_coefficients = np.load("./train_coefficients.npy")
# train_label = np.load("./train_label.npy")
# zip_coefficients = zip(train_coefficients, train_label)
# dataset = []
# for i in range(10):
#     dataset.append(np.array([dat for dat, k in zip_coefficients if k == i]))

# image_n = np.empty([10, dataset[0].shape[1], dataset[0].shape[2]])
# for k in range(10):
#     for i in range(dataset[k].shape[1]):
#         for j in range(dataset[k].shape[2]):
#             image_n[k, i, j] = np.mean(dataset[k][:, i, j, 0])

# plt.imshow(image_n[3], cmap='Greys')
# plt.colorbar()
# plt.show()

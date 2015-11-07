#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from . import utils
from .dA import dA


N_MNIST = 70000
N_MNIST_TRAIN = 60000
N_MNIST_TEST = N_MNIST - N_MNIST_TRAIN
N_TRAIN = N_MNIST_TRAIN
N_TEST = 100
N_EPOCH = 20
PLOT_N_CLOUMNS = 10

# prepare MNIST data
X, y = utils.prepare_data()
# split data into train and test one
perm = np.random.permutation(N_MNIST_TRAIN)
X_train, y_train = X[perm[: N_TRAIN]], y[perm[: N_TRAIN]]
perm = np.random.permutation(N_MNIST_TEST)
X_test, y_test = X[perm[: N_TEST] + N_MNIST_TRAIN], y[perm[: N_TEST] + N_MNIST_TRAIN]

# train
da = dA(784, 500, n_epoch=N_EPOCH, verbose=True)
da.fit(X_train)

# test = show output
pred = da.predict(X_test)
n_row = (N_TEST // PLOT_N_CLOUMNS + (1 if N_TEST % PLOT_N_CLOUMNS > 0 else 0)) * 2
# plt.figure(figsize=(15, 30))
plt.figure().suptitle('Output of Second Layer')
for i in range(N_TEST):
    fig = plt.subplot(n_row, PLOT_N_CLOUMNS, i % PLOT_N_CLOUMNS + i // PLOT_N_CLOUMNS * 2 * PLOT_N_CLOUMNS + 1)
    utils.imshow(fig, X_test[i])
    fig = plt.subplot(n_row, PLOT_N_CLOUMNS, i % PLOT_N_CLOUMNS + (i // PLOT_N_CLOUMNS * 2 + 1) * PLOT_N_CLOUMNS + 1)
    utils.imshow(fig, pred[i])
plt.show()

# show weights of first layer
n_row = N_TEST // PLOT_N_CLOUMNS + (1 if N_TEST % PLOT_N_CLOUMNS > 0 else 0)
plt.figure().suptitle('Weights of First Layer')
for i in range(N_TEST):
    fig = plt.subplot(n_row, PLOT_N_CLOUMNS, i % PLOT_N_CLOUMNS + i // PLOT_N_CLOUMNS * PLOT_N_CLOUMNS + 1)
    utils.imshow(fig, da.encoder.W[i])
plt.show()

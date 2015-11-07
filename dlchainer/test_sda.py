#-*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from . import utils
from .SdA import SdAClassifier


N_MNIST = 70000
N_MNIST_TRAIN = 60000
N_MNIST_TEST = N_MNIST - N_MNIST_TRAIN
N_TRAIN = N_MNIST_TRAIN
N_TEST = N_MNIST_TEST

# prepare MNIST data
X, y = utils.prepare_data()
# split data into train and test one
perm = np.random.permutation(N_MNIST_TRAIN)
X_train, y_train = X[perm[: N_TRAIN]], y[perm[: N_TRAIN]]
perm = np.random.permutation(N_MNIST_TEST)
X_test, y_test = X[perm[: N_TEST] + N_MNIST_TRAIN], y[perm[: N_TEST] + N_MNIST_TRAIN]

# train
sda = SdAClassifier(784, [500, 500], 10, [0.3, 0.3], [0.5, 0.5], do_pretrain=True, verbose=True)
sda.fit(X_train, y_train)

# test
pred = sda.predict(X_test)

cm = confusion_matrix(y_test, pred)
print(cm)
print('accuracy: {}'.format(accuracy_score(y_test, pred)))

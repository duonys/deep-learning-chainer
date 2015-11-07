#-*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator
from chainer import Variable, FunctionSet, optimizers, cuda
import chainer.functions as F
from . import utils


class dA(BaseEstimator):
    """
    Denoising Autoencoder

    reference:
    http://deeplearning.net/tutorial/dA.html
    https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py
    """
    def __init__(self, n_visible, n_hidden, noise_level=0.0, dropout_ratio=0.3,
                 batch_size=100, n_epoch=20, optimizer=optimizers.Adam(),
                 activation_func=F.relu, verbose=False, gpu=-1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.noise_level = noise_level
        self.dropout_ratio = dropout_ratio
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        # construct model and setup optimizer
        self.model = FunctionSet(
            encoder=F.Linear(n_visible, n_hidden),
            decoder=F.Linear(n_hidden, n_visible)
        )
        self.optimizer = optimizer
        self.optimizer.setup(self.model)
        self.activation_func = activation_func
        self.verbose = verbose
        # set gpu
        self.gpu = gpu
        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.model.to_gpu()


    def fit(self, X):
        xp = cuda.cupy if self.gpu >= 0 else np
        for epoch in range(self.n_epoch):
            utils.disp('epoch: {}'.format(epoch + 1), self.verbose)

            perm = np.random.permutation(len(X))
            sum_loss = 0
            for i in range(0, len(X), self.batch_size):
                X_batch = xp.asarray(X[perm[i: i + self.batch_size]])
                loss = self._fit(X_batch)
                sum_loss += float(loss.data) * len(X_batch)
            utils.disp('train mean loss={}'.format(sum_loss / len(X)), self.verbose)
        return self


    def _fit(self, X):
        self.optimizer.zero_grads()
        y_var = self._forward(X, train=True)
        loss = F.mean_squared_error(y_var, Variable(X.copy()))
        loss.backward()
        self.optimizer.update()
        return loss


    def _forward(self, X, train):
        X_var = Variable(dA.add_noise(X, self.noise_level, train))
        h1 = self.encode(X_var, train)
        y_var = self.model.decoder(h1)
        return y_var


    def predict(self, X):
        return self._forward(X, train=False).data


    @property
    def encoder(self):
        return self.model.encoder


    def encode(self, X_var, train):
        return F.dropout(self.activation_func(self.encoder(X_var)), ratio=self.dropout_ratio, train=train)


    @staticmethod
    def add_noise(X, noise_level, train):
        return (np.random.binomial(size=X.shape, n=1, p=1 - (noise_level if train else 0.0)) * X).astype(np.float32)



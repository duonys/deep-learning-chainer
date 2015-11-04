#-*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.externals.six import with_metaclass
from chainer import Variable, FunctionSet, optimizers, cuda
import chainer.functions as F
from dA import dA
import utils


class SdAMixin(with_metaclass(ABCMeta, BaseEstimator)):
    """
    Stacked Denoising Autoencoder

    References:
    http://deeplearning.net/tutorial/SdA.html
    https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py
    """
    def __init__(self, n_input, n_hiddens, n_output, noise_levels=None, dropout_ratios=None, do_pretrain=True,
                 batch_size=100, n_epoch_pretrain=20, n_epoch_finetune=20, optimizer=optimizers.Adam(),
                 activation_func=F.relu, verbose=False, gpu=-1):
        self.n_input = n_input
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.do_pretrain = do_pretrain
        self.batch_size = batch_size
        self.n_epoch_pretrain = n_epoch_pretrain
        self.n_epoch_finetune = n_epoch_finetune
        self.optimizer = optimizer
        self.dAs = \
            [dA(self.n_input, self.n_hiddens[0],
                self._check_var(noise_levels, 0), self._check_var(dropout_ratios, 0), self.batch_size,
                self.n_epoch_pretrain, copy.deepcopy(optimizer),
                activation_func, verbose, gpu)] + \
            [dA(self.n_hiddens[i], self.n_hiddens[i + 1],
                self._check_var(noise_levels, i + 1), self._check_var(dropout_ratios, i + 1), self.batch_size,
                self.n_epoch_pretrain, copy.deepcopy(optimizer),
                activation_func, verbose, gpu) for i in range(len(n_hiddens) - 1)]
        self.verbose = verbose
        # setup gpu
        self.gpu = gpu
        if self.gpu >= 0:
            cuda.check_cuda_available()
        self.xp = cuda.cupy if self.gpu >= 0 else np


    def _check_var(self, var, index, default_val=0.0):
        return var[index] if var is not None else default_val


    def fit(self, X, y):
        if self.do_pretrain:
            self._pretrain(X)
        self._finetune(X, y)


    def _pretrain(self, X):
        for layer, dA in enumerate(self.dAs):
            utils.disp('*** pretrain layer: {} ***'.format(layer + 1), self.verbose)
            if layer == 0:
                layer_input = X
            else:
                layer_input = self.dAs[layer - 1].encode(Variable(layer_input), train=False).data
            dA.fit(layer_input)


    def _finetune(self, X, y):
        utils.disp('*** finetune ***', self.verbose)
        # construct model and setup optimizer
        params = {'l{}'.format(layer + 1): dA.encoder for layer, dA in enumerate(self.dAs)}
        params.update({'l{}'.format(len(self.dAs) + 1): F.Linear(self.dAs[-1].n_hidden, self.n_output)})
        self.model = FunctionSet(**params)
        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.model.to_gpu()
        self.optimizer.setup(self.model)

        n = len(X)
        for epoch in range(self.n_epoch_finetune):
            utils.disp('epoch: {}'.format(epoch + 1), self.verbose)

            perm = np.random.permutation(n)
            sum_loss = 0
            for i in range(0, n, self.batch_size):
                X_batch = self.xp.asarray(X[perm[i: i + self.batch_size]])
                y_batch = self.xp.asarray(y[perm[i: i + self.batch_size]])

                self.optimizer.zero_grads()
                y_var = self._forward(X_batch)
                loss = self._loss_func(y_var, Variable(y_batch))
                loss.backward()
                self.optimizer.update()

                sum_loss += float(loss.data) * len(X_batch)

            utils.disp('fine tune mean loss={}'.format(sum_loss / n), self.verbose)


    def _forward(self, X, train=True):
        X_var = Variable(X)
        output = X_var
        for dA in self.dAs:
            output = dA.encode(output, train)
        y_var = self.model['l{}'.format(len(self.dAs) + 1)](output)
        return y_var


    @abstractmethod
    def _loss_func(self, y_var, t_var):
        pass


class SdAClassifier(SdAMixin, ClassifierMixin):
    """

    References:
    http://scikit-learn.org/stable/developers/#rolling-your-own-estimator
    """
    def _loss_func(self, y_var, t_var):
        return F.softmax_cross_entropy(y_var, t_var)


    def fit(self, X, y):
        assert X.dtype == np.float32 and y.dtype == np.int32
        super().fit(X, y)


    def transform(self, X):
        return self._forward(X, train=False).data


    def predict(self, X):
        return np.apply_along_axis(lambda x: np.argmax(x), arr=self.transform(X), axis=1)


class SdARegressor(SdAMixin, RegressorMixin):
    """

    References:
    http://scikit-learn.org/stable/developers/#rolling-your-own-estimator
    """
    def _loss_func(self, y_var, t_var):
        y_var = F.reshape(y_var, [len(y_var)])
        return F.mean_squared_error(y_var, t_var)


    def fit(self, X, y):
        assert X.dtype == np.float32 and y.dtype == np.float32
        super().fit(X, y)


    def transform(self, X):
        return self._forward(X, train=False).data


    def predict(self, X):
        return self.transform(X)
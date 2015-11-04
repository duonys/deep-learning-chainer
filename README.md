# deep-learning-chainer
Deep Learning(SdA Classifier, Regressor), compatible with scikit-learn

## Requirements

* python3
* [chainer](https://github.com/pfnet/chainer)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* numpy

## Options

* matplotlib - If you want to run the following test script.

## Run
### Test Denoising Autoencoder(dA)
This script trains two layers perceptron by using MNIST original data.
You could see weights of hidden layer and output images.

```
$ python3 test_da.py
```


### Test Stacked Denoising Autoencoder(SdA) + Softmax
This script trains SdA with MNIST original train data and predict test one.

```
$ python3 test_sda.py
```

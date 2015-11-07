#-*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt


# fetch or load MNIST data
def prepare_data():
    mnist = fetch_mldata('MNIST original')
    mnist.data = mnist.data.astype(np.float32) / 255
    mnist.target = mnist.target.astype(np.int32)
    return mnist.data, mnist.target


def disp(message, verbose):
    if verbose:
        print(message)


def imshow(fig, data, width=28, height=28):
    fig.tick_params(bottom=False, top=False, left=False, right=False, labelbottom='off', labelleft='off')
    fig.imshow(data.reshape(width, height), cmap=plt.get_cmap('gray'), interpolation='nearest')
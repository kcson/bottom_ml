# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from ch08.deep_convnet import DeepConvNet
from trainer import Trainer

batch_size = 500
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
x_test, t_test = x_test[:batch_size], t_test[:batch_size]

# print(x_test)
# print(t_test)

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")
x = network.predict(x_test, train_flag=False)
x = np.argmax(x, axis=1)
r = np.sum(t_test == x)/batch_size*100
print(r)

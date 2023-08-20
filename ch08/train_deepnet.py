# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from dataset.mnist import load_local_images
from ch08.deep_convnet import DeepConvNet
from trainer import Trainer

# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

(x_train, t_train), (x_test, t_test) = load_local_images()
print(x_train.shape)
print(t_train.shape)

network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=6, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 매개변수 보관
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")

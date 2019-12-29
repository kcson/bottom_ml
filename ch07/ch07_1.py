import numpy as np
from util import im2col

# W = np.random.rand(2, 3, 5, 5)
# x1 = np.random.rand(1, 3, 7, 7)
# col1 = im2col(x1, 5, 5, stride=1, pad=0)
# col_W = W.reshape(2, -1).T
# print(col1.shape)
# print(col_W.shape)
# out = np.dot(col1, col_W)
# print(out.shape)
# out = out.reshape(1, 3, 3, -1)
# print(out.shape)
# dout = out.transpose(0, 2, 3, 1).reshape(-1, 2)
# print(dout.shape)

# col = im2col(x, FH, FW, self.stride, self.pad)
# col_W = self.W.reshape(FN, -1).T
# out = np.dot(col, col_W) + self.b
#
# out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

dout = np.random.rand(1, 2, 2, 3)
dmax = np.random.rand(12, 4)
print(dmax)
print(dout.shape)
dmax = dmax.reshape(dout.shape + (4,))
print(dmax.shape)
dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
print(dcol.shape)
print(dcol)

x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
print(x.reshape(2, -1))

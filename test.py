import numpy as np
from util import im2col

x = np.array([[
    [
        [1, 2, 3, 0],
        [0, 1, 2, 4],
        [1, 0, 4, 2],
        [3, 2, 0, 1]
    ],
    [
        [3, 0, 6, 5],
        [4, 2, 4, 3],
        [3, 0, 1, 0],
        [2, 3, 3, 1]
    ],
    [
        [4, 2, 1, 2],
        [0, 1, 0, 4],
        [3, 0, 6, 2],
        [4, 2, 4, 5]
    ]
]])

col = im2col(x, 2, 2, 2, 0)
print(col)
col = col.reshape(-1, 4)
print(col)
out = np.max(col, axis=1)
print(out)
out = out.reshape(1, 2, 2, 3)
print(out)
out = out.transpose(0, 3, 1, 2)
print(out)

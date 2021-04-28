import numpy as np

a = np.array([
    [[1, 2, 3],
     [4, 5, 6]],
    [[7, 8, 9],
     [10, 11, 12]]
])

print(a.shape)
print(a.reshape(1, -1).T)
print(a.reshape(-1, 1))

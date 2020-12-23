import random
import numpy as np
from util import im2col
from matplotlib import pyplot as plt

linker1_x = []
linker1_y = []
linker2_x = []
linker2_y = []
num = 50
for i in range(0, num):
    x = random.randint(0, num)
    y = random.randint(0, num)
    plt.text(x+0.1, y+0.1, "{}".format(i + 1))
    linker1_x.append(x)
    linker1_y.append(y)
    # linker2_x.append(random.randint(0, num))
    # linker2_y.append(random.randint(0, num))

plt.plot(linker1_x, linker1_y, 'o--')
# plt.plot(linker2_x, linker2_y, '^--')
plt.show()

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

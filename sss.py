import random
import numpy as np
import copy
from matplotlib import pyplot as plt

# %% md ## Таблица #%%
Table = [(21000, 50000), (50000, 84000), (786000, 987000), (97000, 156000), (125000, 239000), (1008000, 1383000),
         (902000, 115800), (462000, 96000), (465000, 790000), (545000, 908000), (311000, 396000), (675000, 984000),
         (128000, 189000), (105000, 103000), (215000, 285000), (31000, 70000), (42000, 90000), (78000, 73000), ]
border = int(len(Table) * 0.7)
test = Table[border:]
train = Table[:border]


def MD(x, T=train):
    s = 0.0
    for el in T:
        s += abs(x[0] * el[0] ** x[1] - el[1])
    return s


def generate(N, lb=0.0, rb=10.0):
    r = []
    for i in range(N):
        r.append((random.uniform(lb, rb), random.uniform(lb, rb)))
    return r


def ra(N, T, c1=0.1, c2=0.2):
    Y = np.array(generate(N))
    X = Y.copy()
    ymax = copy.copy(Y[0])
    v = [0] * N
    x0 = []
    x1 = []
    for i in range(T):
        x0.append(i)
        x1.append(MD(ymax))
        for j in range(N):
            if MD(X[j]) < MD(Y[j]):
                Y[j] = X[j]
            if MD(Y[j]) < MD(ymax):
                ymax = Y[j]
        for j in range(N):
            v[j] = v[j] + c1 * random.uniform(0.0, 1.0) * (Y[j] - X[j]) + c2 * random.uniform(0.0, 1.0) * (
                    ymax - X[j])
        X[j] = X[j] + v[j]
    x0.append(T + 1)
    x1.append(MD(ymax))
    return [ymax, x0, x1]


maximum, x0, x1 = ra(200, 1000)
print(maximum, MD(maximum, test))
plt.plot(x0[1:], x1[1:])
plt.show()
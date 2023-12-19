import numpy as np
import re

from model.perceptron import Perceptron
from model.mlp import *
from model.math import Math

def main():
    g = Math.get('tanh')['g']
    d = Math.get('tanh')['d']

    dataset = []
    with open('data/5_9/5_9_pmc_proj_pratico_2.csv') as f:
        lines = f.readlines()
        first = True
        for line in lines:
            if (not first):
                sentence = re.sub(r"\s+", "", line, flags=re.UNICODE)
                dataset.append([float(i) for i in sentence.split(';')])
            first = False
    dataset = np.array(dataset)

    mlp = MLP(layers=[
        Layer(n_j=15, n_i=4, g=g, gd=d),
        Layer(n_j=3, n_i=15, g=g, gd=d)])

    mlp.fit(X_train=dataset[:, :-3], D_train=dataset[:, -3:],
            precision=0.0001, verbose=True)

# Rodar com somente 1 época!!!
def _main():
    g = Math.get('logi')['g']
    d = Math.get('logi')['d']

    mlp = MLP(layers=[
        Layer(n_j=3, n_i=2, g=g, gd=d, W=np.array([
            [0.2, -0.1, 0.3],
            [-0.5, -0.8, 0.4],
            [-0.1, 0.1, -0.6]
        ])),
        Layer(n_j=2, n_i=3, g=lambda x: x, gd= lambda x: 1, W=np.array([
            [0.1, 0.5, -0.3, 0.1],
            [0.3, 0.2, 0.6, -0.9],
        ]))])

    mlp.fit(X_train=np.array([[2, 1]]), D_train=np.array([[
            1, 1]]), precision=0.0001, verbose=True)


def _main():
    rna = Perceptron(g=lambda x: 1 if x >= 0 else -1)
    train_set = np.array([
        [-0.6508,	0.1097,	4.0009,	-1.0000],
        [-1.4492,	0.8896,	4.4005,	-1.0000],
        [2.0850,	0.6876,	12.0710,	-1.0000],
        [0.2626,	1.1476,	7.7985,	1.0000],
        [0.6418,	1.0234,	7.0427,	1.0000],
        [0.2569,	0.6730,	8.3265,	-1.0000],
        [1.1155,	0.6043,	7.4446,	1.0000],
        [0.0914,	0.3399,	7.0677,	-1.0000],
        [0.0121,	0.5256,	4.6316,	1.0000],
        [-0.0429,	0.4660,	5.4323,	1.0000],
        [0.4340,	0.6870,	8.2287,	-1.0000],
        [0.2735,	1.0287,	7.1934,	1.0000],
        [0.4839,	0.4851,	7.4850,	-1.0000],
        [0.4089,	-0.1267,	5.5019,	-1.0000],
        [1.4391,	0.1614,	8.5843,	-1.0000],
        [-0.9115,	-0.1973,	2.1962,	-1.0000],
        [0.3654,	1.0475,	7.4858,	1.0000],
        [0.2144,	0.7515,	7.1699,	1.0000],
        [0.2013,	1.0014,	6.5489,	1.0000],
        [0.6483,	0.2183,	5.8991,	1.0000],
        [-0.1147,	0.2242,	7.2435,	-1.0000],
        [-0.7970,	0.8795,	3.8762,	1.0000],
        [-1.0625,	0.6366,	2.4707,	1.0000],
        [0.5307,	0.1285,	5.6883,	1.0000],
        [-1.2200,	0.7777,	1.7252,	1.0000],
        [0.3957,	0.1076,	5.6623,	-1.0000],
        [-0.1013,	0.5989,	7.1812,	-1.0000],
        [2.4482,	0.9455,	11.2095,	1.0000],
        [2.0149,	0.6192,	10.9263,	-1.0000],
        [0.2012,	0.2611,	5.4631,	1.0000],
    ])

    result = rna.fit(train_set, learn_rate=0.1)

    train_set = np.array([
        [-0.3665,	0.0620,	5.9891],
        [-0.7842,	1.1267,	5.5912],
        [0.3012,	0.5611,	5.8234],
        [0.7757,	1.0648,	8.0677],
        [0.1570,	0.8028,	6.3040],
        [-0.7014,	1.0316,	3.6005],
        [0.3748,	0.1536,	6.1537],
        [-0.6920,	0.9404,	4.4058],
        [-1.3970,	0.7141,	4.9263],
        [-1.8842,	-0.2805,	1.2548]
    ])
    print(result)

    Y = {}
    for i in range(train_set.shape[0]):
        Y[i] = rna.predict(train_set[i, :])
    print(Y)


if __name__ == '__main__':
    main()

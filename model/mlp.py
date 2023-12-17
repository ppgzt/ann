import numpy as np


class MLP:

    def __init__(self, layers=[]):
        self.layers = layers

    def fit(self, X_train, D_train, learn_rate=0.1, precision=0.001, verbose=False):
        epoch = -1
        Eqm = None

        while True:
            epoch += 1
            Eq = 0

            for i in range(0, X_train.shape[0]):
                X = X_train[i]

                Yi_stack = [X]
                for layer in self.layers:
                    Yi_stack.append(layer.predict(Yi_stack[-1]))

                last_layer = True
                Sk = None
                Yi_stack.pop()

                for layer in reversed(self.layers):
                    layer.fit(Yi=Yi_stack.pop(),
                              last_layer=last_layer, Dj=D_train[i], Sk=Sk)

                    Sk = []
                    for j in range(1, layer.W.shape[0]):
                        Sk.append((layer.S*layer.W[:, j]).sum())

                    if last_layer:
                        Eq += (layer.E ** 2).sum()
                        last_layer = False

            Eqm_epoch = Eq / X_train.shape[0]
            if Eqm != None and abs(Eqm_epoch - Eqm) < precision:
                Eqm = Eqm_epoch
                break
            else:
                Eqm = Eqm_epoch

        if verbose:
            print('Epochs: ', epoch, 'Eqm: ', Eqm, 'X: ', X, 'W: ')
            for k in range(len(self.layers)):
                print(' Layer ', k, ':\n', self.layers[k].W)
        return {'epochs': epoch+1}

    def predict(self, X):
        inputs = np.insert(X, 0, -1)
        return self.g(np.dot(self.W, inputs))


class Layer:

    def __init__(self, n_j=1, n_i=1, g=None, gd=None):
        self.W = np.zeros((n_j, n_i))
        self.S = None
        self.E = None

        self.g = g
        self.gd = gd

    def fit(self, Yi=None, last_layer=False, Dj=None, Sk=None, learn_rate=0.1):
        Yi = np.insert(Yi, 0, -1)

        Ij = np.dot(self.W, Yi)
        Yj = self.g(Ij)

        if last_layer:
            self.E = Dj - Yj
            self.S = self.E * self.gd(Ij)
        else:
            self.S = Sk * self.gd(Ij)

        self.W = self.W + learn_rate * np.dot(np.reshape(
            self.S, (self.S.shape[0], 1)), np.reshape(Yi, (1, Yi.shape[0])))

    def predict(self, X):
        return self.g(np.dot(self.W, np.insert(X, 0, -1)))

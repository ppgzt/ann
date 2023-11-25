import numpy as np

class Perceptron:

    def __init__(self, g=lambda x: 1 if x >= 0 else 0):
        self.params = np.array([])
        self.g = g

    def fit(self, train_set, learn_rate=0.1, verbose=False):
        self.params = np.random.random_sample((train_set.shape[1]))
        epoch = -1

        while True:
            error = False
            epoch += 1

            for sample in train_set:
                inputs = np.insert(sample[:-1], 0, -1)
                d = sample[-1]

                u = np.dot(self.params, inputs)
                y = self.g(u)
                
                if d != y:
                    error = True
                    self.params = self.params + (learn_rate * (d - y)) * inputs
                
                if verbose:
                    print('E: ', epoch, 'A: ', inputs, 'W: ', self.params)

            if error == False:
                break

        return {'epochs': epoch+1, 'W': self.params}

    def predict(self, X):
        inputs = np.insert(X, 0, -1)
        return self.g(np.dot(self.params, inputs))
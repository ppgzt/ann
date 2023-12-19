import numpy as np


class Math:

    def get(name="degrau"):
        functions = {
            "degrau": {
                "g": lambda x: 1 if x >= 0 else 0,
                "d": None
            },
            "logi": {
                "g": lambda x, b=1: 1/(1+np.exp(-b*x)),
                "d": lambda x, b=1: np.exp(-b*x)/((1+np.exp(-b*x))**2)
            },
            "tanh": {
                "g": lambda x, b=1: (1-np.exp(-b*x))/(1+np.exp(-b*x)),
                "d": lambda x, b=1: (2 * np.exp(-b*x)*b)/((1+np.exp(-b*x))**2)
            },
        }
        return functions[name]

import numpy as np


class Math:

    def get(name="degrau"):
        functions = {
            "degrau": {
                "g": lambda x: 1 if x >= 0 else 0,
                "d": None
            },
            "tanh": {
                "g": lambda x, b=1: 1/(1+np.exp(-b*x)),
                "d": lambda x: -2*(1/(1+np.exp(-x)))*np.exp(-x)
            }
        }
        return functions[name]

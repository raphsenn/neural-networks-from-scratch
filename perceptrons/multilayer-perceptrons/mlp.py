import numpy as np


class Perceptron:
    def __init__(self, inp: int) -> None:
        self.w = np.zeros(inp)
        self.b = np.zeros(1)

    def __call__(self, x: np.array) -> float:
        return np.dot(x, self.w) + self.b


class Layer:
    pass

class MLP:
    pass


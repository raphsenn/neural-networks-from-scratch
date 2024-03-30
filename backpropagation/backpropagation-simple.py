import numpy as np


class NN:
    """
    Design a basic neural network consisting of two nodes,
    representing the input and output layers respectively.
    This network operates without bias and assumes linearity.

    Our objective is to train this neural network to appropriately respond to an
    input value of 1.5 by producing an output of 0.5.
    """
    def __init__(self) -> None:
        """
        >>> nn = NN()
        >>> nn.w
        0.0
        """ 
        self.w = 0.0

    def train(self, X: float, epochs: int=1, learning_rate: float=0.1) -> None:
        for _ in range(epochs):
            a = x * self.w


    def __call__(self, x: float) -> float:
        """
        >>> nn = NN()
        >>> X_train = 1.5
        >>> y_train = 0.5
        """
        return x * self.w




import numpy as np


class NN:
    """
    Design a basic neural network consisting of two nodes,
    representing the input and output layers respectively.
    This network operates without bias and assumes linearity.

    Our objective is to train this neural network to appropriately respond to an
    input value of [0, 1, 2, 3, 4] by producing an output of [0, 0.5, 1.0, 1.5, 2.0].
    """
    def __init__(self) -> None:
        """
        >>> nn = NN()
        >>> nn.w
        0.0
        """ 
        self.w = 0.0

    def train(self, X: np.array, y: np.array, epochs: int=1, learning_rate: float=0.1) -> None:
        """
        """ 
        for _ in range(epochs):
            for i, x in enumerate(X): 
                # Forward prop. 
                a = x * self.w

                # Back prop.
                error = y[i] - a
                grad = 2 * x * error
                self.w += learning_rate * grad

    def __call__(self, x: float) -> float:
        """
        >>> nn = NN()
        >>> X_train = np.array([0, 1, 2, 3, 4, 5])
        >>> y_train = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        >>> nn.train(X_train, y_train, 1000, 0.01)
        >>> nn(0)
        0.0
        >>> nn(1)
        0.5
        >>> nn(2)
        1.0
        >>> nn(3)
        1.5
        >>> nn(4)
        2.0
        >>> nn(5)
        2.5
        >>> nn(1000)
        500.0
        >>> nn(-1)
        -0.5
        >>> nn(7)
        3.5
        >>> nn(2.5)
        1.25
        """
        return x * self.w

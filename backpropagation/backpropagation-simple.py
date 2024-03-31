

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
        0.8
        """ 
        self.w = 0.8

    def train(self, X: float, y: float, epochs: int=1, learning_rate: float=0.1) -> None:
        """
        >>> nn = NN()
        >>> X_train = 1.5
        >>> y_train = 0.5
        >>> nn.train(X_train, y_train)
        >>> nn.w
        0.59
        >>> nn.train(X_train, y_train)
        >>> nn.w
        0.4745
        >>> nn.train(X_train, y_train)
        >>> nn.w
        0.410975
        >>> nn.train(X_train, y_train)
        >>> nn.w
        0.37603625
        """ 
        for _ in range(epochs):
            a = X * self.w
            error = y - a
            grad = X * 2 * error
            self.w += learning_rate * grad

    def __call__(self, x: float) -> float:
        """
        >>> nn = NN()
        >>> X_train = 1.5
        >>> y_train = 0.5
        >>> nn.train(X_train, y_train, 100, 0.1)
        >>> nn(1.5)
        0.5
        """
        return x * self.w

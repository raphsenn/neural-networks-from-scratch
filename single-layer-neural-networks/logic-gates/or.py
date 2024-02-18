import numpy as np


class Perceptron:
    """
    A simple single-layer neuronal network (perceptron) for binary classification task
    """
    def __init__(self):
        """
        Initialize the perceptron with zero weights and learning rate.    
        """ 
        self.w = np.zeros(2)
        self.learning_rate = 0.1
        self.threshold = 0

    def __call__(self, x):
        """
        Compute the output of the perceptron for a given input.

        Args:
            x (array-like): 1-D Input vector with two elements either zero or one.

        Returns:
            int: Output of the perceptron (0 or 1).
        """ 
        if np.dot(x, self.w) > self.threshold:
            return 1
        return 0
    
    def train(self, X, y, epochs: int = 1):
        """
        Train the perceptron using the provided input-output pairs.

        Args:
            X (array-like): Input data.
            y (array-like): Target labels corresponding to the input data.

        """ 
        for _ in range(epochs):
            np.random.shuffle(X)
            for i, xi in enumerate(X):
                x = xi[:2]
                pred = np.dot(x, self.w)
                if pred != y[i]:
                    self.w = self.w + self.learning_rate * (y[i] - pred) * x


if __name__ == '__main__':
    X = np.array([[1, 1], [1, 0], [0, 0], [0, 1], [1, 0]])
    y = np.array([1, 1, 0, 1, 1])
    or_gate = Perceptron()
    or_gate.train(X, y)
    print(f"OR(0, 0) = {or_gate(np.array([0, 0]))}") 
    print(f"OR(0, 1) = {or_gate(np.array([0, 1]))}") 
    print(f"OR(1, 0) = {or_gate(np.array([1, 0]))}") 
    print(f"OR(1, 1) = {or_gate(np.array([1, 1]))}") 

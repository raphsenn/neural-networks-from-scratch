import numpy as np


class Neuron:
    """
    A simple single-layer neuronal network (perceptron) for binary classification task.
    """
    def __init__(self):
        """
        Initialize the perceptron with zero weights, learning rate and bias.    
        """ 
        self.w = np.zeros(3)
        self.learning_rate = 0.1
        self.bias = np.array([1.0]) 

    def __call__(self, x: np.array) -> int:
        """
        Compute the output of the perceptron for a given input.

        Args:
            x (array-like): 1-D Input vector with two elements either zero or one.

        Returns:
            int: Output of the perceptron (0 or 1).
        """ 
        if np.sign(np.dot(np.append(self.bias, x), self.w)) > 0:
            return 1
        return 0
    
    def train(self, X: np.array, y: np.array, epochs: int = 10) -> None:
        """
        Train the perceptron using the provided input-output pairs.

        Args:
            X (array-like): Input data.
            y (array-like): Target labels corresponding to the input data.
        """ 
        for _ in range(epochs):
            for i, xi in enumerate(X):
                x = np.append(self.bias, xi) # bias + inputs
                prediction = np.sign(np.dot(x, self.w))
                if prediction != y[i]:
                    if prediction <= 0:
                        self.w = self.w + x 
                    else:
                        self.w = self.w - x


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    or_gate = Neuron()
    or_gate.train(X, y)
    print(f"OR(0, 0) = {or_gate(np.array([0, 0]))}") 
    print(f"OR(0, 1) = {or_gate(np.array([0, 1]))}") 
    print(f"OR(1, 0) = {or_gate(np.array([1, 0]))}") 
    print(f"OR(1, 1) = {or_gate(np.array([1, 1]))}")

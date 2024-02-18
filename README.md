# neural-networks-from-scratch
Learn a little bit about neural networks with raw python.

## Biological neuron

<p float="left">
   <img src="./res/biological_neuron.png">
</p>

Neurons are interconnected nerve cells that process and transmits electrochemical signals in the brain.
There are approximately 100 billion of them inside of us, with each neuron connected to about 10,000 other neurons.

## Artifical neuron (Perceptron)

<p float="left">
   <img src="./res/artifical_neuron.jpg">
</p>

A perceptron is an artificial neuron. Instead of electrochemical signals, data is represented as numerical values.
These input values, normally labeled as X (See Figure 2), are multiplied by weights (W) and then added to represent the input values’ total strength.
If this weighted sum exceeds the threshold, the perceptron will trigger, sending a signal

You can call a perceptron a single-layer neural network.

## Application

### Logical gates
Our first application of a single layer neural network are logic gates.

#### OR Gate

<p float="left">
   <img src="./res/OR_gate.jpg">
</p>

<p float="left">
   <img src="./res/perceptron_OR.jpg">
</p>

```python
class Perceptron:
    """
    A simple single-layer neuronal network (perceptron) of the OR Gate. 
    """
    def __init__(self):
        self.w = np.zeros(2)
        self.learning_rate = 0.1
        self.threshold = 0

    def __call__(self, x: np.array):
        if np.dot(x, self.w) > self.threshold:
            return 1
        return 0
    
    def train(self, X: np.array, y: np.array, epochs: int = 1):
        for _ in range(epochs):
            for i, xi in enumerate(X):
                x = xi[:2]
                pred = np.dot(x, self.w)
                if pred != y[i]:
                    self.w = self.w + self.learning_rate * (y[i] - pred) * x
```

<p float="left">
   <img src="./res/update_OR.jpg">
</p>




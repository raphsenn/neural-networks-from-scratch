import numpy as np
import matplotlib.pyplot as plt


def a(w: float) -> float:
    return 1.5 * w


def cost(a: float) -> float:
    return (a - 0.5) ** 2

if __name__ == '__main__':
    x = np.linspace(-0.3, 1.0, 100)
    a = a(x)
    c = cost(a)
    plt.plot(x, a, 'b', label='a(w)')
    plt.plot(x, c, 'r', label='cost(a)')
    plt.legend()
    plt.show()




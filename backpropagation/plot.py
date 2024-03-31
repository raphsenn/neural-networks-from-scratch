import numpy as np
import matplotlib.pyplot as plt


def a(w: float) -> float:
    return 1.5 * w


def cost(a: float) -> float:
    return (a - 0.5) ** 2

def dcost_w(a: float) -> float:
    return 1.5 * 2 * (a - 0.5)

if __name__ == '__main__':
    x = np.linspace(-0.1, 1.0, 100)
    a = a(x)
    c = cost(a)
    dc_w = dcost_w(a)
    plt.plot(x, a, 'b', label='a(w)')
    plt.plot(x, c, 'r', label='cost(a)')
    plt.plot(x, dc_w, 'g', label='dcost_w(a)')
    plt.legend()
    plt.show()




import matplotlib.pyplot as plt

def plot_array(x):
    y = range(len(x))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.show()

def population_dynamic(r, A, N, K, t):
    return r * N(t) * (1 - N(t)/K ) * (N(t)/A - 1)

def f(t, w):
    return population_dynamic(0, 0, w, 0, t)

def getNextR4(previous, f, h):
    k1 = h * f(previous)
    k2 = h * f(previous + (1/2) * k1)
    k3 = h * f(previous + (1/2) * k2)
    k4 = h * f(previous + k3)
    return previous + (1/6)(k1 + 2*k2 + 2*k3 + k4)

def R4(f, h = 1e-6, iterations = 1e6):
    ret = []
    curr_it = 0
    for iter in range(iterations):
        curr_it = getNextR4(curr_it, f, h)
        ret.append(curr_it)
    return ret

def eulerMethod(w0, h, f, iterations):
    for i in range(iterations):
        w0 = w0 + h * 
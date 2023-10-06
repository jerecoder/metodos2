import matplotlib.pyplot as plt

def plot_array(x):
    y = range(len(x))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.show()

def population_dynamic(r, A, N, K, t): #w es N, no depende del tiempo
    return r * N * (1 - N/K ) * (N/A - 1)

def f(w):
    return population_dynamic(1, 1, w, 20, 0)

def getNextR4(previous, f, h):
    k1 = h * f(previous)
    k2 = h * f(previous + (1/2) * k1)
    k3 = h * f(previous + (1/2) * k2)
    k4 = h * f(previous + k3)
    return previous + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

def R4(w0, f, h = 1e-6, iterations = int(1e6)):
    ret = []
    curr_it = w0
    for iter in range(iterations):
        curr_it = getNextR4(curr_it, f, h)
        ret.append(curr_it)
    return ret

def eulerMethod(w0, f, h = 1e-6, iterations = int(1e6)):
    result_list = []
    for i in range(iterations):
        result_list.append(w0)
        w0 = w0 + h * f(w0)
    
    return result_list

def main():
    plot_array(R4(10, f, 0.1 , 200))
    plot_array(eulerMethod(10, f, 0.1, 200))

if __name__ == "__main__":
    main()
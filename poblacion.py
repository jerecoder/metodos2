import matplotlib.pyplot as plt

def plot_array(points):
    x, y = zip(*points)
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.xlabel("time")
    plt.ylabel("amount of individuals")
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

# def R4(w0, f, h = 1e-6, iterations = int(1e6)):
#     ret = []
#     curr_it = w0
#     for iter in range(iterations):
#         curr_it = getNextR4(curr_it, f, h)
#         ret.append(curr_it)
#     return ret

def R4(w0, f, t0, tn , iterations = int(1e6)):
    ret = []
    curr_it = w0
    h = (tn - t0)/iterations
    for iter in range(iterations):
        curr_it = getNextR4(curr_it, f, h)
        t0 += h
        ret.append((t0, curr_it))
    return ret


def eulerMethod(w0, f, t0, tn, iterations = 20000):
    result_list = []
    h = (tn - t0)/iterations
    for i in range(iterations):
        result_list.append((t0, w0))
        w0 = w0 + h * f(w0)
        t0 += h
    
    return result_list

def main():
    plot_array(R4(10, f, 0, 2.5, 2000))
    plot_array(eulerMethod(10, f, 0, 2.5, 2000))

if __name__ == "__main__":
    main()
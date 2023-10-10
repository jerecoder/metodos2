import math
import numpy as np
import matplotlib.pyplot as plt

ITER = 2000

def r(angle, l, t):
    return (l * math.sin(angle(t)), -l * math.cos(angle(t)))

def plot_pendulum_motion(results, V_results, T_results, E_results):
    """
    Plots the angular position, potential energy, kinetic energy, 
    and total energy of a pendulum over time.
    
    Parameters:
        results (list of tuples): Output from eulerMethod/R4teta, containing time, angular position,
                                  and angular velocity at each time point.
        V_results, T_results, E_results (list of tuples): Containing time and respective energies.
    """
    # Extracting time and angular position

    times, angular_positions, _ = zip(*results)
    
    # Create subplots
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot angular position vs time
    ax[0, 0].plot(times, angular_positions, color='b')
    ax[0, 0].set_xlabel('Tiempo')
    ax[0, 0].set_ylabel('Posición angular (rad)')
    ax[0, 0].set_title('Posición angular vs. Tiempo')
    ax[0, 0].grid(True)

    # Plot potential energy vs time
    _, V_vals = zip(*V_results)
    ax[0, 1].plot(times, V_vals, color='g')
    ax[0, 1].set_xlabel('Tiempo')
    ax[0, 1].set_ylabel('Energía Potencial (J)')
    ax[0, 1].set_title('Energía Potencial vs. Tiempo')
    ax[0, 1].grid(True)

    # Plot kinetic energy vs time
    _, T_vals = zip(*T_results)
    ax[1, 0].plot(times, T_vals, color='r')
    ax[1, 0].set_xlabel('Tiempo')
    ax[1, 0].set_ylabel('Energía Cinética (J)')
    ax[1, 0].set_title('Energía Cinética vs. Tiempo')
    ax[1, 0].grid(True)

    # Plot total mechanical energy vs time
    _, E_vals = zip(*E_results)
    ax[1, 1].plot(times, E_vals, color='purple')
    ax[1, 1].set_xlabel('Tiempo')
    ax[1, 1].set_ylabel('Total Energy (J)')
    ax[1, 1].set_title('Total Energy vs. Tiempo')
    ax[1, 1].grid(True)
    
    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

def plot_pendulum_motion_with_ground_truth(results, ground_truth_results, V_gt, T_gt, E_gt, V_results, T_results, E_results):
    """
    Plots the angular position, potential energy, kinetic energy, 
    and total energy of a pendulum over time.
    
    Parameters:
        results (list of tuples): Output from eulerMethod/R4teta, containing time, angular position,
                                  and angular velocity at each time point.
        ground_truth (list of tuples): Time and true angular position.
        V_gt, T_gt, E_gt (list of tuples): Time and true values for potential, kinetic, and total energy.
        V_results, T_results, E_results (list of tuples): Containing time and respective energies.
    """
    # Extracting time and angular position
    times, angular_positions, _ = zip(*results)
    
    # Create subplots
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot angular position vs time
    ax[0, 0].plot(times, angular_positions, color='b', label='Calculated')
    gt_times, gt_positions = zip(*ground_truth_results)
    ax[0, 0].plot(gt_times, gt_positions, color='b', linestyle='--', label='Ground Truth')
    ax[0, 0].set_xlabel('Tiempo')
    ax[0, 0].set_ylabel('Angular Position (rad)')
    ax[0, 0].set_title('Angular Position vs. Tiempo')
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # Plot potential energy vs time
    _, V_vals = zip(*V_results)
    ax[0, 1].plot(times, V_vals, color='g', label='Calculated')
    _, V_gt_vals = zip(*V_gt)
    ax[0, 1].plot(gt_times, V_gt_vals, color='g', linestyle='--', label='Ground Truth')
    ax[0, 1].set_xlabel('Tiempo')
    ax[0, 1].set_ylabel('Potential Energy (J)')
    ax[0, 1].set_title('Potential Energy vs. Tiempo')
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # Plot kinetic energy vs time
    _, T_vals = zip(*T_results)
    ax[1, 0].plot(times, T_vals, color='r', label='Calculated')
    _, T_gt_vals = zip(*T_gt)
    ax[1, 0].plot(gt_times, T_gt_vals, color='r', linestyle='--', label='Ground Truth')
    ax[1, 0].set_xlabel('Tiempo')
    ax[1, 0].set_ylabel('Kinetic Energy (J)')
    ax[1, 0].set_title('Kinetic Energy vs. Tiempo')
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Plot total mechanical energy vs time
    _, E_vals = zip(*E_results)
    ax[1, 1].plot(times, E_vals, color='purple', label='Calculated')
    _, E_gt_vals = zip(*E_gt)
    ax[1, 1].plot(gt_times, E_gt_vals, color='purple', linestyle='--', label='Ground Truth')
    ax[1, 1].set_xlabel('Tiempo')
    ax[1, 1].set_ylabel('Total Energy (J)')
    ax[1, 1].set_title('Total Energy vs. Tiempo')
    ax[1, 1].legend()
    ax[1, 1].grid(True)
    
    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_max_error(error_list):
    """
    Plots the error_list points as a continuous line.

    Parameters:
        error_list (list): A list of tuples containing pairs (i, error)
                           where `i` is an integer and `error` is a float.
    """
    # Unzip the error_list into two lists: `i_values` and `errors`.
    i_values, errors = zip(*error_list)
    
    # Plotting
    plt.plot(i_values, errors, label='Max Error', linestyle='-')
    plt.title('Max Error per Iteration')
    plt.xlabel('Theta')
    plt.ylabel('Max Error')
    plt.legend()
    plt.grid(True)
    plt.show()

acceleration = (0, 9.80665)

def linearized_ground_truth(t, w0, angle0):
    return angle0 * math.cos(w0 * t)

def linearized_ground_truth_derivative(t, w0, angle0):
    return -angle0 * w0 * math.sin( w0 * t)

def penislum_dynamic(angle_second_dertivative, w0, angle, t): # w0 sqrt(g/l), no cambia a lo largo de las iteraciones
    return angle_second_dertivative(t) + w0**2 * math.sin(angle(t))

def penislum_linealized_dynamic(angle_second_dertivative, w0, angle, t): # w0 sqrt(g/l), no cambia a lo largo de las iteraciones
    return angle_second_dertivative(t) + w0**2 * angle(t)


# def euler_method_linealized(u0, angle0, w0, t0, tn, iterations = 20000):
#     result_list = []
#     h = (tn - t0)/iterations
#     for i in range(iterations):
#         result_list.append((t0, u0, angle0))
#         new_angle = angle0 + h * u0
#         new_u = u0 + h * (-1)* (w0**2) * angle0
#         u0 = new_u
#         angle0 = new_angle
#         t0 += h
    
#     return result_list

def ground_truth_iteration(initial_state, w0, t0, tn, iterations=ITER):
    ret = []
    h = (tn - t0) / iterations
    angle0 = initial_state[0]

    for i in range(iterations):
        ret.append((t0, angle0))
        t0 += h
        angle0 = linearized_ground_truth(t0, w0, initial_state[0])
    
    return ret

def euler_method_teta(initial_state, w0, t0, tn, iterations=ITER):
    
    # Initialize arrays for results
    state = np.zeros((iterations, 2))
    t = np.linspace(t0, tn, iterations)
    
    # Set initial conditions
    state[0] = initial_state
    
    # Compute step size
    h = (tn - t0)/iterations
    
    # Euler method using vectorized operations
    for i in range(1, iterations):
        state[i] = state[i-1] + h * non_linear_f(state[i-1], w0)
    
    # Return results as a list of tuples
    return list(zip(t, state[:, 0], state[:, 1])) #devuelve t, angulo aproximado (se aproxima en base a su derivada, cosa que devuelve non linear f), u aproximado (seria como una aproximacion de la deivada del angulo)


def non_linear_f(state, w0):
    angle, u = state #angle es tita, u es la derivada de tita
    return np.array([u, -(w0**2)*math.sin(angle)]) #derivada del angulo, derivada de u (derivada segunda del angulo)

def getNextR4_two_dimensional(previous, f, h, w0):
    k1 = h * f(previous, w0)
    k2 = h * f(previous + 0.5 * k1, w0)
    k3 = h * f(previous + 0.5 * k2, w0)
    k4 = h * f(previous + k3, w0)
    return previous + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

def R4teta(initial_state, w0, t0, tn, iterations=ITER):
    ret = []
    h = (tn - t0) / iterations
    current_state = initial_state
    
    for _ in range(iterations):
        ret.append((t0, *current_state))
        current_state = getNextR4_two_dimensional(current_state, non_linear_f, h, w0)
        t0 += h
        
    return ret #devuelve t, angulo aproximado (se aproxima en base a su derivada, cosa que devuelve non linear f), u aproximado (seria como una aproximacion de la deivada del angulo)

def w0(l):
    return (acceleration[1]/l)**(1/2)

def max_error_calc(result1, result2):
    listoid = []
    for i in range(len(result1) - 1):
        listoid.append(abs(result1[i][1] - result2[i][1]))
    
    return max(listoid)

def max_error_tetha_iteration(length):

    error_list = []

    for i in length:

        initial_state = np.array((i, 0)) #tita, u (u es la derivada de tita)
        l = 10
        w_0 = w0(l)
        t0 = 0
        tn = 20
        mass = 1
        # plot_pendulum_motion(euler_method_linealized(5, math.radians(300), w0(10),0 , 20))
        R4T = R4teta(initial_state, w_0, t0, tn)
        groundTruth = ground_truth_iteration(initial_state, w_0, t0, tn)

        error_list.append((i, max_error_calc(R4T, groundTruth)))
    
    return error_list


def main():

    error = max_error_tetha_iteration([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1, 2, 3, 4, 5, 10, 20, 30])

    plot_max_error(error)


    
    # initial_state = np.array((0.5, 0)) #tita, u (u es la derivada de tita)
    # l = 10
    # w_0 = w0(l)
    # t0 = 0
    # tn = 20
    # mass = 1
    # # plot_pendulum_motion(euler_method_linealized(5, math.radians(300), w0(10),0 , 20))
    # eulerTeta = euler_method_teta(initial_state, w_0, t0 , tn) # (t, tita, u)
    # R4T = R4teta(initial_state, w_0, t0, tn)
    # groundTruth = ground_truth_iteration(initial_state, w_0, t0, tn)

    # plot_pendulum_motion(groundTruth, V_result_gt, T_result_gt, E_result_gt)

if __name__ == "__main__":
    main()
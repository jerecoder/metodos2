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
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Angular Position (rad)')
    ax[0, 0].set_title('Angular Position vs. Time')
    ax[0, 0].grid(True)

    # Plot potential energy vs time
    _, V_vals = zip(*V_results)
    ax[0, 1].plot(times, V_vals, color='g')
    ax[0, 1].set_xlabel('Time')
    ax[0, 1].set_ylabel('Potential Energy (J)')
    ax[0, 1].set_title('Potential Energy vs. Time')
    ax[0, 1].grid(True)

    # Plot kinetic energy vs time
    _, T_vals = zip(*T_results)
    ax[1, 0].plot(times, T_vals, color='r')
    ax[1, 0].set_xlabel('Time')
    ax[1, 0].set_ylabel('Kinetic Energy (J)')
    ax[1, 0].set_title('Kinetic Energy vs. Time')
    ax[1, 0].grid(True)

    # Plot total mechanical energy vs time
    _, E_vals = zip(*E_results)
    ax[1, 1].plot(times, E_vals, color='purple')
    ax[1, 1].set_xlabel('Time')
    ax[1, 1].set_ylabel('Total Energy (J)')
    ax[1, 1].set_title('Total Energy vs. Time')
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
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Angular Position (rad)')
    ax[0, 0].set_title('Angular Position vs. Time')
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # Plot potential energy vs time
    _, V_vals = zip(*V_results)
    ax[0, 1].plot(times, V_vals, color='g', label='Calculated')
    _, V_gt_vals = zip(*V_gt)
    ax[0, 1].plot(gt_times, V_gt_vals, color='g', linestyle='--', label='Ground Truth')
    ax[0, 1].set_xlabel('Time')
    ax[0, 1].set_ylabel('Potential Energy (J)')
    ax[0, 1].set_title('Potential Energy vs. Time')
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # Plot kinetic energy vs time
    _, T_vals = zip(*T_results)
    ax[1, 0].plot(times, T_vals, color='r', label='Calculated')
    _, T_gt_vals = zip(*T_gt)
    ax[1, 0].plot(gt_times, T_gt_vals, color='r', linestyle='--', label='Ground Truth')
    ax[1, 0].set_xlabel('Time')
    ax[1, 0].set_ylabel('Kinetic Energy (J)')
    ax[1, 0].set_title('Kinetic Energy vs. Time')
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Plot total mechanical energy vs time
    _, E_vals = zip(*E_results)
    ax[1, 1].plot(times, E_vals, color='purple', label='Calculated')
    _, E_gt_vals = zip(*E_gt)
    ax[1, 1].plot(gt_times, E_gt_vals, color='purple', linestyle='--', label='Ground Truth')
    ax[1, 1].set_xlabel('Time')
    ax[1, 1].set_ylabel('Total Energy (J)')
    ax[1, 1].set_title('Total Energy vs. Time')
    ax[1, 1].legend()
    ax[1, 1].grid(True)
    
    # Adjust layout and show plots
    plt.tight_layout()
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

# V y T
def V_on_point(t, m, l):
    return -m * acceleration[1] * l * math.cos(t) + m * acceleration[1] + l

def V(results, mass, length):
    #−mgℓ cos θ + mgℓ.
    return [(x[0], V_on_point(x[1], mass, length)) for x in results]

def T(results, mass, length):
    #1/2 * m * ℓ^2 * (θ')^2
    tuploide = []
    for res in results:
        tuploide.append((res[0], 1/2 * mass * (length ** 2) * (res[2]**2)))
    return tuploide

def E(V_results, T_results):
    return [(V_res[0], V_res[1] + T_res[1]) for V_res, T_res in zip(V_results, T_results)]

def T_for_gt(initial_state, w0, t0 , tn, mass, length, iterations = ITER):

    tuploide = []
    h = (tn - t0) / iterations

    for i in range(iterations):
        tuploide.append((t0, 1/2 * mass * (length ** 2) * (linearized_ground_truth_derivative(t0, w0, initial_state[0])**2)))
        t0  += h

    return tuploide


# V y T

def main():
    initial_state = np.array((0.5, 0)) #tita, u (u es la derivada de tita)
    l = 10
    w_0 = w0(l)
    t0 = 0
    tn = 20
    mass = 1
    # plot_pendulum_motion(euler_method_linealized(5, math.radians(300), w0(10),0 , 20))
    eulerTeta = euler_method_teta(initial_state, w_0, t0 , tn) # (t, tita, u)
    R4T = R4teta(initial_state, w_0, t0, tn)
    groundTruth = ground_truth_iteration(initial_state, w_0, t0, tn)

    V_result_euler = V(eulerTeta, mass, l)
    V_result_R4 = V(R4T, mass, l)
    V_result_gt = V(groundTruth, mass, l)

    T_result_euler = T(eulerTeta, mass, l)
    T_result_R4 = T(R4T, mass, l)
    T_result_gt = T_for_gt(initial_state, w_0, t0, tn, mass, l)

    E_result_euler = E(V_result_euler, T_result_euler)
    E_result_R4 = E(V_result_R4, T_result_R4)
    E_result_gt = E(V_result_gt, T_result_gt)


    plot_pendulum_motion(R4T, V_result_R4, T_result_R4, E_result_R4)
    plot_pendulum_motion_with_ground_truth(R4T, groundTruth, V_result_gt, T_result_gt, E_result_gt, V_result_R4, T_result_R4, E_result_R4)

    plot_pendulum_motion(eulerTeta, V_result_euler, T_result_euler, E_result_euler)
    plot_pendulum_motion_with_ground_truth(eulerTeta, groundTruth, V_result_gt, T_result_gt, E_result_gt, V_result_euler, T_result_euler, E_result_euler)

#    plot_pendulum_motion(eulerTeta, V_results_euler, T_result_euler, E_result_euler)
#    plot_pendulum_motion(R4T, V_result_R4, T_result_R4, E_result_R4)

if __name__ == "__main__":
    main()
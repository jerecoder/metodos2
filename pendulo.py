import math
import numpy as np

def r(angle, l, t):
    return (l * math.sin(angle(t)), -l * math.cos(angle(t)))

import matplotlib.pyplot as plt

def plot_pendulum_motion(results):
    """
    Plots the angular position and angular velocity of a pendulum over time.
    
    Parameters:
        results (list of tuples): Output from eulerMethod, containing time, angular velocity, 
                                  and angular position at each time point.
    """
    # Extracting time, angular velocity, and angular position
    times, angular_velocities, angular_positions = zip(*results)
    
    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot angular position vs time
    ax[0].plot(times, angular_positions, color='b')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Angular Position (rad)')
    ax[0].set_title('Angular Position vs. Time')
    ax[0].grid(True)

    # Plot angular velocity vs time
    ax[1].plot(times, angular_velocities, color='r')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Angular Velocity (rad/s)')
    ax[1].set_title('Angular Velocity vs. Time')
    ax[1].grid(True)
    
    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

acceleration = (0, 9.80665)

def penislum_dynamic(angle_second_dertivative, w0, angle, t): # w0 sqrt(g/l), no cambia a lo largo de las iteraciones
    return angle_second_dertivative(t) + w0**2 * math.sin(angle(t))

def penislum_linealized_dynamic(angle_second_dertivative, w0, angle, t): # w0 sqrt(g/l), no cambia a lo largo de las iteraciones
    return angle_second_dertivative(t) + w0**2 * angle(t)

def E(angle, T, V, t):
    return T(angle(t)) + V(angle(t))

def T(m, l, angle_derivative, t):
    return (1/2) * m * (l ** 2) * (angle_derivative(t) ** 2)

def V(m, l, angle, t):
    return m*acceleration*l*math.cos(angle(t)) + m * acceleration * l

def first_order_dynamic(u, angle, w0):
    return (u, (-1) * w0**2 * angle)

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

def non_linear_fu(u):
    return u

def non_linear_fangle(angle, w0):
    return -(w0**2)*math.sin(angle)

def euler_method(u0, angle0, w0, t0, tn, iterations = 2000):
    result_list = []
    h = (tn - t0)/iterations
    for i in range(iterations):
        result_list.append((t0, u0, angle0))
        new_angle = angle0 + h * u0
        new_u = u0 + h * (-1)* (w0**2) * np.sin(angle0)
        u0 = new_u
        angle0 = new_angle
        t0 += h
    
    return result_list

def getNextR4_two_dimensional(previous1, previous2, f1, f2, h, w0):
    k1_1 = h * f1(previous1)
    k2_1 = h * f1(previous1 + (1/2) * k1_1)
    k3_1 = h * f1(previous1 + (1/2) * k2_1)
    k4_1 = h * f1(previous1 + k3_1)
    k1_2 = h * f2(previous2, w0)
    k2_2 = h * f2(previous2 + (1/2) * k1_2, w0)
    k3_2 = h * f2(previous2 + (1/2) * k2_2, w0)
    k4_2 = h * f2(previous2 + k3_2, w0)
    return (previous1 + (1/6) * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1), previous2 + (1/6)* (k1_2 + 2*k2_2 + 2*k3_2 + k4_2))

def R4(u0, angle0, w0, t0, tn, iterations = 2000):
    ret = []
    h = (tn - t0)/iterations
    for iter in range(iterations):
        ret.append((t0, u0, angle0))
        u0, angle0 = getNextR4_two_dimensional(u0, angle0, non_linear_fu, non_linear_fangle, h, w0)
        t0 += h
    return ret

def w0(l):
    return (acceleration[1]/l)**(1/2)

def main():
    # plot_pendulum_motion(euler_method_linealized(5, math.radians(300), w0(10),0 , 20))
    plot_pendulum_motion(euler_method(0, 0.5, w0(10),0 , 20))
    plot_pendulum_motion(R4(0, 5, w0(10),0 , 20))

if __name__ == "__main__":
    main()
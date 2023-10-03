import math

def r(angle, l, t):
    return (l * math.sin(angle(t)), -l * math.cos(angle(t)))

acceleration = (0, -9.80665)

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

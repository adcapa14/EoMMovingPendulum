import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del sistema
g = 9.81      # Gravedad (m/s^2)
m = 0.5       # Masa del péndulo (kg)
L = 1.5      # Longitud del péndulo (m)

# Ecuación diferencial para el caso general
def pendulum_motion(t, y, x_func, dx_func, ddx_func):
    theta, theta_dot = y  # y[0] = theta, y[1] = theta_dot
    x = x_func(t)
    dx = dx_func(t)
    ddx = ddx_func(t)
    theta_ddot = - (g / L) * np.sin(theta) - (ddx / L) * np.cos(theta)
    return [theta_dot, theta_ddot]

# Definición de los casos de movimiento de x(t)
def linear_x(t):
    v = 0.5  # Velocidad constante en x (m/s)
    return v * t

def dlinear_x(t):
    v = 0.5  # Velocidad constante en x (m/s)
    return v

def ddlinear_x(t):
    return 0  # Aceleración constante

def harmonic_x(t):
    A = 0.5  # Amplitud del movimiento armónico (m)
    omega = 2.0  # Frecuencia angular (rad/s)
    return A * np.sin(omega * t)

def dharmonic_x(t):
    A = 0.5  # Amplitud del movimiento armónico (m)
    omega = 2.0  # Frecuencia angular (rad/s)
    return A * omega * np.cos(omega * t)

def ddharmonic_x(t):
    A = 0.5  # Amplitud del movimiento armónico (m)
    omega = 2.0  # Frecuencia angular (rad/s)
    return -A * omega**2 * np.sin(omega * t)

# Parámetros comunes para la simulación
theta0 = 0.1  # Ángulo inicial (rad)
theta_dot0 = 0.0  # Velocidad angular inicial (rad/s)
y0 = [theta0, theta_dot0]

t_span = (0, 10)  # Intervalo de tiempo (s)
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Puntos de evaluación

# Resolver para ambos casos
solutions = {}

for case_name, funcs in {
    "Lineal": (linear_x, dlinear_x, ddlinear_x),
    "Armónico": (harmonic_x, dharmonic_x, ddharmonic_x),
}.items():
    x_func, dx_func, ddx_func = funcs
    sol = solve_ivp(pendulum_motion, t_span, y0, t_eval=t_eval, args=(x_func, dx_func, ddx_func))
    solutions[case_name] = sol

# Graficar resultados
for case_name, sol in solutions.items():
    t = sol.t
    theta = sol.y[0]
    theta_dot = sol.y[1]
    
    x_func, dx_func, _ = {
        "Lineal": (linear_x, dlinear_x, ddlinear_x),
        "Armónico": (harmonic_x, dharmonic_x, ddharmonic_x),
    }[case_name]
    
    x_anchor = x_func(t)
    dx_anchor = dx_func(t)
    
    # Posición del péndulo
    x_p = x_anchor + L * np.sin(theta)
    y_p = -L * np.cos(theta)
    
    # Energías
    kinetic_energy = 0.5 * m * (L * theta_dot)**2
    potential_energy = m * g * (y_p - y_p.min())
    total_energy = kinetic_energy + potential_energy
    
    # Graficar
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Movimiento {case_name}")
    
    # Ángulo vs tiempo
    plt.subplot(2, 2, 1)
    plt.plot(t, theta, label=r"$\theta(t)$")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (rad)")
    plt.title("Evolución del ángulo")
    plt.legend()
    plt.grid()
    
    # Velocidad angular vs tiempo
    plt.subplot(2, 2, 2)
    plt.plot(t, theta_dot, label=r"$\dot{\theta}(t)$", color='orange')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad angular (rad/s)")
    plt.title("Velocidad angular")
    plt.legend()
    plt.grid()
    
    # Energías
    plt.subplot(2, 2, 3)
    plt.plot(t, kinetic_energy, label="Energía cinética", color='green')
    plt.plot(t, potential_energy, label="Energía potencial", color='red')
    plt.plot(t, total_energy, label="Energía total", color='blue', linestyle='--')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Energía (J)")
    plt.title("Energías del sistema")
    plt.legend()
    plt.grid()
    
    # Trayectoria del péndulo
    plt.subplot(2, 2, 4)
    plt.plot(x_anchor, np.zeros_like(x_anchor), label="Soporte", color='purple')
    plt.plot(x_p, y_p, label="Péndulo", color='brown')
    plt.xlabel("Posición x (m)")
    plt.ylabel("Posición y (m)")
    plt.title("Trayectoria del péndulo")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
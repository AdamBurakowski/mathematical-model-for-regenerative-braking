import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def velocity_model(v, t, mass, F_braking_func, C_d, A, rho, C_rr, theta):
    F_drag = 0.5 * rho * C_d * A * v**2
    F_rolling = C_rr * mass * 9.81 * np.cos(theta)
    F_braking = F_braking_func(t)
    F_total = F_drag + F_rolling + F_braking + mass * 9.81 * np.sin(theta)
    dv_dt = -F_total / mass
    return max(dv_dt, -v / (t + 1e-6))

def energy_regen(mass, v0, F_braking_func, C_d, A, rho, t_max, t_points, a, b, c, C_rr, theta):
    time = np.linspace(0, t_max, t_points)

    v_t = odeint(velocity_model, v0, time, args=(mass, F_braking_func, C_d, A, rho, C_rr, theta)).flatten()

    efficiency_v = np.clip(a * v_t**2 + b * v_t + c, 0, 1)

    F_braking_values = np.array([F_braking_func(t) for t in time])
    power = np.maximum(0, efficiency_v * F_braking_values * v_t)
    energy = np.trapz(power, time)

    return time, v_t, power, energy, efficiency_v

def F_braking_func(t):
    return max(0, F_braking_max * (1 - t / t_max))


mass = int(input("Podaj masę pojazdu (kg): "))
v0 = int(input("Podaj prędkość początkową (m/s): "))
F_braking_max = float(input("Podaj maksymalną siłę hamowania (N): "))
C_d = float(input("Podaj współczynnik oporu powietrza (np. 0.3 dla sapomochodu): "))
A = float(input("Podaj powierzchnię czołową pojazdu (m²): "))
rho = 1.225  # Air density (kg/m³)
theta = np.radians(float(input("Podaj nachylenie drogi w stopniach (ujemne dla zjazdu): ")))

# Efficiency coefficients
a = -0.005
b = 0.1
c = 0.2

# Rolling resistance coefficient
C_rr = 0.015  # Typical value for cars on asphalt

# linespace parameters
t_max = 10
t_points = 1000

time, v_t, power, energy, efficiency_v = energy_regen(
    mass, v0, F_braking_func, C_d, A, rho, t_max, t_points, a, b, c, C_rr, theta
)

energy_kJ = energy / 1000

print(f"Odzyskana energia: {energy:.2f} J")

plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(time, v_t, label="Prędkość (v)", color="blue")
plt.xlabel("Czas (s)")
plt.ylabel("Prędkość (m/s)")
plt.title("Zmiana prędkości w czasie")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, power, label="Moc (P)", color="orange")
plt.fill_between(time, power, color="orange", alpha=0.3, label=f"Energia = {energy_kJ:.0f} kJ")
plt.xlabel("Czas (s)")
plt.ylabel("Moc (W)")
plt.title("Rekuperacja energii w czasie")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, efficiency_v, label="Sprawność (η)", color="green")
plt.xlabel("Czas (s)")
plt.ylabel("Sprawność")
plt.title("Sprawność systemu w czasie")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

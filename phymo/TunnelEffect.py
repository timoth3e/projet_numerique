import numpy
import matplotlib.pyplot as plt
import scipy.integrate

import PaquetOndes as po
from SchrodingerEvolution import (
    Initial_condition,
    Evolve_stable_visscher,
    Compute_norm_evolution,
)

HBAR = po.HBAR
MASS = po.MASS


def Build_potential_barrier(x, V0, a, x_left=0.0):
    V = numpy.zeros_like(x)
    inside = (x >= x_left) & (x <= x_left + a)
    V[inside] = V0
    return V


def Classical_energy(k0):
    return HBAR**2 * k0**2 / (2 * MASS)


def Transmission_probability(psi, x, x_right):
    density = numpy.abs(psi)**2
    mask = x > x_right
    if mask.sum() < 2:
        return 0.0
    return scipy.integrate.simpson(density[mask], x[mask])


def Crossing_time(times, psi_history, x, x_right, threshold=0.5):
    n_steps = psi_history.shape[0]
    proba_right = numpy.array(
        [Transmission_probability(psi_history[n], x, x_right) for n in range(n_steps)]
    )

    proba_final = proba_right[-1]
    if proba_final <= 1e-12:
        return None, proba_right

    target = threshold * proba_final
    index = numpy.searchsorted(proba_right, target)
    if index <= 0 or index >= n_steps:
        return times[min(index, n_steps - 1)], proba_right

    t0, t1 = times[index - 1], times[index]
    p0, p1 = proba_right[index - 1], proba_right[index]
    t_cross = t0 + (t1 - t0) * (target - p0) / (p1 - p0)
    return t_cross, proba_right


if __name__ == "__main__":

    x = po.x
    dx = x[1] - x[0]
    k0 = po.K0
    E = Classical_energy(k0)

    a_barrier = 5.0
    V0_barrier = 1.5 * E
    x_left = 0.0
    x_right = x_left + a_barrier

    V_free = numpy.zeros_like(x)
    V_barrier = Build_potential_barrier(x, V0_barrier, a_barrier, x_left)

    psi0 = Initial_condition(x)

    dt = 0.1 * MASS * dx**2 / HBAR
    n_steps = 4000
    times = numpy.arange(n_steps) * dt

    print(f"Energie centrale du paquet  E   = {E:.4f}")
    print(f"Hauteur de la barriere      V0  = {V0_barrier:.4f}   (E/V0 = {E / V0_barrier:.3f})")
    print(f"Largeur de la barriere      a   = {a_barrier}")
    print(f"dx = {dx:.4f}, dt = {dt:.6f}, t_max = {times[-1]:.3f}")

    print("\nEvolution libre (reference tau_0)...")
    psi_free = Evolve_stable_visscher(psi0, x, V_free, dt, n_steps)
    tau0_num, proba_free = Crossing_time(times, psi_free, x, x_right)
    print(f"tau_0,num = {tau0_num}")

    print("\nEvolution avec barriere (tau_t)...")
    psi_barrier = Evolve_stable_visscher(psi0, x, V_barrier, dt, n_steps)
    tau_t_num, proba_barrier = Crossing_time(times, psi_barrier, x, x_right)
    print(f"tau_t,num = {tau_t_num}")

    coefficient_transmission = proba_barrier[-1]
    print(f"\nCoefficient de transmission numerique (a t_max) T = {coefficient_transmission:.4f}")

    norm_barrier = Compute_norm_evolution(psi_barrier, x)
    print(f"Norme finale avec barriere (devrait rester ~1) : {norm_barrier[-1]:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    axes[0].plot(times, proba_free, label="particule libre")
    axes[0].plot(times, proba_barrier, label="avec barriere")
    if tau_t_num is not None:
        axes[0].axvline(tau_t_num, color="crimson", ls="--", lw=0.8, label=r"$\tau_t$ estime")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$P(x > x_{droite})$")
    axes[0].legend()
    axes[0].set_title("Probabilite de presence a droite de la barriere")

    density_final = numpy.abs(psi_barrier[-1])**2
    axes[1].plot(x, density_final, label=r"$|\Psi(x,t_{max})|^2$")
    axes[1].axvspan(x_left, x_right, color="gray", alpha=0.3, label="barriere")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$|\Psi|^2$")
    axes[1].legend()
    axes[1].set_title(f"Densite finale (E/V0 = {E / V0_barrier:.2f})")

    plt.tight_layout()
    plt.savefig("tunnel_effect_check.png", dpi=120)
    print("\nGraphique enregistre : tunnel_effect_check.png")

import numpy
import matplotlib.pyplot as plt
import scipy.integrate

import PaquetOndes as po

HBAR = po.HBAR
MASS = po.MASS


def Initial_condition(x, sigma=po.SIGMA, k0=po.K0, x0=po.X0):
    return po.Compute_gaussian_wp(x, 0.0, sigma=sigma, k0=k0, x0=x0)


def Build_potential_free(x):
    return numpy.zeros_like(x)


def Hamiltonian_action(psi, x, V, dx):
    laplacian = numpy.zeros_like(psi, dtype=complex)
    laplacian[1:-1] = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / dx**2
    return -HBAR**2 / (2 * MASS) * laplacian + V * psi


def Evolve_naive_ftcs(psi0, x, V, dt, n_steps):
    dx = x[1] - x[0]
    nx = len(x)
    psi_history = numpy.zeros((n_steps, nx), dtype=complex)
    psi_history[0] = psi0

    psi = psi0.copy()
    for n in range(1, n_steps):
        psi = psi - 1j * dt / HBAR * Hamiltonian_action(psi, x, V, dx)
        psi_history[n] = psi

    return psi_history


def Evolve_stable_visscher(psi0, x, V, dt, n_steps):
    dx = x[1] - x[0]
    nx = len(x)

    R = numpy.real(psi0).astype(complex)
    I = numpy.imag(psi0).astype(complex)

    I = I - 0.5 * dt / HBAR * Hamiltonian_action(R, x, V, dx)

    psi_history = numpy.zeros((n_steps, nx), dtype=complex)
    psi_history[0] = R + 1j * numpy.real(I)

    for n in range(1, n_steps):
        R = R + dt / HBAR * Hamiltonian_action(I, x, V, dx)
        I = I - dt / HBAR * Hamiltonian_action(R, x, V, dx)
        psi_history[n] = numpy.real(R) + 1j * numpy.real(I)

    return psi_history


def Compute_norm_evolution(psi_history, x):
    n_steps = psi_history.shape[0]
    norms = numpy.zeros(n_steps)
    for n in range(n_steps):
        density = numpy.abs(psi_history[n])**2
        norms[n] = scipy.integrate.simpson(density, x)
    return norms


if __name__ == "__main__":

    x = po.x
    dx = x[1] - x[0]
    V = Build_potential_free(x)

    psi0 = Initial_condition(x)

    dt = 0.1 * MASS * dx**2 / HBAR
    n_steps = 3000
    times = numpy.arange(n_steps) * dt

    print(f"dx = {dx:.4f}, dt = {dt:.5f}, t_max = {times[-1]:.3f}")

    print("\nEvolution avec le schema naif (instable)...")
    psi_naive = Evolve_naive_ftcs(psi0, x, V, dt, n_steps)
    norm_naive = Compute_norm_evolution(psi_naive, x)

    print("Evolution avec le schema de Visscher (stable)...")
    psi_stable = Evolve_stable_visscher(psi0, x, V, dt, n_steps)
    norm_stable = Compute_norm_evolution(psi_stable, x)

    psi_analytic_final = po.Compute_gaussian_wp(x, times[-1])
    density_analytic_final = numpy.abs(psi_analytic_final)**2
    density_stable_final = numpy.abs(psi_stable[-1])**2

    ecart = numpy.sqrt(scipy.integrate.simpson(
        (density_stable_final - density_analytic_final)**2, x))

    print()
    print(f"Norme finale (schema naif)     : {norm_naive[-1]:.4f}  (devrait rester ~1)")
    print(f"Norme finale (schema Visscher) : {norm_stable[-1]:.4f}  (devrait rester ~1)")
    print(f"Ecart quadratique densite (Visscher vs analytique) : {ecart:.4e}")

    fig, (ax_norm, ax_density) = plt.subplots(2, 1, figsize=(9, 7))

    ax_norm.plot(times, norm_naive, label="schema naif (instable)")
    ax_norm.plot(times, norm_stable, label="schema de Visscher (stable)")
    ax_norm.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax_norm.set_xlabel("t")
    ax_norm.set_ylabel("norme de Psi")
    ax_norm.set_title("Conservation de la norme : schema naif vs schema stable")
    ax_norm.legend()
    ax_norm.set_yscale("log")

    ax_density.plot(x, density_analytic_final, label="analytique (PaquetOndes.py)", lw=2)
    ax_density.plot(x, density_stable_final, "--", label="numerique (Visscher)", lw=2)
    ax_density.set_xlabel("x")
    ax_density.set_ylabel(r"$|\Psi(x,t_{max})|^2$")
    ax_density.set_title(f"Densite de probabilite a t = {times[-1]:.2f}")
    ax_density.legend()

    plt.tight_layout()
    plt.savefig("schrodinger_free_check.png", dpi=120)
    print("\nGraphique enregistre : schrodinger_free_check.png")

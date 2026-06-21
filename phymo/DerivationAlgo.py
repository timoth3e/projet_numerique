import numpy


def Square(x):
    return x**2


def Derivative_of_square(x):
    return 2 * x


def First_derivative(f_values: numpy.ndarray, dx: float) -> numpy.ndarray:
    n = len(f_values)
    derivative = numpy.zeros(n)

    derivative[1:-1] = (f_values[2:] - f_values[:-2]) / (2 * dx)
    derivative[0] = (f_values[1] - f_values[0]) / dx
    derivative[-1] = (f_values[-1] - f_values[-2]) / dx

    return derivative


def Second_derivative(f_values: numpy.ndarray, dx: float) -> numpy.ndarray:
    n = len(f_values)
    second_derivative = numpy.zeros(n)

    second_derivative[1:-1] = (f_values[2:] - 2 * f_values[1:-1] + f_values[:-2]) / dx**2
    second_derivative[0] = second_derivative[1]
    second_derivative[-1] = second_derivative[-2]

    return second_derivative


def Relative_error(numeric: numpy.ndarray, analytic: numpy.ndarray) -> numpy.ndarray:
    analytic_safe = numpy.where(analytic == 0, 1.0, analytic)
    error = numpy.abs(numeric - analytic) / numpy.abs(analytic_safe)
    error = numpy.where(analytic == 0, numpy.abs(numeric - analytic), error)
    return error


if __name__ == "__main__":

    n_pts = 200
    x = numpy.linspace(-10, 10, n_pts)
    dx = x[1] - x[0]

    f = Square(x)

    f_prime_numeric = First_derivative(f, dx)
    f_prime_analytic = Derivative_of_square(x)
    error1 = Relative_error(f_prime_numeric, f_prime_analytic)

    print("=== Derivee premiere de x**2 (comparaison a 2x) ===")
    print(f"Erreur relative max (points interieurs)    : {error1[1:-1].max():.3e}")
    print(f"Erreur relative moyenne (points interieurs) : {error1[1:-1].mean():.3e}")

    f_second_numeric = Second_derivative(f, dx)
    f_second_analytic = numpy.full_like(x, 2.0)
    error2 = Relative_error(f_second_numeric, f_second_analytic)

    print()
    print("=== Derivee seconde de x**2 (comparaison a 2) ===")
    print(f"Erreur relative max (points interieurs)    : {error2[1:-1].max():.3e}")
    print(f"Erreur relative moyenne (points interieurs) : {error2[1:-1].mean():.3e}")

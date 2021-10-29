import numpy as np
import cvxpy as cp

from PEPit.pep import PEP
from PEPit.Function_classes.smooth_convex_function import SmoothConvexFunction
from PEPit.Function_classes.smooth_strongly_convex_function import SmoothStronglyConvexFunction
from PEPit.Function_classes.convex_indicator import ConvexIndicatorFunction
from PEPit.Primitive_steps.bregmangradient_step import BregmanGradient_Step


def wc_iipp(L, mu, c, lam, n, verbose=True):
    """
    Consider the composite convex minimization problem,
        min_x { F(x) = f_1(x) + f_2(x) }
    where f_1(x) is a smooth convex function, and f_2(x) is a closed convex indicator.
    We use a kernel h that is assumed to be non-smooth strongly convex.

    This code computes a worst-case guarantee for Improved Interior Point method.
    That is, it computes the smallest possible tau(n,L) such that the guarantee
        F(x_n) - F(x_*) <= tau(n,L) * (c* Dh(x_*,x_0) + f(x0) - fs)
    is valid, where x_n is the output of the Improved Interior Point method,
    where x_* is a minimizer of F,when Dh is the Bregman distance generated by h.

    The detailed approach is available in
    [1] Alfred Auslender, and Marc Teboulle. "Interior gradient and proximal
     methods for convex and conic optimization."
     SIAM Journal on Optimization (2006).

    :param L: (float) the smoothness parameter.
    :param mu: (float) the strong-convexity parameter
    :param c: (float) initial value
    :param lam: (float) the step size.
    :param n: (int) number of iterations
    :param verbose: (bool) if True, print conclusion

    :return: (tuple) worst_case value, theoretical value
    """

    # Instantiate PEP
    problem = PEP()

    # Declare three convex functions
    func1 = problem.declare_function(SmoothConvexFunction, param={'L': L})
    func2 = problem.declare_function(ConvexIndicatorFunction, param={'D': np.inf})
    h = problem.declare_function(SmoothStronglyConvexFunction, param={'mu': mu, 'L': np.inf})

    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    fs = func.value(xs)
    ghs, hs = h.oracle(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()
    gh0, h0 = h.oracle(x0)
    g10, f10 = func1.oracle(x0)

    # Compute n steps of the Improved Interior Algorithm starting from x0
    x = x0
    z = x0
    g = g10
    f = f10
    gh = gh0
    ck = c
    for i in range(n):
        alphak = (np.sqrt((ck * lam) ** 2 + 4 * ck * lam) - lam * ck) / 2
        ck = (1 - alphak) * ck
        y = (1 - alphak) * x + alphak * z
        if i >= 1:
            g, f = func1.oracle(y)
        z, _, _ = BregmanGradient_Step(g, gh, h + func2, alphak / ck)
        x = (1 - alphak) * x + alphak * z
        gh, _ = h.oracle(z)

    # Set the initial constraint that is a Lyapunov distance between x0 and x^*
    problem.set_initial_condition((hs - h0 - gh0 * (xs - x0)) * c + f10 - fs <= 1)

    # Set the performance metric to the final distance in function values to optimum
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    try:
        pepit_tau = problem.solve(solver=cp.MOSEK, verbose=verbose)
    except cp.error.SolverError:
        pepit_tau = problem.solve(verbose=verbose)
        print("\033[93m(PEP-it) We recommend to use an other solver, such as MOSEK. \033[0m")

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 4 * L / (n + 1) ** 2 / c

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of the Improved Interior Point method in function values ***')
        print('\tPEP-it guarantee:\t f(x_n)-f_* <= {:.6} (c * Dh(x0,xs) + f1(x0) - f(xs))'.format(pepit_tau))
        print(
            '\tTheoretical guarantee :\t f(x_n)-f_* <= {:.6} (c * Dh(x0,xs) + f1(x0) - f(xs))'.format(theoretical_tau))
    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    mu = 1
    c = 1
    lam = 1 / L
    n = 5

    pepit_tau, theoretical_tau = wc_iipp(L=L,
                                         mu=mu,
                                         c=c,
                                         lam=lam,
                                         n=n)

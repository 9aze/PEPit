import numpy as np

from PEPit.pep import PEP
from PEPit.functions.smooth_strongly_convex_function import SmoothStronglyConvexFunction


def wc_sgdo(L, mu, gamma, n, verbose=True):
    """
    Consider the finite sum minimization problem

    .. math:: F_\\star \\triangleq \\min_x \{F(x) \\equiv \\frac{1}{n} (f_1(x) + ... + f_n(x))\},

    where :math:`f_1, ..., f_n` are :math:`L`-smooth and :math:`\\mu`-strongly convex.

    In addition, we assume a zero variance at the optimal point

    .. math:: \\mathbb{E}\\left[\\|\\nabla f_i(x_\\star)\\|^2\\right] = \\frac{1}{n} \\sum_{i=1}^n \\|\\nabla f_i(x_\\star)\\|^2 = 0,

    which happens for example in machine learning in the interpolation regime,
    that is if there exists a model :math:`x_\\star`
    such that the loss :math:`\\mathcal{L}` on any observation :math:`(z_i)_{i \\in [|1, n|]}`,
    :math:`\\mathcal{L}(x_\\star, z_i) = f_i(x_\\star)` is zero.

    This code computes a worst-case guarantee for one step of the **stochastic gradient descent (SGD)** in expectation,
    for the distance to optimal point.

    That is, it computes the smallest possible :math:`\\tau(L, \\mu, \\gamma, n)` such that

    .. math:: \\mathbb{E}\\left[\\|x_1 - x_\\star\\|^2\\right] \\leqslant \\tau(L, \\mu, \\gamma, n) \\|x_0 - x_\\star\\|^2

    Here, where :math:`x_1` is the output of one step of **stochastic gradient descent (SGD)**.

    **Algorithm**:

        .. math:: x_{t+1} = x_t - \\gamma \\nabla f_{i_t}(x_t)

        with

        .. math:: i_t \\sim \\mathcal{U}\\left([|1, n|]\\right)

    **Theoretical guarantee**:

        TODO

        The **tight** guarantee obtained in ?? is

        .. math:: \\mathbb{E}\\left[\\|x_1 - x_\\star\\|^2\\right] \\leqslant \\left(1-\\frac{\\mu}{L}\\right)^2 \\|x_0 - x_\\star\\|^2.

    Notes:

        We will observe it does not depend on the number `math:`n` of functions for this particular setting,
        hence the guarantees are also valid for expectation minimization settings (i.e., when :math:`n` goes to infinity).

    References:

        TODO

    Args:
        L (float): the smoothness parameter.
        mu (float): the strong convexity parameter.
        gamma (float): the step-size.
        n (int): number of functions.
        verbose (bool): if True, print conclusion.

    Returns:
        tuple: worst_case value, theoretical value

    Example:
        >>> mu = 0.1
        >>> L = 1
        >>> gamma = 1 / L
        >>> pepit_tau, theoretical_tau = wc_sgdo(L=L, mu=mu, gamma=gamma, n=5, verbose=True)
        (PEP-it) Setting up the problem: size of the main PSD matrix: 16x16
        (PEP-it) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEP-it) Setting up the problem: initial conditions (2 constraint(s) added)
        (PEP-it) Setting up the problem: interpolation conditions for 5 function(s)
                 function 1 : 6 constraint(s) added
                 function 2 : 6 constraint(s) added
                 function 3 : 6 constraint(s) added
                 function 4 : 6 constraint(s) added
                 function 5 : 6 constraint(s) added
        (PEP-it) Compiling SDP
        (PEP-it) Calling SDP solver
        (PEP-it) Solver status: optimal (solver: SCS); optimal value: 0.8100000763718881
        *** Example file: worst-case performance of stochastic gradient descent with fixed step-size and with zero variance at the optimal point ***
            PEP-it guarantee:		 E[||x_1 - x_*||^2] <= 0.81 ||x0 - x_*||^2
            Theoretical guarantee:	 E[||x_1 - x_*||^2] <= 0.81 ||x0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    fn = [problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu}) for _ in range(n)]
    func = np.mean(fn)

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the bounded variance and the distance between initial point and optimal one
    var = np.mean([f.gradient(xs) ** 2 for f in fn])

    problem.set_initial_condition(var <= 0.)
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Compute the *expected* distance to optimality after running one step of the stochastic gradient descent
    distavg = np.mean([(x0 - gamma * f.gradient(x0) - xs) ** 2 for f in fn])

    # Set the performance metric to the distance average to optimal point
    problem.set_performance_metric(distavg)

    # Solve the PEP
    pepit_tau = problem.solve(verbose=verbose)

    # Compute theoretical guarantee (for comparison)
    kappa = L / mu
    theoretical_tau = (1 - 1 / kappa) ** 2

    # Print conclusion if required
    if verbose:
        print('*** Example file: worst-case performance of stochastic gradient descent'
              ' with fixed step-size and with zero variance at the optimal point ***')
        print('\tPEP-it guarantee:\t\t E[||x_1 - x_*||^2] <= {:.6} ||x0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t E[||x_1 - x_*||^2] <= {:.6} ||x0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":

    mu = 0.1
    L = 1
    gamma = 1 / L
    pepit_tau, theoretical_tau = wc_sgdo(L=L, mu=mu, gamma=gamma, n=5, verbose=True)

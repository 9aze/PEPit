from PEPit import PEP
from PEPit import null_point
from PEPit.functions import ConvexIndicatorFunction
from PEPit.primitive_steps import proximal_step


def wc_dykstra(n, verbose=1):
    """
    Consider the convex feasibility problem:

    .. math:: \\mathrm{Find}\\, x\\in Q_1\\cap Q_2

    where :math:`Q_1` and :math:`Q_2` are two closed convex sets.

    This code computes a worst-case guarantee for the **Dykstra projection method**, and looks for a low-dimensional
    worst-case example nearly achieving this worst-case guarantee.
    That is, it computes the smallest possible :math:`\\tau(n)` such that the guarantee

    .. math:: \\|\\mathrm{Proj}_{Q_1}(x_n)-\\mathrm{Proj}_{Q_2}(x_n)\\|^2 \\leqslant \\tau(n) \\|x_0 - x_\\star\\|^2

    is valid, where :math:`x_n` is the output of the **Dykstra projection method**,
    and :math:`x_\\star\\in Q_1\\cap Q_2` is a solution to the convex feasibility problem.

    In short, for a given value of :math:`n`,
    :math:`\\tau(n)` is computed as the worst-case value of
    :math:`\\|\\mathrm{Proj}_{Q_1}(x_n)-\\mathrm{Proj}_{Q_2}(x_n)\\|^2` when :math:`\\|x_0 - x_\\star\\|^2 \\leqslant 1`.
    Then, it looks for a low-dimensional nearly achieving this performance.
    
    **Algorithm**: The Dykstra projection method can be written as

        .. math::
            \\begin{eqnarray}
                y_{t} & = & \\mathrm{Proj}_{Q_1}(x_t+p_t), \\\\
                p_{t+1} & = & x_t + p_t - y_t,\\\\
                x_{t+1} & = & \\mathrm{Proj}_{Q_2}(y_t+q_t),\\\\
                q_{t+1} & = & y_t + q_t - x_{t+1}.
            \\end{eqnarray}

    **References**: This method is due to [1].

    `[1] J.P. Boyle, R.L. Dykstra (1986). A method for finding projections onto the intersection of convex sets in Hilbert spaces.
    Lecture Notes in Statistics. Vol. 37. pp. 28–47.
    <https://link.springer.com/chapter/10.1007/978-1-4613-9940-7_3>`_
    
    Args:
        n (int): number of iterations.
        verbose (int): Level of information details to print.
                        
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.

    Returns:
        pepit_tau (float): worst-case value
        theoretical_tau (None): no theoretical value.

    Example:
        >>> pepit_tau, theoretical_tau = wc_dykstra(n=10, verbose=1)
        (PEPit) Setting up the problem: size of the main PSD matrix: 24x24
        (PEPit) Setting up the problem: performance measure is minimum of 1 element(s)
        (PEPit) Setting up the problem: Adding initial conditions and general constraints ...
        (PEPit) Setting up the problem: initial conditions and general constraints (1 constraint(s) added)
        (PEPit) Setting up the problem: interpolation conditions for 2 function(s)
                         function 1 : Adding 144 scalar constraint(s) ...
                         function 1 : 144 scalar constraint(s) added
                         function 2 : Adding 121 scalar constraint(s) ...
                         function 2 : 121 scalar constraint(s) added
        (PEPit) Compiling SDP
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal_inaccurate (solver: SCS); optimal value: 0.020649148184166164
        (PEPit) Postprocessing: 3 eigenvalue(s) > 0.003245910668057083 before dimension reduction
        (PEPit) Calling SDP solver
        (PEPit) Solver status: optimal_inaccurate (solver: SCS); objective value: 0.02124334648210737
        (PEPit) Postprocessing: 3 eigenvalue(s) > 0.002134191248999246 after 1 dimension reduction step(s)
        (PEPit) Solver status: optimal_inaccurate (solver: SCS); objective value: 0.02124334648210737
        (PEPit) Postprocessing: 3 eigenvalue(s) > 0.002134191248999246 after dimension reduction
        *** Example file: worst-case performance of the Dykstra projection method ***
                PEPit example:   ||Proj_Q1 (xn) - Proj_Q2 (xn)||^2 == 0.0212433 ||x0 - x_*||^2

    """

    # Instantiate PEP
    problem = PEP()

    # Declare the two indicator functions and the feasibility problem
    ind_Q1 = problem.declare_function(ConvexIndicatorFunction)
    ind_Q2 = problem.declare_function(ConvexIndicatorFunction)
    func = ind_Q1 + ind_Q2

    # Start by defining a solution xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Run the Dykstra projection method
    x = x0
    y = x0
    p = null_point
    q = null_point
    for _ in range(n):
        y, _, _ = proximal_step(x + p, ind_Q1, 1)
        p = x + p - y
        x, _, _ = proximal_step(y + q, ind_Q2, 1)
        q = y + q - x

    # Set the performance metric
    proj1_x, _, _ = proximal_step(x, ind_Q1, 1)
    proj2_x = x
    problem.set_performance_metric((proj2_x - proj1_x) ** 2)
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose, dimension_reduction_heuristic="logdet1")
    theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Dykstra projection method ***')
        print('\tPEPit example:\t ||Proj_Q1 (xn) - Proj_Q2 (xn)||^2 == {:.6} ||x0 - x_*||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_dykstra(n=10, verbose=1)

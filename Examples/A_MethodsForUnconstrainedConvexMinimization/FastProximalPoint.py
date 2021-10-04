import numpy as np

from PEPit.pep import PEP
from PEPit.Function_classes.convex_function import ConvexFunction
from PEPit.Primitive_steps.proximal_step import proximal_step


def wc_fppa(A0, gammas, n):
    """
    In this example, we use the fast proximal point method of Guler [1] for
    solving the non-smooth convex minimization problem
    min_x F(x); for notational convenience we denote xs=argmin_x F(x);

    [1] O. Güler. New proximal point algorithms for convex minimization.
        SIAM Journal on Optimization, 2(4):649–664, 1992.

    We show how to compute the worst-case value of F(xN)-F(xs) when xN is
    obtained by doing N steps of the method starting with an initial
    iterate satisfying f(x0)-f(xs)+A/2*||x0-xs||^2<=1 for some A>0.

    Alternative interpretations:
    (1) the following code compute the solution to the problem
        max_{F,x0,...,xN,xs} (F(xN)-F(xs))/( f(x0)-f(xs)+A/2*||x0-xs||^2 )
            s.t. x1,...,xN are generated via Guler's method,
            F is closed, proper, and convex.
    where the optimization variables are the iterates and the convex
    function F.

    (2) the following code compute the smallest possible value of
    C(N, step sizes) such that the inequality
    F(xN)-F(xs)  <= C(N, step sizes) * ( f(x0)-f(xs)+A/2*||x0-xs||^2 )
    is valid for any closed, proper and convex F and any sequence of
    iterates x1,...,xN generated by Guler's method on F.

    :param A0: (float) intial value of A0.
    :param gammas: (list) step size.
    :param n: (int) number of iterations.

    :return:
    """

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(ConvexFunction, {})

    # Start by defining its unique optimal point
    xs = func.optimal_point()
    fs = func.value(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition(func.value(x0) - fs + A0 / 2 * (x0 - xs) ** 2 <= 1)

    # Run the GD method
    x = x0
    v = x0  # second sequence of iterates
    A = A0

    for i in range(n):
        alpha = (np.sqrt((A * gammas[i]) ** 2 + 4 * A * gammas[i]) - A * gammas[i]) / 2
        y = (1 - alpha) * x + alpha * v
        x, _, _ = proximal_step(y, func, gammas[i])
        v = v + 1 / alpha * (x - y)
        A = (1 - alpha) * A

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(func.value(x) - fs)

    # Solve the PEP
    wc = problem.solve()

    # Return the rate of the evaluated method
    # Theoretical guarantee (for comparison)
    accumulation = 0
    for i in range(n):
        accumulation += np.sqrt(gammas[i])
    theory = 4 / A0 / accumulation ** 2

    print('*** Example file: worst-case performance of the fast proximal point in function values ***')
    print('\tPEP-it guarantee:\t\t\t\t f(y_n)-f_* <= ', wc)
    print('\tTheoretical upper guarantee:\t f(y_n)-f_* <= ', theory)
    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return wc


if __name__ == "__main__":
    n = 3
    A0 = 5
    gammas = [(i + 1) / 1.1 for i in range(n)]

    wc = wc_fppa(A0, gammas, n)

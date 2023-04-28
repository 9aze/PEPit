from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import inexact_gradient_step
import numpy as np
from PEPit import Point


def wc_EF1(L, mu, epsilon, n, verbose=1):

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    Leps = (1 + epsilon) * L
    meps = (1 - epsilon) * mu
    gamma = 1 / (L + mu)

    # Then define the starting point x0 of the algorithm
    # as well as corresponding inexact gradient and function value g0 and f0
    e=problem.set_initial_point()
    x0 = problem.set_initial_point()
    beta=gamma*L /np.sqrt(1-gamma*L)
    coeff=(1-L*gamma)/(2*gamma*epsilon **2*(1+1/beta))
    # Set the initial constraint that is the distance between f0 and f_*
    
    problem.set_initial_condition(func(x0-e) - fs +coeff*e**2 <= 1)

    # Run n steps of the inexact gradient method
    

    x = x0
    
    for i in range(n):
        
        # Get the gradient gx0 and function value fx0 of f in x0.
        gx, fx = func.oracle(x)

        # Define dx0 as a proxy to gx0.
        dx = Point()
        func.add_constraint((gamma*gx+e - dx) ** 2 - epsilon ** 2 * ((gamma*gx+e) ** 2) <= 0)

        # Perform an inexact gradient step in the direction dx0.
        x = x - dx
        e=e+gamma*gx-dx



    x_hat=x-e
    
    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func(x_hat) - fs+coeff*e**2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = (1-min(gamma* mu /2, 1-epsilon**2*(1+gamma*L/np.sqrt(1-gamma*L))**2))**n

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of inexact gradient method in distance in function values ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} (f(x_0)-f_*)'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_EF1(L=1, mu=.1, epsilon=.1, n=1, verbose=1)



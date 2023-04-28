from PEPit.examples.unconstrained_convex_minimization import inexact_gradient_descent_EF1
import numpy as np
import matplotlib.pyplot as plt
L=1
mu=0.1
epsilon=list(np.linspace(0.1,0.3,50))
n=2
pepit_tau, theoretical_tau=[], []
for epsilon_ in epsilon:
    print(epsilon_)
    a,b=inexact_gradient_descent_EF1.wc_EF1(L, mu, epsilon_, n, verbose=1)
    pepit_tau.append(a)
    theoretical_tau.append(b)

plt.plot(epsilon, pepit_tau)
plt.plot(epsilon, theoretical_tau)
plt.show()
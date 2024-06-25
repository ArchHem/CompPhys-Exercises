import numpy as np
import matplotlib.pyplot as plt

def ADAM(func,v0,mu=1e-3,b1=0.9,b2=0.999,dx=1e-9,de=1e-9,maxiter=400,grad_crit = 1e-9):
    """
    ADAM schoastic gradient descent algo
    :param func: scalar function to be minimized, shape (N,1)
    :param v0: starting point
    :param mu: learning rate
    :param b1: beta 1 parameter, i.e. forget parameter
    :param b2: beta 2 parameter, i.e. gradient^^ forget parameter
    :param dx: dx used in gradient definition
    :param de: numerical divisor corrector
    :param maxiter: maximum  umber of iterations
    :param grad_crit: gradient norm criteria under which we assume convergence
    :return: list of locations the algorithm visited
    """

    N = v0.shape[0]

    def gradient(vec):
        grad = np.zeros((N,1))
        for i in range(N):
            diffvec = np.zeros((N,1))
            diffvec[i] = dx
            grad[i] = (func(vec+diffvec)-func(vec-diffvec))/(2*dx)
        return grad

    locs = [v0]
    vec = v0
    mprev = 0
    vprev = 0

    for i in range(1,maxiter):
        print(i/maxiter)
        grad = gradient(locs[-1])
        if np.amin(np.abs(grad)) <= grad_crit:
            """No hessian for now,implement it here"""
            break
        m = b1*mprev + (1 - b1) * grad
        v = b2*vprev + (1 - b2) * grad ** 2
        m_norm = m/(1-b1**i)
        v_norm = v/(1-b2**i)
        vec = vec - mu*m_norm/(np.sqrt(v_norm+de))
        locs.append(vec)
        mprev = m
        vprev = v


    return locs

def parabola(vec):

    return np.sum(vec**2)
v0 = np.array([[1],[2]])
locs = ADAM(parabola,v0,mu=1e-2,maxiter=800)
N = len(locs)
locs = np.asarray(locs)
locs = locs.reshape(N,2)

plt.plot(locs[:,0],locs[:,1])
plt.show()
















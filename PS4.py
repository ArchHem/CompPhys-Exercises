import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def simpson_integrator_full_step(h,lb,hb,func):
    """
    Simpson's integrator for sinc(x^2) x, o explanation required.
    :param h: step size/quadratic interpolation limit
    :param lb: lower integral bound
    :param hb: higher integral limit
    :param func: function to integrate
    :return: Value of integral
    """

    x_samples_1 = np.arange(lb, hb, h)
    x_samples_2 = h + x_samples_1
    x_samples_midpoint = h/2 + x_samples_1

    y = h/6 * (func(x_samples_1) + 4 * func(x_samples_midpoint) + func(x_samples_2))

    result = np.sum(y)

    return result

def congruent_rand(x0,a,b,N,c=0):
    """
    Conguent random number generator, used for exercise 1. Returns a array of shape (N,) of generated 'random' numbers
    :param x0: starting iteration. Should be an integer
    :param a: Parameter a, linear term
    :param b: Parameter b, modulo term
    :param N: Length of returned array
    :return: Integer array of shape (N,)
    """

    output = np.zeros((N),dtype='int')

    output[0] = x0

    for i in range(1,N):
        output[i] = (a*output[i-1] + c) % b


    return output



"""
if b = 12: it is NOT possible to choose a st. we improve the iteration length: for reference, see:

https://en.wikipedia.org/wiki/Linear_congruential_generator#Period_length

12 is a multiple of 4 thus has 2^2 in its prime decomposition

This might be wrong because of the first criteria though
"""

N = 100000
x = np.random.uniform(size=N)


def transform_sin(x):
    """
    Use method in lecture notes ot transform randomly generated variables to sin-like pdf on 0 to pi
    :param x: uniform variable on range 0-1.
    :return: transformed variable y
    """

    y = np.arccos(1-2*x)

    """normalize done outside"""

    return y

def transform_linear(x,a=2):
    """
    Uses inversion method to get PDF of the form 2/a^2 x
    :param x: random, uniform variable on range 0 - 1
    :param a: See above
    :return: New random variable distributed according to above PDF
    """
    y = np.sqrt(a**2 * x)
    return y

def transform_power(x):
    """
    Uses inversion method to get PDF of 1/x between 1 and e
    :param x: randomly distributed variable between 0 and 1
    :return: New random variable distributed according to new PDF defined above
    """

    y = np.exp(x)

    return y
"""
X = np.random.uniform(0,1,100000)
counts, bins = np.histogram(transform_power(X),bins = 25)
normalization = np.sum(np.diff(bins)*counts)
plt.stairs(counts/normalization,bins,color = 'red')
plt.hist(bins[:-1],bins,weights = counts/normalization,rwidth = 0.9)
y = np.linspace(1,np.exp(1),1000)
plt.plot(y,1/y)
plt.show()
"""


def rejection_method(PDF,a,b,add=0.1,N=1000,N_wanted = 1000):
    """
    Rejection method based random variable generator.
    :param PDF: Probabilizy distribution being generated, should be a ufunc
    :param a: Lower bound of PDF's range
    :param b: Higher bounds of PDF's range
    :param add: Numerical value being added to numerically determined maximum value of the PDF
    :param N: Number of sample points for PDF's maximum value detection
    :param N_wanted: Number of wanted points
    :return: Random variable distributed according to the PDF
    """

    y = np.linspace(a,b,N)

    maxval = np.amax(PDF(y)) + add

    res = []

    count = 0

    while True:


        x1 = np.random.uniform(a,b,1)
        x2 = np.random.uniform(0,maxval,1)

        if PDF(x1) >= x2:
            res.append(x1)
            count = count + 1
            if count >= N_wanted:
                break


    return res


"""
def PD(x):
    return np.sin(x)**2 / (pi/2)


counts, bins = np.histogram(rejection_method(PD,0,pi,0.01,N_wanted=100000),bins = 25)
normalization = np.sum(np.diff(bins)*counts)
plt.stairs(counts/normalization,bins,color = 'red')
plt.hist(bins[:-1],bins,weights = counts/normalization,rwidth = 0.9)
y = np.linspace(0,pi,1000)
plt.plot(y,PD(y))
plt.show()
"""

def one_d_monte_carlo(func,a,b,N=1000):
    """
    Integrates function x from a to b use MC method with N samples
    :param func: Function to integrate, ufunc type
    :param a: Lower integral bound
    :param b: Higher integral bound
    :param N: Number of samples
    :return: [Integral estimate, Error estimate]
    """

    x = np.random.uniform(a,b,N)

    vals = func(x)

    V = b-a

    integral_est = V/N * np.sum(vals)

    mean = np.average(vals)

    sigma_square = np.sum((vals-mean)**2)/(N-1)

    error_estimate = V*np.sqrt(sigma_square)/np.sqrt(N)

    return [integral_est,error_estimate]

"""
def locfunc(x):
    return np.sin(x**2)/x

int_val_true = 0.781112733444528
vals, errs = [],[]
xvals = [2**i for i in range(1,17)]
for i in xvals:
    intval_est, int_val_err = one_d_monte_carlo(locfunc, 0, 10, i)

    vals.append(intval_est)
    errs.append(int_val_err)

vals = np.array(vals)
errs = np.array(errs)

plt.plot(xvals,int_val_true*np.ones_like(vals))
plt.scatter(xvals,vals)
plt.errorbar(xvals,vals,yerr=errs,capsize = 4,color='green',ls='none')
plt.xscale('log')
plt.show()
"""

def hyper_sphere_integrator(N,Ndims,R=1):
    """
    Determines volume of N-dimensional sphere
    :param N: Number of samples drawn per dimension
    :param Ndims: Number of dimensions
    :param R: Radius of sphere
    :return: Predicted volume and error bound
    """

    """technically we could just generate from 0 to R and adjust volume by 2^Ndims"""

    x = np.random.uniform(-R,R,N*Ndims)

    x = x.reshape((Ndims,N))

    radii = np.sqrt(np.sum(x**2,axis=0))

    vals = np.where(radii <= R,1,0)

    V = (2*R)**Ndims

    integral_est = V / N * np.sum(vals,axis=-1)

    mean = np.average(vals,axis=-1)

    sigma_square = np.sum((vals - mean) ** 2,axis=-1) / (N - 1)

    error_estimate = V * np.sqrt(sigma_square) / np.sqrt(N)

    return [integral_est,error_estimate]

def hypercube_integrator(N,Ndims,D: float=1 ,vol_bord = 2):
    """
        Determines volume of N-dimensional cube
        :param N: Number of samples drawn per dimension
        :param Ndims: Number of dimensions
        :param D: Half-width of cube: can be an (N,1) shaped array
        :param vol_bord: half-width of integration volume
        :return: Predicted volume and error bound
        """

    """technically we could just generate from 0 to R and adjust volume by 2^Ndims"""

    x = np.random.uniform(-vol_bord, vol_bord, N * Ndims)

    x = x.reshape((Ndims, N))

    cond = D

    vals = np.where(np.abs(x) <= cond, 1, 0)

    """dark magic"""

    vals = np.sum(vals,axis=0)

    vals = np.where(vals==Ndims,1,0)

    V = (2 * vol_bord) ** Ndims

    integral_est = V / N * np.sum(vals, axis=-1)

    mean = np.average(vals, axis=-1)

    sigma_square = np.sum((vals - mean) ** 2, axis=-1) / (N - 1)

    error_estimate = V * np.sqrt(sigma_square) / np.sqrt(N)

    return [integral_est, error_estimate]

"""
N = 100000
R = np.linspace(0,3,60)
vals = []
for D in range(1,7):
    vals.append([])
    for r in R:
        vals[D-1].append(hyper_sphere_integrator(N,D,r)[0])

vals = np.array(vals)
for count, rows in enumerate(vals):
    plt.plot(R,rows,label = '%s-th Dimensional sphere volume' %(str(count)))

plt.xlabel('Radii')
plt.ylabel('Volume')
plt.legend(loc='upper left')
plt.show()
"""

def MetroGaussian(xt,maxiter,distr,sigma: float = 1):
    """
    Generates random samples of distr ufunc
    N- D Metropolitian-Hasting algorthm using a Normal distribution as guidance
    :param xt: original guess
    :param maxiter: Maximum number of iterations
    :param sigma: Sigma of Gaussian
    :param distr: normalized distribution according to which we sample
    :return: values of whether random walk steps are in sphere of radius R or not
    """

    def standard_distr(x,y,o):
        """Bit of a cheating, but rejection algo would clip of tails of the Gaussians"""

        return o*np.random.randn(*x.shape) + y

    """Solved - PDF needs to be normalized"""

    vals = []

    count = 0

    for i in range(maxiter):

        x_pot = standard_distr(xt,xt,sigma)

        alpha = distr(x_pot)/distr(xt)

        """this legitimately looks ugly, how to vectorize?"""

        u = np.random.uniform(0,1)

        if u <= alpha:
            xt = x_pot
            count = count + 1

        vals.append(xt)

    vals = np.array(vals)
    return vals, count, maxiter

def sphere(x,R=0.5):

    return np.where(R <= np.sqrt(np.sum(x**2,axis = -1)),0,1)

def norm_distr(x,sigma=1):

    """We need to normalize it to N dimensions"""

    k = x.shape[-1]

    return np.exp(-(np.sum(x**2,axis=-1)/sigma**2)/2) * 1/(sigma*np.sqrt((2*pi)**k))

def Q_func(x):

    return sphere(x)/norm_distr(x)


"""q, count, maxiter = MetroGaussian(np.array([0,0]),100000,norm_distr,sigma = 0.1)

z = sphere(q)

norm = plt.Normalize(vmin=0, vmax=1)
plt.scatter(q[:,0],q[:,1],s=0.5, c = z, cmap=mcolors.ListedColormap(["black", "green"]))
y = Q_func(q)

predicted_value = np.mean(y)

act_val = pi * (0.5)**2

print(predicted_value/act_val)

plt.show()"""


def mean_field_approx_updater(T,H,s0,maxiter = 1000):
    """Iterative solver for mean spin in Ising model. All non specified variables are taken as natural units.
    :param T: temperature of the system
    :param H: External H field
    :param s0: starting value: should be symetric about 0, i.e. different parities should converge to same values under
    parity transform
    :param maxiter: number of iterations
   """

    s_mean_update = s0

    #vals = []

    for i in range(maxiter):
        #vals.append(s_mean_update)
        s_loc = np.tanh(1 / T * (H + s_mean_update))

        s_mean_update = s_loc


    return s_mean_update #,vals

"""
plt.plot(mean_field_approx_updater(0.1,0,0.5)[1],lw=1.5,color='green',ls = 'dashed')
plt.ylim(-1.2,1.2)
plt.xlabel('Iteration cycle index')
plt.ylabel('Mean spin')
plt.grid()
plt.show()
"""

"""
T = np.linspace(0.01,3,1000)
dT = np.diff(T)[0]
H = 0
s = mean_field_approx_updater(T,H,0.5,maxiter=2000)
E = -(H+s)*s
E_deriv_use = np.pad(E,pad_width=1,mode='edge')
Heat_cap = (-E_deriv_use[:-2]/2 + E_deriv_use[2::]/2)/dT
plt.plot(T,s)
plt.plot(T,E)
plt.plot(T,Heat_cap)
"""

"""
fig, ax = plt.subplots()
H_var = np.linspace(-2,2,1000)
dH = np.diff(H_var)[0]
s_bellow = mean_field_approx_updater(0.5,H_var,0.5)

s_above = mean_field_approx_updater(1.5,H_var,0.5)
E_above = -(H_var+s_above)*s_above
E_deriv_use_above = np.pad(E_above,pad_width=1,mode='edge')
Heat_cap_above = (-E_deriv_use_above[:-2]/2 + E_deriv_use_above[2::]/2)/dH


E_bellow = -(H_var+s_bellow)*s_bellow
E_deriv_use_bellow = np.pad(E_bellow,pad_width=1,mode='edge')
Heat_cap_bellow = (-E_deriv_use_bellow[:-2]/2 + E_deriv_use_bellow[2::]/2)/dH


ax.plot(H_var,E_bellow)
ax.plot(H_var,E_above)
ax.plot(H_var,Heat_cap_bellow)
ax.plot(H_var,Heat_cap_above)
plt.show()
"""

def grid_structure_iterator(N,T,H):

    vals = np.array([-1,1])

    gridvals = np.random.choice(vals,size=(N,N))

    """create BC-s"""

    gridvals_full = np.pad(gridvals, ((1,1),(1,1)), 'constant', constant_values=((0,0),(0,0)))

    def boltzman(E):

        return np.exp(-E/T)

print(grid_structure_iterator(5,10,1))

























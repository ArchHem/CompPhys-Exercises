import numpy as np
import matplotlib.pyplot as plt

def naive_lagrange_poly(x,xn,yn):
    """
    Lagrange polynomial, naive implementation
    :param x: variable x, where we evaluate
    :param xn: x-coordinate location of interpolation points
    :param yn: y-coordinate location of interpolation points
    :return: lagrange polynomial at location x
    """
    N = xn.shape[0]

    sum = 0
    for i in range(N):
        loc_list = [u for u in range(N)]
        loc_list.pop(i)
        product = 1
        for j in loc_list:
            product = product*(x-xn[j])/(xn[i]-xn[j])
        sum = sum + product*yn[i]
    return sum

def efficient_lagrange_poly_weights(xn):
    """
    Computes the weights used to run the algorithm, allowing subsequent O(n) Lagrange polynomial implementation. Ref: https://people.maths.ox.ac.uk/trefethen/barycentric.pdf
    :param xn: x location of datapoints, shape N
    :return: node weights used in computing algorithm, shape N
    """
    N = xn.shape[0]
    """use vectorization"""

    xn_loc = xn.reshape((N,1))

    deltaX = xn_loc - xn_loc.T
    deltaX[np.diag_indices(N)] = 1
    mult = np.prod(deltaX,axis=1)
    res = 1/mult
    return res

def lagrange_poly_comp(x,xn,yn,wn):
    """
    Computes lagrange polynomial given pre-computed weights w_n
    :param x: array of shape N, polynomial is computed here
    :param xn: x location of datapoints
    :param yn: y location of datapoints
    :param wn: pre-computed weights. If set to True, computes it locally
    :return: computed values of polynomial
    """

    N = yn.shape[0]

    if type(wn) == type(True):
        if wn:
            wn = efficient_lagrange_poly_weights(xn)

    l_val = np.array([(x - xi) for xi in xn])

    l_x = np.prod(l_val,axis=0)

    sum = 0

    for i in range(N):
        sum = sum + wn[i]*yn[i]/(x-xn[i])

    sum = np.nan_to_num(sum)

    res = sum*l_x

    return res

def finite_diff_coeff_calc(order,deriv_order):
    """
    Calculates coefficients of a central, finite different scheme for given derivative order and O(h^n) order. Via https://en.wikipedia.org/wiki/Finite_difference_coefficient
    :param order: O(h^n) order error
    :param derivative_order: d^n/dx^n being approximated
    :return: coefficient array, N long, flattened. Central element belongs to y(x),
    central + 1 to y(x + 1h), central - 1 to y(x-1h) etc.
    """

    p = int((2*np.floor((deriv_order+1)/2) - 2 + order)/2)

    mat_size = 2*p + 1

    sample_row = np.array(range(-p,p+1)).reshape((1,mat_size))

    powers = np.array(range(mat_size)).reshape((mat_size,1))

    """vectorization magic"""

    to_solve_mat = sample_row**powers

    to_solve_mat = to_solve_mat.astype(float)

    result_vec = np.zeros((mat_size,1),dtype='float64')

    result_vec[deriv_order] = np.math.factorial(deriv_order)

    solution_vector = np.linalg.solve(to_solve_mat,result_vec)

    return solution_vector.flatten()



def sinc_integrator_full_step(h,lb,hb):
    """
    Trapezoid integrator for sinc(x^2) x, o explanation required.
    :param h: setp size
    :param lb: lower integral bound
    :param hb: higher integral limit
    :return: Value of integral
    """

    def sinc(x):
        return np.sin(x**2)/x

    x_samples_1 = np.arange(lb,hb,h)

    x_samples_2 = x_samples_1 + h

    y_1 = sinc(x_samples_1)

    y_2 = sinc(x_samples_2)

    result = np.sum((y_1+y_2)/2) *h

    return result

def simpson_integrator_full_step(h,lb,hb):
    """
    Simpson's integrator for sinc(x^2) x, o explanation required.
    :param h: step size/quadratic interpolation limit
    :param lb: lower integral bound
    :param hb: higher integral limit
    :return: Value of integral
    """

    def sinc(x):
        return np.sin(x ** 2) / x

    x_samples_1 = np.arange(lb, hb, h)
    x_samples_2 = h + x_samples_1
    x_samples_midpoint = h/2 + x_samples_1

    y = h/6 * (sinc(x_samples_1) + 4 * sinc(x_samples_midpoint) + sinc(x_samples_2))

    result = np.sum(y)

    return result



def sinc(x):
    return np.sin(x**2)/x

def FFT(x):
    """
    Returns FT of array.
    :param x: 2^N  long array of shape (2^N,): if not, it will not return anything.
    :return: DFT of array x
    """

    def is_power_of_two(n):
        """from: https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two"""
        return (n != 0) and (n & (n - 1) == 0)

    N = x.shape[0]
    output = np.zeros_like(x,dtype='complex128')
    if output.shape[0] == 1:
        return x

    if is_power_of_two(N) != True:
        raise TypeError('Array is not of power of two length, exiting.')

    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)

        X = np.concatenate(
            [X_even + factor[:int(N / 2)] * X_odd,
             X_even + factor[int(N / 2):] * X_odd])
        return X

sr = 256
x = np.linspace(0,1,sr)

y = np.sin(2*np.pi*x*4) + 0.6*np.sin(2*np.pi*x*16)

X = FFT(y)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T

plt.plot(freq,np.real(X))
plt.show()


"""
zero_to_ten_int_val = 0.781112733444528

dx1 = 1
dx2 = 1e-2
dx3 = 1e-4

correction1 = (sinc(dx1/2) + sinc(dx1))*dx1/2

correction2 = (sinc(dx2/2) + sinc(dx2))*dx2/2

correction3 = (sinc(dx3/2) + sinc(dx3))*dx3/2

correction4 = simpson_integrator_full_step(dx1/2,dx1/2,dx1)

correction5 = simpson_integrator_full_step(dx2/2,dx2/2,dx2)

correction6 = simpson_integrator_full_step(dx3/2,dx3/2,dx3)


res1 = sinc_integrator_full_step(dx1,dx1,10)
res2 = sinc_integrator_full_step(dx2,dx2,10)
res3 = sinc_integrator_full_step(dx3,dx3,10)

res4 = simpson_integrator_full_step(dx1,dx1,10)
res5 = simpson_integrator_full_step(dx2,dx2,10)
res6 = simpson_integrator_full_step(dx3,dx3,10)

print(res1-zero_to_ten_int_val,res2-zero_to_ten_int_val,res3-zero_to_ten_int_val)
print(res1+correction1-zero_to_ten_int_val,res2+correction2-zero_to_ten_int_val,res3+correction3-zero_to_ten_int_val)
print(res4-zero_to_ten_int_val,res5-zero_to_ten_int_val,res6-zero_to_ten_int_val)
print(res4+correction4-zero_to_ten_int_val,res5+correction5-zero_to_ten_int_val,res6+correction6-zero_to_ten_int_val)
"""

"""
print(finite_diff_coeff_calc(2,2))
"""

"""
N = 30
x_discr = np.linspace(-2,2,N)
y_discrete = np.exp(-x_discr**2)

x_cont = np.linspace(-2,2,200)
w_locs = efficient_lagrange_poly_weights(x_discr)

y_cont0 = lagrange_poly_comp(x_cont,x_discr,y_discrete,w_locs)
y_cont = naive_lagrange_poly(x_cont,x_discr,y_discrete)

plt.plot(x_cont,y_cont,color = 'red',lw = 0.9)
plt.plot(x_cont,y_cont,color = 'blue',lw = 0.9)
plt.scatter(x_discr,y_discrete,s=3,color='green')
plt.show()
"""




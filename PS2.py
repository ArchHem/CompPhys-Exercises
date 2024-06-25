import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sci
import warnings

"""Exercise one - LU decomposition"""

"""
Matrix_1 = 1/6 * np.array([[5.0,4,3,2,1],[4,8,6,4,2],[3,6,9,6,3],[2,4,6,8,4],[1,2,3,4,5]])

b = np.array([0,1,2,3,4]).reshape((5,1))
"""

def LU_solver(M,b):
    """
    LU decomposition based solver
    :param M: matrix of variables
    :param b: Equals M@x, vector of output
    :return: solves for x
    """
    L = np.zeros_like(M)
    U = np.zeros_like(M)
    N = L.shape[0]

    """set diagonal value to 1"""
    for i in range(N):

        for j in range(N):

            s1 = M[i,j]

            s2 = M[i,j]/U[j,j]

            for k in range(i):
                s1 = s1 - U[k,j]*L[i,k]

            for k in range(j):
                s2 = s2 - U[k,j]*L[i,k]/U[j,j]

            L[i,j] = s2
            U[i,j] = s1
        L[i, i] = 1.0

    L = np.nan_to_num(L)
    U = np.nan_to_num(U)

    corrector_u = np.zeros_like(M)
    corrector_l = np.zeros_like(M)

    corrector_u[np.triu_indices(N)] = 1
    corrector_l[np.tril_indices(N)] = 1

    L = corrector_l*L
    U = corrector_u*U


    """Solve L @ y = b where y = U @ x
    Use forward substitution"""

    y = np.zeros_like(b,dtype='float64')



    for i in range(N):

        temp = (b[i] - L[i:i+1].dot(y))/L[i,i]

        y[i] = temp

    """Now solve U @ x = y"""
    x = np.zeros_like(b, dtype='float64')

    for i in range(1,N+1):

        temp = (y[-i] - U[-i].dot(x.flatten()))/U[-i,-i]

        x[-i] = temp



    return [x, L, b]



"""test different cases as well"""
"""
M2 = np.random.randint(1,10,(5,5))
M2 = M2.astype('float')


solution = sci.solve(M2,b)
x, L, U = LU_solver(M2,b)
print(solution-x)
"""

"""Exercise 2 - method of powers"""

"""
N_shape = 5
M_2 = np.zeros((N_shape,N_shape))
M_3 = (3 + np.random.rand(N_shape,N_shape))**3
diag_val = -10

rows, cols = np.indices((N_shape,N_shape))

row_upper = np.diag(rows, k=1)
col_upper = np.diag(cols, k=1)
row_lower = np.diag(rows, k=-1)
col_lower = np.diag(cols, k=-1)

M_2[row_upper,col_upper] = diag_val
M_2[row_lower,col_lower] = diag_val
"""

"""implement method of powers"""

def method_of_power(A, num_it: int = 600, conv_rat = 1e-5, min_cycles = 200):
    """
    uses randomly selected orthonormal basis to isolate the largest magnitude eigenvalue

    :param A: Square matrix whose maximum eigenvalue we seek
    :param num_it: maximum number of iterations
    :param conv_rat: convergence ratio to stop
    :return: maximum eigenvalue's approximation.
    """
    N = A.shape[0]

    """normalize A"""

    C = N * np.amax(np.abs(A))
    A = A/C

    basis = []

    for i in range(N):
        v_loc = np.random.rand(N,1)
        v_loc = v_loc/np.linalg.norm(v_loc)
        basis.append(v_loc)

    new_basis = []

    for i in range(N):
        new_basis.append(np.zeros((N,1)))

    """generate orthonormal basis"""

    for i in range(N):
        loc_orth = basis[i]
        for j in range(i):
            loc_orth = loc_orth - new_basis[j].T.dot(basis[i]) * new_basis[j]
        new_basis[i] = loc_orth/np.linalg.norm(loc_orth)

    num_of_iters = []
    max_eig_found = []
    min_error = []

    for vec in new_basis:
        v1 = vec
        iter_count = 0
        while True:
            iter_count = iter_count + 1
            v0 = v1
            v1 = A @ v0
            v2 = A @ v1

            prev_eig = v0.T.dot(v1)/v0.T.dot(v0)
            loc_eig = v1.T.dot(v2)/v1.T.dot(v1)

            """it is better to use total error here to lessen numerical mistakes here"""
            rel_error = np.abs(np.abs(prev_eig-loc_eig))

            if min_cycles < iter_count:
                if rel_error < conv_rat:
                    break
                if num_it < iter_count:
                    break



        num_of_iters.append(iter_count)
        max_eig_found.append(loc_eig)
        min_error.append(rel_error)

    max_eig_found = np.array(max_eig_found)

    id = np.argmax(np.abs(max_eig_found))

    lambda_val = max_eig_found[id]


    """shift eigenvalues according to normalization"""

    lambda_val = lambda_val*C

    lambda_val = np.nan_to_num(lambda_val)

    return float(lambda_val)

"""Exercise 3 - Jacobini method"""

"""
N_shape = 5
M_4 = np.zeros((N_shape,N_shape))
diag_val = -1
middle_val = 2

b = np.array([-1,0,0,0,5])
b = b.reshape((5,1))


rows, cols = np.indices((N_shape,N_shape))

row_upper = np.diag(rows, k=1)
col_upper = np.diag(cols, k=1)
row_lower = np.diag(rows, k=-1)
col_lower = np.diag(cols, k=-1)

M_4[row_upper,col_upper] = diag_val
M_4[row_lower,col_lower] = diag_val
M_4[np.diag(rows),np.diag(cols)] = middle_val

print(M_4)
"""

def Jacobi_solver(M,b,maxruns = 600,converge_limit = 1e-10):
    """
    Solves for x in M @ x = b
    :param M: Known matrix, ideally diagonally dominant
    :param b: known, right-hand solution vector
    :return: x, solution
    """
    N = M.shape[0]

    rows, cols = np.indices((N,N))

    U = np.triu(M,k=1)

    L = np.tril(M,k=-1)

    diagonal_vals = M[np.diag(rows),np.diag(cols)]

    D = np.zeros_like(M,dtype='float64')

    D[np.diag(rows),np.diag(cols)] = diagonal_vals

    D_inverse = np.divide(1,D,out=np.zeros_like(D), where=D!=0)

    x_guess = np.ones_like(b,dtype='float64')

    iter_matrix = (D_inverse @ (L + U))

    maxlambda = method_of_power(iter_matrix)

    if 1 < np.abs(maxlambda):
        raise ValueError('There exists an eigenvalue larger than one in magnitude, the method will not converge!')



    for i in range(maxruns):

        loc_ver = D_inverse @ (b - ((L+U) @ x_guess))

        diff_ammount = np.sum(np.abs(loc_ver - x_guess))

        x_guess = loc_ver

        if diff_ammount < converge_limit:
            break

    return x_guess


"""
N = 5
np.random.seed(0)
"""

def vec_field_zero_finder(field,dx=10e-5,conv_limit = 10e-12, maxruns0 = 200,maxruns1 = 100,dxratio = 100,Ndim = 2,v0 = True):
    """
    finds zero point of the field
    :param field: Input of shape N,1 should return array of N,1
    :param dx: first numerical delta
    :param conv_limit: convergence limit, to zero
    :param maxruns0: number of runs before switching to smaller dx
    :param maxruns1: number of runs with smaller dx
    :param dxratio: old dx's ratio to the new one
    :param Ndim: Number of dimensions of the field, denoted N
    :param v0: initial vector: True value uses the zero vector as a start
    :return:
    """
    N = Ndim
    def jacobi_matrix_inverse(field,vec,dr):
        output = np.zeros((N,N),dtype='float64')
        for i in range(N):
            for j in range(N):
                diffvec = np.zeros((N,1),dtype='float64')
                diffvec[j] = dr
                output[i,j] = (field(vec + diffvec)[i] - field(vec)[i])/dr


        return np.linalg.inv(output)

    if v0:
        v0 = np.zeros((N,1),dtype='float64')

    vector = v0.copy()

    to_proceed = True

    for i in range(maxruns0):
        v_loc = vector - jacobi_matrix_inverse(field,vector,dx) @ field(vector)
        diff = np.sqrt(np.sum(field(v_loc)**2))
        vector = v_loc
        if diff < conv_limit:
            to_proceed = False
            break

    if to_proceed:
        for i in range(maxruns1):
            v_loc = vector - jacobi_matrix_inverse(field, vector, dx/dxratio) @ field(vector)
            diff = np.sqrt(np.sum(field(v_loc)**2))

            vector = v_loc
            if diff < conv_limit:
                break
            if i == maxruns1 -1:
                warnings.warn('Maximum iteration count reached in second cycle')

    return vector

def electric_field(vec):
    angle = 2*np.pi/3
    charges = [1,0.5,1]
    vectors = []
    for i in range(3):
        vi = np.array([[np.cos(angle*i)],[np.sin(angle*i)]])
        vectors.append(vi)
    strength = np.zeros((2,1),dtype = 'float64')

    for i in range(3):

        strength = strength + (charges[i] * (vec-vectors[i])) / (np.sum((vec-vectors[i])**2) ** (3/2))

    return strength

"""
vector = np.array([[0],[1]])

zero_vec = vec_field_zero_finder(electric_field)

print(electric_field(zero_vec))
"""


"""Task 5-ish: Gauss-Seidel"""

def triang_inverter(M,ori = 'Up'):
    """
    Inverts M, where it is some triangular, non-singular matrix
    :param M: NxN, upper or lower triangular matrix
    :param ori: 'Up' or 'Low' string, sets orientation
    :return: M^-1
    """
    N = M.shape[0]

    if ori == 'Up':
        vectors_solved = []
        for j in range(N):
            x = np.zeros((N,1),dtype='float64')
            y = np.zeros((N, 1), dtype='float64')
            y[j] = 1


            for i in range(1, N + 1):
                temp = (y[-i] - M[-i].dot(x.flatten())) / M[-i, -i]

                x[-i] = temp
            vectors_solved.append(x)

    if ori == 'Low':
        vectors_solved = []
        for j in range(N):
            x = np.zeros((N, 1), dtype='float64')
            y = np.zeros((N, 1), dtype='float64')
            y[j] = 1

            for i in range(N):
                temp = (y[i] - M[i:i + 1].dot(x)) / M[i, i]

                x[i] = temp
            vectors_solved.append(x)
    else:
        TypeError('No matrix type picked, see description.')

    solution = vectors_solved[0]

    for i in range(1,N):
        solution = np.hstack((solution,vectors_solved[i]))
    return solution

def Gauss_Seidel(M,b,x0=True, maxruns = 200,conv_limit = 1e-9):
    """
    Uses Gauss-Seidel scheme to solve for x in: M @ x = b, optimized fro lower-diagonal systems
    :param M: see above
    :param b: see above
    :param x0: set to zero, will start with the N,1 1-filled vector if so. Can be set as pleased, starts initial x guess.
    :param maxruns: Maximum number of iterations
    :param conv_limit = residual convergence limit

    :return: solution vector x of the shape (N,1)
    """

    """try to normalize the system"""
    maxlambda = method_of_power(M)

    if 1 <= np.abs(maxlambda):
        warnings.warn('Largest magnitude eigenvalue is larger than one. Convergence not guaranteed.')

    N = M.shape[0]

    U = np.triu(M, k=1)

    L = np.tril(M, k=0)

    if x0:
        x0 = np.ones_like(b, dtype='float64')

    loc_x_guess = x0.copy()

    L_inv = triang_inverter(L,ori = 'Low')



    for i in range(maxruns):
        loc_x_guess_update = L_inv @ (b - (U @ loc_x_guess))
        residual = np.sqrt(np.sum(((M @ loc_x_guess_update) - b)**2))


        loc_x_guess = loc_x_guess_update
        if residual < conv_limit:
            break
        if i == maxruns - 1:
            warnings.warn('Maximum number of iterations reached')


    return loc_x_guess

def generalized_jacobi_method(M,b,x0=True, maxruns = 300,conv_limit = 1e-9,w=True):
    """
        Uses Gauss-Seidel scheme to solve for x in: M @ x = b
        :param M: see above
        :param b: see above
        :param x0: set to zero, will start with the N,1 1-filled vector if so. Can be set as pleased, starts initial x guess.
        :param maxruns: Maximum number of iterations
        :param conv_limit = residual convergence limit
        :param w: relaxing parameter. True results in autotuning.

        :return: solution vector x of the shape (N,1)
        """

    N = M.shape[0]

    maxlambda = method_of_power(M)

    if 1 <= np.abs(maxlambda):
        warnings.warn('Largest magnitude eigenvalue is larger than one. Convergence not guaranteed.')

    rows, cols = np.indices((N, N))

    U = np.triu(M, k=1)

    L = np.tril(M, k=-1)

    diagonal_vals = M[np.diag(rows), np.diag(cols)]

    D = np.zeros_like(M, dtype='float64')

    D[np.diag(rows), np.diag(cols)] = diagonal_vals

    """generate weighting matrix, lower triangular"""

    DL = (D + w*L)

    DLI = triang_inverter(DL,ori='Low')

    if x0:
        x0 = np.ones_like(b, dtype='float64')

    if w:
        D_inverse = np.divide(1, D, out=np.zeros_like(D), where=D != 0)
        max_w = 2/np.abs(method_of_power(D_inverse @ M))

        """factor based on stackexchange comment"""

        w = 2*max_w/3


    loc_x_guess = x0.copy()

    for i in range(maxruns):
        loc_x_guess_update = DLI @ (w*b - ((w*U + (w - 1)*D) @ loc_x_guess))
        residual = np.sqrt(np.sum(((M @ loc_x_guess_update) - b)**2))


        loc_x_guess = loc_x_guess_update
        if residual < conv_limit:
            break
        if i == maxruns - 1:
            warnings.warn('Maximum number of iterations reached')

    return loc_x_guess

"""
M_test = np.zeros((5,5))
rows, columns = np.indices((5,5))
M_test[np.diag(rows),np.diag(columns)] = 20

M_test[0,0] = 3.0
M_test[1,2] = 1.0


M_solution = np.random.rand(5,1)
print(M_test)
print(np.linalg.eig(M_test)[0])


print(np.linalg.solve(M_test,M_solution) - generalized_jacobi_method(M_test,M_solution))
"""









































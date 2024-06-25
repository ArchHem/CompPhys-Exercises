import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

def forward_euler_integrator_sin(dt,u0=0,v0=1,T=1,t0=0):
    """
    integrates d^2 u / dt^2 + u^2 = 0 between t0 and T, with dt timestep, using forward euler
    :param dt: Timestep
    :param u0: u at time t0
    :param v0: v, i.e. du/dt at time t0
    :param T: Maximum time
    :param t0: Beginning of time
    :return: timepoints, u_points, v_points
    """

    t = np.arange(t0,T+dt,dt)

    """we do not want to duplicate first timestep"""

    N = t.shape[0] - 1

    u = u0
    v = v0

    ul = [u]
    vl = [v]

    for i in range(N):
        u_loc = u + v*dt
        v_loc = v - u*dt

        u = u_loc
        v = v_loc

        ul.append(u)
        vl.append(v)

    ul, vl = np.array(ul), np.array(vl)

    return t, ul, vl

"""
T = 1
N = 6
dts = [0.1**i for i in range(N)]
fig, ax = plt.subplots(4)
x = np.linspace(0,T,1000)

for i, val in enumerate(dts):
    t, ul, vl = forward_euler_integrator_sin(val)
    ax[0].plot(t, ul, lw=0.8, label='dt = %.2e' %(val))
    ax[1].plot(t, vl, lw=0.8, label='dt = %.2e' %(val))
    ax[2].plot(t, vl**2 + ul**2, lw = 0.8, label='Energy for dt = %.2e' %(val))
    ax[3].plot(t, np.abs(np.sin(t) - ul), lw=0.8, label = 'Absolute error in u(t) for dt %.2e' %(val) )

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].set_ylabel('u(t)')
ax[1].set_ylabel('v(t)')
ax[2].set_ylabel('E(t)')
ax[3].set_ylabel(r'$\epsilon(t)$')
ax[0].plot(x,np.sin(x),lw=0.8,label='Exact, for N = 1000 subdivions')
ax[1].plot(x,np.cos(x),lw=0.8,label='Exact, for N = 1000 subdivions')

for a in ax:
    a.set_xlabel('t')
    a.legend(prop={'size': 6})

plt.show()
"""

def implicit_euler_sin(dt,u0=0,v0=1,T=1,t0=0):
    """
        integrates d^2 u / dt^2 + u^2 = 0 between t0 and T, with dt timestep, using implicit/backward euler
        :param dt: Timestep
        :param u0: u at time t0
        :param v0: v, i.e. du/dt at time t0
        :param T: Maximum time
        :param t0: Beginning of time
        :return: timepoints, u_points, v_points
    """

    t = np.arange(t0, T + dt, dt)

    """we do not want to duplicate first timestep"""

    N = t.shape[0] - 1


    x = np.array([[u0],[v0]])

    vev_list = x.copy()

    """write equation as x_dot = M @ x, where M = ((0,1),(-1,0))
    
    update happens as x_(n+1) = x_n + dt x_dot_n+1 = x_n + dt * M @ x_n+1: can be reorganized for x_(n+1)"""

    M = np.array([[0,1],[-1,0]])

    I = np.identity(2)

    updater = np.linalg.inv(I-dt*M)

    for i in range(N):

        x_n_1 = updater @ x

        vev_list = np.hstack((vev_list,x_n_1))

        x = x_n_1


    ul, vl = vev_list[0], vev_list[1]

    return t, ul, vl


"""
T = 1
N = 6
dts = [0.1**i for i in range(N)]
fig, ax = plt.subplots(4)
x = np.linspace(0,T,1000)

for i, val in enumerate(dts):
    t, ul, vl = forward_euler_integrator_sin(val)
    ax[0].plot(t, ul, lw=0.8, label='dt = %.2e' %(val))
    ax[1].plot(t, vl, lw=0.8, label='dt = %.2e' %(val))
    ax[2].plot(t, vl**2 + ul**2, lw = 0.8, label='Energy for dt = %.2e' %(val))
    ax[3].plot(t, np.abs(np.sin(t) - ul), lw=0.8, label = 'Absolute error in u(t) for dt %.2e' %(val) )

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].set_ylabel('u(t)')
ax[1].set_ylabel('v(t)')
ax[2].set_ylabel('E(t)')
ax[3].set_ylabel(r'$\epsilon(t)$')
ax[0].plot(x,np.sin(x),lw=0.8,label='Exact, for N = 1000 subdivions')
ax[1].plot(x,np.cos(x),lw=0.8,label='Exact, for N = 1000 subdivions')

for a in ax:
    a.set_xlabel('t')
    a.legend(prop={'size': 6})

plt.show()
"""

def perfect_sin_integrator(dt,u0=0,v0=1,T=1,t0=0):
    """
    integrates d^2 u / dt^2 + u^2 = 0 between t0 and T, with dt timestep, using a cheat timestep
    :param dt: Timestep
    :param u0: u at time t0
    :param v0: v, i.e. du/dt at time t0
    :param T: Maximum time
    :param t0: Beginning of time
    :return: timepoints, u_points, v_points
    """

    t = np.arange(t0, T + dt, dt)

    """we do not want to duplicate first timestep"""

    N = t.shape[0] - 1

    u = u0
    v = v0

    ul = [u]
    vl = [v]

    for i in range(N):
        u_loc = u*np.cos(dt) + v * np.sin(dt)
        v_loc = v*np.cos(dt) - u * np.sin(dt)

        u = u_loc
        v = v_loc

        ul.append(u)
        vl.append(v)

    ul, vl = np.array(ul), np.array(vl)

    return t, ul, vl

"""
T = 1
N = 6
dts = [0.1**i for i in range(N)]
fig, ax = plt.subplots(4)
x = np.linspace(0,T,1000)

for i, val in enumerate(dts):
    t, ul, vl = perfect_sin_integrator(val)
    ax[0].plot(t, ul, lw=0.8, label='dt = %.2e' %(val))
    ax[1].plot(t, vl, lw=0.8, label='dt = %.2e' %(val))
    ax[2].plot(t, vl**2 + ul**2, lw = 0.8, label='Energy for dt = %.2e' %(val))
    ax[3].plot(t, np.abs(np.sin(t) - ul), lw=0.8, label = 'Absolute error in u(t) for dt %.2e' %(val) )

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].set_ylabel('u(t)')
ax[1].set_ylabel('v(t)')
ax[2].set_ylabel('E(t)')
ax[3].set_ylabel(r'$\epsilon(t)$')

for a in ax:
    a.set_xlabel('t')
    a.legend(prop={'size': 6})

plt.show()
"""


def adam_bashforth_2_sin(dt,u0=0,v0=1,T=1,t0=0):
    """
    integrates d^2 u / dt^2 + u^2 = 0 between t0 and T, with dt timestep, using Adam-Bashforth method
        :param dt: Timestep
        :param u0: u at time t0
        :param v0: v, i.e. du/dt at time t0
        :param T: Maximum time
        :param t0: Beginning of time
        :return: timepoints, u_points, v_points
    """

    t = np.arange(t0-dt, T + dt, dt)

    """we want to include u-1 in AB 2"""

    N = t.shape[0] - 2

    updater = np.array([[0,1],[-1,0]])

    vec = np.array([[u0],[v0]])

    """execute back-euler step"""

    pre_vec = vec - dt * (updater @ vec)

    vec_l = np.hstack((pre_vec,vec))

    for i in range(N):

        vec_prev = vec_l[:,-2:-1]

        vec_local = vec + 1/2 * (3 * updater @ vec - updater @ vec_prev) * dt

        vec = vec_local

        vec_l = np.hstack((vec_l,vec))

    ul, vl = vec_l[0], vec_l[1]

    return t, ul, vl

"""
T = 1
N = 6
dts = [0.1**i for i in range(N)]
fig, ax = plt.subplots(4)
x = np.linspace(0,T,1000)

for i, val in enumerate(dts):
    t, ul, vl = adam_bashforth_2_sin(val)
    ax[0].plot(t, ul, lw=0.8, label='dt = %.2e' %(val))
    ax[1].plot(t, vl, lw=0.8, label='dt = %.2e' %(val))
    ax[2].plot(t, vl**2 + ul**2, lw = 0.8, label='Energy for dt = %.2e' %(val))
    ax[3].plot(t, np.abs(np.sin(t) - ul), lw=0.8, label = 'Absolute error in u(t) for dt %.2e' %(val) )

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].set_ylabel('u(t)')
ax[1].set_ylabel('v(t)')
ax[2].set_ylabel('E(t)')
ax[3].set_ylabel(r'$\epsilon(t)$')

ax[0].plot(x,np.sin(x),lw=0.8,label='Exact, for N = 1000 subdivions',color = 'blue')
ax[1].plot(x,np.cos(x),lw=0.8,label='Exact, for N = 1000 subdivions',color='blue')

for a in ax:
    a.set_xlabel('t')
    a.set_xlim(0,1)
    a.legend(prop={'size': 6})

plt.show()
"""

def AM2_sin(dt,u0=0,v0=1,T=1,t0=0):
    """
    integrates d^2 u / dt^2 + u^2 = 0 between t0 and T, with dt timestep, using Adam-Moulton method
    :param dt: Timestep
    :param u0: u at time t0
    :param v0: v, i.e. du/dt at time t0
    :param T: Maximum time
    :param t0: Beginning of time
    :return: timepoints, u_points, v_points
    """

    t = np.arange(t0 - dt, T + dt, dt)

    """we want to include u-1 in AB 2"""

    N = t.shape[0] - 1

    updater = np.array([[0, 1], [-1, 0]])

    vec = np.array([[u0], [v0]])

    """execute back-euler step"""

    vec_l = vec.copy()

    for i in range(N):

        """use euler predictor"""
        f_a = updater @ vec
        f_b = updater @ (vec + dt*f_a)

        """best prediction"""

        vec_predict = vec + (f_a + f_b)*dt / 2

        vec_local = vec + 1 / 2 * (3 * updater @ vec - updater @ vec_predict) * dt

        vec = vec_local

        vec_l = np.hstack((vec_l, vec))

    ul, vl = vec_l[0], vec_l[1]

    return t, ul, vl

"""
T = 1
N = 6
dts = [0.1**i for i in range(2,N)]
fig, ax = plt.subplots(4)
x = np.linspace(0,T,1000)

for i, val in enumerate(dts):
    t, ul, vl = AM2_sin(val)
    ax[0].plot(t, ul, lw=0.8, label='dt = %.2e' %(val))
    ax[1].plot(t, vl, lw=0.8, label='dt = %.2e' %(val))
    ax[2].plot(t, vl**2 + ul**2, lw = 0.8, label='Energy for dt = %.2e' %(val))
    ax[3].plot(t, np.abs(np.sin(t) - ul), lw=0.8, label = 'Absolute error in u(t) for dt %.2e' %(val) )

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].set_ylabel('u(t)')
ax[1].set_ylabel('v(t)')
ax[2].set_ylabel('E(t)')
ax[3].set_ylabel(r'$\epsilon(t)$')

ax[0].plot(x,np.sin(x),lw=0.8,label='Exact, for N = 1000 subdivions',color = 'blue')
ax[1].plot(x,np.cos(x),lw=0.8,label='Exact, for N = 1000 subdivions',color='blue')

for a in ax:
    a.set_xlabel('t')
    a.set_xlim(0,1)
    a.legend(prop={'size': 6})

plt.show()
"""
def RK2_sin(dt,u0=0,v0=1,T=1,t0=0,alpha = 1/2):
    """
    integrates d^2 u / dt^2 + u^2 = 0 between t0 and T, with dt timestep, using Adam-Moulton method
    :param dt: Timestep
    :param u0: u at time t0
    :param v0: v, i.e. du/dt at time t0
    :param T: Maximum time
    :param t0: Beginning of time
    :param alpha: chooses value of
    :return: timepoints, u_points, v_points
    """

    t = np.arange(t0 - dt, T + dt, dt)

    """we want to include u-1 in AB 2"""

    N = t.shape[0] - 1

    updater = np.array([[0, 1], [-1, 0]])

    vec = np.array([[u0], [v0]])

    """execute back-euler step"""

    vec_l = vec.copy()

    for i in range(N):
        f_a = updater @ vec
        f_b = updater @ (vec + alpha*dt*f_a)

        vec_local = vec + 1/(2*alpha) * ((2*alpha-1)*f_a + f_b ) * dt

        vec = vec_local
        vec_l = np.hstack((vec_l, vec))


    ul, vl = vec_l[0], vec_l[1]

    return t, ul, vl

"""
T = 1
N = 6
dts = [0.1**i for i in range(N)]
fig, ax = plt.subplots(4)
x = np.linspace(0,T,1000)

for i, val in enumerate(dts):
    t, ul, vl = RK2_sin(val,alpha = 3/4)
    ax[0].plot(t, ul, lw=0.8, label='dt = %.2e' %(val))
    ax[1].plot(t, vl, lw=0.8, label='dt = %.2e' %(val))
    ax[2].plot(t, vl**2 + ul**2, lw = 0.8, label='Energy for dt = %.2e' %(val))
    ax[3].plot(t, np.abs(np.sin(t) - ul), lw=0.8, label = 'Absolute error in u(t) for dt %.2e' %(val) )

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].set_ylabel('u(t)')
ax[1].set_ylabel('v(t)')
ax[2].set_ylabel('E(t)')
ax[3].set_ylabel(r'$\epsilon(t)$')

ax[0].plot(x,np.sin(x),lw=0.8,label='Exact, for N = 1000 subdivions',color = 'blue')
ax[1].plot(x,np.cos(x),lw=0.8,label='Exact, for N = 1000 subdivions',color='blue')

for a in ax:
    a.set_xlabel('t')
    a.set_xlim(0,1)
    a.legend(prop={'size': 6})

plt.show()
"""

def RK4_sin(dt,u0=0,v0=1,T=1,t0=0):
    """
    integrates d^2 u / dt^2 + u^2 = 0 between t0 and T, with dt timestep, using Adam-Moulton method
    :param dt: Timestep
    :param u0: u at time t0
    :param v0: v, i.e. du/dt at time t0
    :param T: Maximum time
    :param t0: Beginning of time
    :return: timepoints, u_points, v_points
    """

    t = np.arange(t0 - dt, T + dt, dt)

    """we want to include u-1 in AB 2"""

    N = t.shape[0] - 1

    updater = np.array([[0, 1], [-1, 0]])

    vec = np.array([[u0], [v0]])

    """execute back-euler step"""

    vec_l = vec.copy()

    for i in range(N):
        f_a = updater @ vec
        f_b = updater @ (vec + dt * f_a / 2)
        f_c = updater @ (vec + dt * f_b / 2)
        f_d = updater @ (vec + dt * f_c / 2)

        vec_local = vec + 1/6 * (f_a + 2 * f_b + 2 * f_c + f_d) * dt

        vec = vec_local
        vec_l = np.hstack((vec_l, vec))


    ul, vl = vec_l[0], vec_l[1]

    return t, ul, vl

"""
T = 1
N = 6
dts = [0.1**i for i in range(1,N)]
fig, ax = plt.subplots(4)
x = np.linspace(0,T,1000)

for i, val in enumerate(dts):
    t, ul, vl = RK4_sin(val)
    ax[0].plot(t, ul, lw=0.8, label='dt = %.2e' %(val))
    ax[1].plot(t, vl, lw=0.8, label='dt = %.2e' %(val))
    ax[2].plot(t, vl**2 + ul**2, lw = 0.8, label='Energy for dt = %.2e' %(val))
    ax[3].plot(t, np.abs(np.sin(t) - ul), lw=0.8, label = 'Absolute error in u(t) for dt %.2e' %(val) )

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].set_ylabel('u(t)')
ax[1].set_ylabel('v(t)')
ax[2].set_ylabel('E(t)')
ax[3].set_ylabel(r'$\epsilon(t)$')

ax[0].plot(x,np.sin(x),lw=0.8,label='Exact, for N = 1000 subdivions',color = 'blue')
ax[1].plot(x,np.cos(x),lw=0.8,label='Exact, for N = 1000 subdivions',color='blue')

for a in ax:
    a.set_xlabel('t')
    a.set_xlim(0,1)
    a.legend(prop={'size': 6})

plt.show()
"""

"""Part 2"""


def solve_ivp(N,x0=0,v0=1/3,t0=0, t1 = 1):
    """
    Solves x d^2 x / dt^2 = x for IVP case of known x0, v0
    :param N: Number of timesteps
    :param x0: x at t0
    :param v0: dx/dt i.e. v at t0
    :param t0: Beginning of time sampling
    :param t1: End of time sampling
    :return: t, x, v
    """
    t = np.linspace(t0,t1,N)

    dt = t[1]-t[0]

    vec = np.array([[x0],[v0]])

    """linear problem, represent as matrix"""


    vec_l = vec.copy()

    for i in range(N-1):

        v_next = vec[1] + t[i] * dt

        x_next = vec[0] + dt * vec[1]

        vec = np.array([x_next,v_next])

        vec_l = np.hstack((vec_l,vec))



    ul, vl = vec_l[0], vec_l[1]

    return t, ul, vl


"""
x, u, v = solve_ivp(100)

plt.plot(x,u,lw=0.8,color = 'red', label = 'First order Forward Euler')

plt.plot(x,x*(x**2 + 2)/6,lw=0.8,label='Exact, for N = 1000 subdivions',color = 'blue')

plt.plot(x, v,lw=0.8,color = 'orange', label = 'First order Forward Euler, derivative')

plt.plot(x,(3* x**2 + 2)/6,lw=0.8,label='Exact derivative, for N = 1000 subdivions',color = 'green')
plt.xlabel('t')
plt.ylabel('x(t),v(t)')
plt.legend()
plt.grid()
plt.show()
"""


def shooting_method(vguess_0,vguess_1,x1_actual=1/2,N=100,maxiter=300,de=10e-8):
    """
    To solve x(0) = 0, x(1) = 1/2, for d^2 x / dt^2 = t
    :param v0: initial guess for v0 at t = 0
    :return: difference between Euler-traced BV and actual BV
    """

    v_list = [vguess_0,vguess_1]

    for i in range(maxiter):

        x1 = v_list[-1]
        x2 = v_list[-2]

        t0, u0, v0 = solve_ivp(N, v0=x1)
        t1, u1, v1 = solve_ivp(N, v0=x2)

        f1 = u0[-1]-x1_actual
        f2 = u1[-1]-x1_actual

        x_new = x1 - f1 * (x1-x2)/(f1-f2)

        v_list.append(x_new)

        if np.abs(1-x_new/x1) < de:
            break

    return v_list

"""
shooting method
"""

"""
vals = shooting_method(0,1,maxiter = 50)
correct_val = vals[-1]

t, u, v = solve_ivp(100,x0=0,v0=correct_val)
plt.plot(t,u,lw=0.8,color = 'red', label = 'First order Forward Euler')

plt.plot(t,t*(t**2 + 2)/6,lw=0.8,label='Exact, for N = 1000 subdivions',color = 'blue')
plt.legend()
plt.grid()
plt.ylabel('u(t)')
plt.xlabel('t')
plt.show()
"""

def matrix_solver(N,x0,x1):

    A = np.zeros((N-1,N-1))

    rows, columns = np.indices(A.shape)

    upper_r, middle_r, lower_r = np.diag(rows,1), np.diag(rows,0), np.diag(rows,-1)
    upper_c, middle_c, lower_c = np.diag(columns, 1), np.diag(columns, 0), np.diag(columns, -1)

    A[upper_r,upper_c] = 1
    A[middle_r,middle_c] = -2
    A[lower_r,lower_c] = 1

    """bug here? uncomment and change N+1 to N-1"""

    t = np.linspace(0,1,N+1)

    dt = t[1]-t[0]

    t = np.linspace(dt,1-dt,N-1)

    dt = t[1]-t[0]


    sol_vector = t.reshape((N-1,1)) * dt**2
    BC_vec = np.zeros_like(sol_vector)
    BC_vec[0] = x0
    BC_vec[-1] = x1

    tot_vector = sol_vector - BC_vec

    A_inv = np.zeros((N-1,N-1))

    for i in range(N-1):
        A_inv[i,0] = 1 + i - N
        A_inv[0,i] = A_inv[i,0]
        for j in range(1,i+1):
            A_inv[i,j] = (j+1)*A_inv[i,0]
            A_inv[j,i] = A_inv[i,j]

    A_inv = A_inv/N

    print('Debug: maxDval between inverts: %.2e ' %(np.amax(np.abs(A_inv-np.linalg.inv(A)))))

    return_vec = A_inv @ tot_vector

    T = np.linspace(0,1,N+1)

    return_vec = return_vec.flatten()

    x0 = np.array(x0)
    x1 = np.array(x1)

    return_vec = np.hstack((x0,return_vec,x1))

    return T, return_vec
"""
t, y = matrix_solver(7,0,1/2)
plt.plot(t, t*(t**2 + 2)/6,lw=1.5,label='Exact, for N = 1000 subdivions',color = 'blue',alpha=0.5)
plt.plot(t, y,lw=1.5,color='red',label='Matrix based solution',alpha=0.5)
plt.legend()
plt.grid()
plt.ylabel('u(t)')
plt.xlabel('t')
plt.show()
"""


def matrix_solver_complicated(N,x0,x1,maxiter=200,cond=10e-9):
    """
    Solves u'' = 3u - t^3 / 2 for x(0) = x0, x(1) = x1 using Classical method matrix
    :param N: Equation solved for N-1 datapoints
    :param x0: See above, lower BC
    :param x1: See above, upper BC
    :return: timepoints, solution at timepoints
    """

    A = np.zeros((N-1,N-1))

    rows, columns = np.indices(A.shape)

    upper_r, middle_r, lower_r = np.diag(rows,1), np.diag(rows,0), np.diag(rows,-1)
    upper_c, middle_c, lower_c = np.diag(columns, 1), np.diag(columns, 0), np.diag(columns, -1)

    A[upper_r,upper_c] = 1
    A[middle_r,middle_c] = -2
    A[lower_r,lower_c] = 1

    t = np.linspace(0, 1, N + 1)

    dt = t[1] - t[0]

    t = np.linspace(dt, 1 - dt, N - 1)

    dt = t[1] - t[0]

    BC_vec = np.zeros((N-1,1))
    BC_vec[0] = x0
    BC_vec[-1] = x1

    A_inv = np.zeros((N-1,N-1))

    for i in range(N-1):
        A_inv[i,0] = 1 + i - N
        A_inv[0,i] = A_inv[i,0]
        for j in range(1,i+1):
            A_inv[i,j] = (j+1)*A_inv[i,0]
            A_inv[j,i] = A_inv[i,j]

    A_inv = A_inv/N

    print('Debug: maxDval between inverts: %.2e ' %(np.amax(np.abs(A_inv-np.linalg.inv(A)))))

    v0_guess = (x1-x0)*t + x0


    v0_guess = v0_guess.reshape((N-1,1))

    t = t.reshape(((N-1,1)))

    for i in range(maxiter):
        b = (3 * v0_guess - t**3 / 2) * dt**2 - BC_vec

        v1_guess = A_inv @ b

        if np.sqrt(np.sum((v0_guess-v1_guess)**2)) < cond:
            v0_guess = v0_guess
            break
        v0_guess = v1_guess


    T = np.linspace(0,1,N+1)

    return_vec = v0_guess

    return_vec = return_vec.flatten()

    x0 = np.array(x0)
    x1 = np.array(x1)

    return_vec = np.hstack((x0,return_vec,x1))

    return T, return_vec

"""t, y = matrix_solver_complicated(100,0,1/2)
plt.plot(t, t*(t**2 + 2)/6,lw=1.5,label='Exact, for N = 1000 subdivisions',color = 'blue',alpha=0.5)
print('Maximum absolute difference: %.2e' %np.amax(np.abs(y-t*(t**2 + 2)/6)))
plt.plot(t, y, lw=1.5,color='red',label='Matrix based solution',alpha=0.5)
plt.legend()
plt.grid()
plt.ylabel('u(t)')
plt.xlabel('t')
plt.show()
"""










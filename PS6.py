import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from warnings import warn
import matplotlib.animation as animation

wind_df = pd.read_csv('solarWind/Wind_Data.csv')
omni_df = pd.read_csv('solarWind/OMNI_Data.csv')

"""solve 1d convection eq. using upwind"""

def adv_eq_1d_periodic(dx,dt,T,x0,x1,y0,y1,vx=1,y_starter = True):
    """
    Solves dimensionless advection equation for periodic BC-s, i.e.:
    ∂u/∂t = -v_x * ∂ u /∂ x
    :param dx: Spatial subdivision
    :param dt: Temporal subdivision. Needs to be smaller than dx**2 /2 for stability
    :param T: Time to be solved for
    :param x0: position of lower bound
    :param x1: position of upper bound
    :param y0: y(x0,0) BC (quasi-redundant)
    :param y1: y(x1,0) BC (quasi-redundant)
    :param vx: velocity parameter, assumed to be positive
    :return: Solution vector of shape of (T/dt, (x1-x0)/dx), x spatial array
    """

    if dt > dx /vx:
        warn('The spatial subdivision does NOT satisfy the stability criterion. '
             'Consider lowering it.',category=Warning)

    x = np.arange(x0,x1+dx,dx)
    N = x.shape[-1]
    t = np.arange(0,T+dt,dt)
    N_t = t.shape[-1]

    """Extra padding for a trick, later on"""

    if y_starter:
        y_starter = (1+np.sin(x*2*np.pi))/2

    y_starter[1], y_starter[-1] = y0, y1
    y_next = y_starter

    a = np.abs(vx)*dt/dx

    tot_solutions = y_starter.reshape((1,N))

    """Execute method described in notes"""

    for i in range(N_t):

        y_minus = np.hstack((np.array([0]),y_next[0:-1]))

        y_loc = (1-a)*y_next + a*y_minus

        y_loc[0] = y_loc[-1]


        y_next = y_loc.copy()

        y_loc = y_loc.reshape((1,N))

        tot_solutions = np.vstack((tot_solutions,y_loc))

    return tot_solutions, x

def adv_eq_1d_neumann(dx,dt,T,x0,x1,f,vx=1,y_starter = True):
    """
    Solves dimensionless advection equation for dirichlet BC-s, i.e.:
    ∂u/∂t = -v_x * ∂ u /∂ x. ∂ u(x1) /∂ x = 0
    :param dx: Spatial subdivision
    :param dt: Temporal subdivision. Needs to be smaller than dx**2 /2 for stability
    :param T: Time to be solved for
    :param x0: position of lower bound
    :param x1: position of upper bound
    :param f: f(x0,t) BC. Simple scalar funtion
    :param vx: velocity parameter, assumed to be positive
    :return: Solution vector of shape of (T/dt, (x1-x0)/dx), x spatial array
    """

    if dt > dx /np.abs(vx):
        warn('The spatial subdivision does NOT satisfy the stability criterion. '
             'Consider lowering it.',category=Warning)

    x = np.arange(x0,x1+dx,dx)
    N = x.shape[-1]
    t = np.arange(0,T+dt,dt)
    N_t = t.shape[-1]

    if y_starter:
        y_starter = np.zeros(N+1)

    y_starter[0] = inflow(0)

    y_next = y_starter

    a = np.abs(vx)*dt/dx

    tot_solutions = y_starter.reshape((1,N+1))

    """Execute method described in notes"""

    for i in range(N_t):

        y_minus = np.roll(y_next,1)

        y_minus[0] = 0

        y_loc = (1-a)*y_next + a*y_minus

        y_loc[0] = f(i*dt)

        """shittiest way to enforce RH BC"""

        """y_loc[-2] = y_loc[-1]"""

        y_next = y_loc.copy()

        y_loc = y_loc.reshape((1,N+1))

        tot_solutions = np.vstack((tot_solutions,y_loc))

    return tot_solutions[:,0:-1], x


def inflow(t):

    return (1 + np.sin(2 * np.pi * t) )/2

x0, x1 = 0, 1
y0, y1 = 0.5, 0.5
T = 10
dx = 0.01
dt = 0.001
solution, x = adv_eq_1d_neumann(dx,dt,T,x0,x1,inflow)
t = np.arange(0,T+dt,dt)
N_t = t.shape[-1]
fig, ax = plt.subplots()
ax.grid()
ax.set_xlim(x0-0.2,x1+0.2)
ax.set_ylim(-2.,2.)
ax.set_ylabel('U(x)')
ax.set_xlabel('x')


line, = ax.plot([],[],lw=0.8,color='red',label='Current solution')
ax.legend()

def init():
    line.set_data([], [])
    return line,

def animate(frame):

    frame = frame % N_t

    y = solution[frame]

    line.set_data(x,y)


    return line,

"""
z = np.linspace(0,10,11)
z_1 = np.roll(z,1)
print(z,z_1)
"""

rate = 10
anim = animation.FuncAnimation(fig, animate,
                            init_func = init,
                            frames=np.arange(0, N_t, 1)[::rate],
                            interval = 20,
                            blit = True)

plt.show()

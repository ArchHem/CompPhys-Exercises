import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from warnings import warn


def heat_eq_1d(dx,dt,T,x0,x1,y0,y1,y_starter = True):
    """
    Solves dimensionless heat equation for fixed-end BC-s, i.e.:
    ∂u/∂t = 1* ∂^2 u /∂^2 x
    :param dx: Spatial subdivision
    :param dt: Temporal subdivision. Needs to be smaller than dx**2 /2 for stability
    :param T: Time to be solved for
    :param x0: position of lower bound
    :param x1: position of upper bound
    :param y0: y(x0,t) BC, fixed in time
    :param y1: y(x1,t) BC, fixed in time
    :return: Solution vector of shape of (T/dt, (x1-x0)/dx), x spatial array
    """

    if dt > dx**2 /2:
        warn('The spatial subdivision does NOT satisfy the stability criterion. '
             'Consider lowering it.',category=Warning)

    x = np.arange(x0,x1+dx,dx)
    N = x.shape[-1]
    t = np.arange(0,T+dt,dt)
    N_t = t.shape[-1]

    """Extra padding for a trick, later on"""

    if y_starter:
        y_starter = np.zeros(N)

    y_starter[1], y_starter[-1] = y0, y1
    y_next = y_starter

    d = dt/dx**2

    tot_solutions = y_starter.reshape((1,N))

    """Execute method described in notes"""

    for i in range(N_t):

        y_minus = np.hstack((np.array([0]),y_next[0:-1]))
        y_plus = np.hstack((y_next[1:],np.array([0])))

        y_loc = (1-2*d)*y_next + d*(y_minus+y_plus)

        y_loc[0] = y0
        y_loc[-1] = y1

        y_next = y_loc.copy()

        y_loc = y_loc.reshape((1,N))

        tot_solutions = np.vstack((tot_solutions,y_loc))

    return tot_solutions, x

x0, x1 = 0, 1
y0, y1 = 0, 1
T = 1
dx = 0.05
dt = 0.0004
solution, x = heat_eq_1d(dx,dt,T,x0,x1,y0,y1)
t = np.arange(0,T+dt,dt)
N_t = t.shape[-1]
fig, ax = plt.subplots()
ax.grid()
ax.set_xlim(x0-0.2,x1+0.2)
ax.set_ylim(y0-0.2,y1+0.2)
ax.set_ylabel('U(x)')
ax.set_xlabel('x')
ax.plot(x,(y1-y0)/(x1-x0) * (x-x0) + y0,ls='--',color = 'green',label='Steady state')

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

rate = 1
anim = animation.FuncAnimation(fig, animate,
                            init_func = init,
                            frames=np.arange(0, N_t, 1)[::rate],
                            interval = 20,
                            blit = True)

plt.show()
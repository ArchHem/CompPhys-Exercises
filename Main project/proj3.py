import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def KDEV_RUNGE(v0,T=1.0,L=1.0,dt=0.001,dx=0.05):

    t = np.arange(0,T+dt,dt)

    maxsteps = t.shape[-1]
    x = np.arange(-L,L+dx,dx)
    N = x.shape[-1]

    vec = v0.reshape(1,N)
    totvec = vec.copy()

    def loc_gradient(vec):

        vp1 = np.roll(vec, -1)
        vp2 = np.roll(vec, -2)

        vm1 = np.roll(vec, +1)
        vm2 = np.roll(vec, +2)

        spatial_grad = -(vp1**2 - vm1**2) / (4 * dx) - \
                       (vp2 - 2 * vp1 + 2 * vm1 - vm2) / (2 * dx**3)
        return spatial_grad


    for i in range(maxsteps):


        """print('%.4f' % (100 * i / maxsteps))"""

        vec = vec.flatten()

        fa = loc_gradient(vec)

        fb = loc_gradient(vec + dt*fa/2)

        fc = loc_gradient(vec + dt*fb/2)

        fd = loc_gradient(vec + dt*fc)

        vec = vec + dt*(fa + 2*fb + 2*fc + fd)/6

        vec.reshape(1, N)

        totvec = np.vstack((totvec, vec))

    return totvec, x, t

def KDEV_ZABUSKY(v0,T=1.0,L=1.0,dt=0.001,dx=0.05):
    t = np.arange(0, T + dt, dt)

    maxsteps = t.shape[-1]
    x = np.arange(-L, L + dx, dx)
    N = x.shape[-1]

    vec = v0.reshape(1, N)
    totvec = vec.copy()

    vec = vec.flatten()

    """https://link.springer.com/content/pdf/10.1007/BF01535405.pdf?fbclid=IwAR0KsbnXgKF-oj9UMuRh9noBlEogJITF3HVWdTc3EGMGfHbmBwaj7KlpaXk"""

    vp2 = np.roll(vec, -2)
    vp1 = np.roll(vec, -1)
    vm1 = np.roll(vec,  1)
    vm2 = np.roll(vec,  2)

    v1 = vec - dt*(vp1+vec+vm1)*(vp1-vm1)/(6*dx) - dt*(vp2 -2*vp1 + 2 * vm1 - vm2)/(2 * dx**3)

    v1 = v1.reshape(1,N)

    totvec = np.vstack((totvec,v1))

    vec = v1.flatten().copy()

    def loc_gradient(vec):

        vp1 = np.roll(vec, -1)
        vp2 = np.roll(vec, -2)

        vm1 = np.roll(vec, +1)
        vm2 = np.roll(vec, +2)

        spatial_grad = -(vp1 + vec + vm1)*(vp1 - vm1) / (3 * dx) - \
                       (vp2 - 2 * vp1 + 2 * vm1 - vm2) / (dx**3)
        return spatial_grad


    for i in range(1,maxsteps):

        print('%.4f' %(100*i/maxsteps))

        vec = vec.flatten()

        vpast = totvec[-2]

        vec = vpast + dt*loc_gradient(vec)

        vec = vec.reshape(1,N)

        totvec = np.vstack((totvec,vec))

    return totvec, x, t




dx = 0.1

alpha = 1
funcmax = 12*alpha**2
L = 5
x = np.arange(-L,L+dx,dx)
dt_max = dx/(funcmax + 4/dx**2)
dt = 1e-4
t0 = 0


def sech(x):
    return 1/np.cosh(x)

def solvec(x,t):
    return 12*alpha**2 * sech(alpha*(x-4*alpha**2 * t))**2

y = solvec(x,t0)

T = 2
solution, x, t = KDEV_ZABUSKY(y,L=L,T=T,dt=dt,dx=dx)

N_t = t.shape[-1]
fig, ax = plt.subplots()
ax.grid()

ax.set_xlim(-0.2-L,L+0.2)
ax.set_ylim(-13,13)

ax.set_ylabel('U(x)')
ax.set_xlabel('x')

line, = ax.plot([],[],lw=0.8,color='red',label='Current solution')
line1, = ax.plot([],[],lw=0.8,color = 'blue',alpha=0.5)
ax.legend()

def init():
    line.set_data([], [])
    line1.set_data([],[])
    return [line, line1]

def animate(frame):

    frame = frame % N_t

    y = solution[frame]

    line.set_data(x,y)

    t = frame*dt

    line1.set_data(x,solvec(x,t))



    return [line, line1]

rate = 100
anim = animation.FuncAnimation(fig, animate,
                            init_func = init,
                            frames=np.arange(0, N_t, 1)[::rate],
                            interval = 20,
                            blit = True)
plt.show()


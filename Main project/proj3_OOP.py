import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

def sech(x):
    return 1/np.cosh(x)

def solvec(x,t,alpha=1):
    return 12*alpha**2 * sech(alpha*(x-4*alpha**2 * t))**2

class KDEV_solving:
    def __init__(self,x,v0,dt: float =1e-4):
        self.x = x
        self.N = self.x.shape[-1]
        self.v0 = v0
        self.y = v0.reshape(1,self.N)
        self.dt = dt
        self.dx = np.diff(x)[0]
        self.lx, self.hx = np.amin(x), np.amax(x)
        self.time = np.array([0])

    def solve_IVP(self,method='RK4',maxsteps = 1000):


        dt, dx = self.dt, self.dx

        idx = np.arange(self.N)

        idm1 = np.mod((idx + 1), self.N)
        idm2 = np.mod((idx + 2), self.N)

        idp1 = np.mod((idx - 1), self.N)
        idp2 = np.mod((idx - 2), self.N)

        if method == 'RK4':
            vec = self.y[-1].reshape(1,self.N)

            locT = np.arange(self.time[-1]+dt,self.time[-1] + dt*(maxsteps+1),dt)

            locT = locT[0:maxsteps]

            totvec = vec.copy()

            def loc_gradient_RK(vec):

                spatial_grad = -(vec[idm1] ** 2 - vec[idp1] ** 2) / (4 * dx) \
                               - (vec[idm2] - 2 * vec[idm1] + 2 * vec[idp1] - vec[idp2]) / (2 * dx ** 3)
                return spatial_grad

            for i in tqdm(range(maxsteps)):
                """print('%.4f' % (100 * i / maxsteps))"""

                vec = vec.flatten()

                fa = loc_gradient_RK(vec)

                fb = loc_gradient_RK(vec + dt * fa / 2)

                fc = loc_gradient_RK(vec + dt * fb / 2)

                fd = loc_gradient_RK(vec + dt * fc)

                vec = vec + dt * (fa + 2 * fb + 2 * fc + fd) / 6

                vec.reshape(1, self.N)

                totvec = np.vstack((totvec, vec))

            self.y = np.vstack((self.y,totvec[1:]))

            self.time = np.hstack((self.time,locT))


        if method == 'zabusky':
            vec = self.y[-1].reshape(1, self.N)

            locT = np.arange(self.time[-1]+dt,self.time[-1] + dt*(maxsteps+1),dt)

            locT = locT[0:maxsteps]

            totvec = vec.copy()

            if self.y.shape[0] == 1:

                vec = vec.flatten()

                """https://link.springer.com/content/pdf/10.1007/BF01535405.pdf?fbclid=IwAR0KsbnXgKF-oj9UMuRh9noBlEogJITF3HVWdTc3EGMGfHbmBwaj7KlpaXk"""

                v1 = vec - dt/2 * (vec[idm1] + vec + vec[idp1]) * (vec[idm1] - vec[idp1]) / (6 * dx) - \
                     dt * (vec[idm2] - 2 * vec[idm1] + 2 * vec[idp1] - vec[idp2]) / (2 * dx ** 3)

                v1 = v1.reshape(1, self.N)
                locT = np.arange(self.time[-1]+dt, self.time[-1] + dt * (maxsteps + 2), dt)
                locT = locT[0:maxsteps]

                totvec = np.vstack((totvec, v1))

            else:
                totvec = np.vstack((self.y[-2].reshape(1,self.N),vec))

            def loc_gradient_ZA(vec):

                spatial_grad = -(vec[idm1] + vec + vec[idp1]) * (vec[idm1] - vec[idp1]) / (3 * dx) \
                               - (vec[idm2] - 2 * vec[idm1] + 2 * vec[idp1] - vec[idp2]) / (dx ** 3)
                return spatial_grad/2

            for i in tqdm(range(maxsteps)):

                vec = vec.flatten()

                fa = loc_gradient_ZA(vec)

                fb = loc_gradient_ZA(vec + dt * fa / 2)

                fc = loc_gradient_ZA(vec + dt * fb / 2)

                fd = loc_gradient_ZA(vec + dt * fc)

                vec = vec + dt * (fa + 2 * fb + 2 * fc + fd) / 6

                vec.reshape(1, self.N)

                totvec = np.vstack((totvec, vec))

            self.y = np.vstack((self.y, totvec[2:]))

            self.time = np.hstack((self.time,locT))



    def animate(self,simple_analytical_show = False, ylims =(-13,13),rate = 50):
        N_t = self.time.shape[-1]
        self.fig, self.ax = plt.subplots()

        self.ax.grid()

        self.ax.set_xlim(1.2*self.lx, 1.2*self.hx)
        self.ax.set_ylim(ylims)

        self.ax.set_ylabel('U(x)')
        self.ax.set_xlabel('x')

        line, = self.ax.plot([], [], lw=0.8, color='red', label='Current solution')

        line1, = self.ax.plot([], [], lw=0.8, color='blue',label='Analytical solution',alpha=0.5)
        self.ax.legend()

        def init():
            line.set_data([], [])
            if simple_analytical_show:
                line1.set_data([], [])
            return [line, line1]

        def animate(frame):

            y = self.y[frame]

            line.set_data(self.x, y)
            if simple_analytical_show:

                t = frame * self.dt
                line1.set_data(self.x, solvec(self.x, t))

            return [line, line1]


        self.anim = animation.FuncAnimation(self.fig, animate,
                                       init_func=init,
                                       frames=np.arange(0, N_t, 1)[::rate],
                                       interval=20,
                                       blit=True)
        self.fig.show()






x = np.linspace(-10,10,200)
v0 = solvec(x,0)
inst_1 = KDEV_solving(x,v0,dt=5*1e-4)
inst_1.solve_IVP(maxsteps=10000,method='RK4')

print(inst_1.y.shape)
print(inst_1.time.shape)
inst_1.animate(rate=100,simple_analytical_show=True)
plt.show()

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


class JJs:
    def __init__(self, Ic1, Ic2, R1, R2, R):
        self.Ic1 = Ic1
        self.Ic2 = Ic2
        self.R1 = R1
        self.R2 = R2
        self.R = R

    def model(self,t, x, I=0):
        # normalize the current to IC1
        i = I/self.Ic1
        ic2 = self.Ic2/self.Ic1

        # normalize the resistors to R1
        r = self.R/self.R1
        r2 = self.R2/self.R1

        # assign each ODE to a vector element
        phi1 = x[0]
        phi2 = x[1]

        # define the ODEs
        dphi1dt = ((r * r2) / (2 * r + 1)) * ( ((r + r2) / (r2 * r))* (i - np.sin(phi1)) - (1 / r) * (i - ic2 * np.sin(phi2)) )
        dphi2dt = ((r * r2) / (2 * r + 1)) * ( (-1 / r) * (i - np.sin(phi1)) + ((r + 1) / r) * (i - ic2 * np.sin(phi2)))

        return [dphi1dt, dphi2dt]
    
    @staticmethod
    def phi_adjust(val):
        '''
        Adjust the phase difference to be between 0 and 2*pi
        '''
        n = int(val/(2*np.pi))
        return val - (2*np.pi*n)
    
    def solve(self, t_span,dt,t_av_start, x0, I, iterative_av = False):
        '''
        Solve the system of ODEs using the solve_ivp function from scipy.integrate
        :param t_span: tuple with the initial and final time
        :param dt: time step
        :param t_av_start: time to start averaging the voltage
        :param x0: initial conditions
        :param I: external current
        :param iterative_av: if True, the voltage is averaged iteratively (more precise, slower)
        :return: V1, V2, and phi1, phi2 which are starting conditions for next iteration
        '''
        methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
       


        if iterative_av == False:
            sol = solve_ivp(self.model, t_span, x0, t_eval=None, method=methods[1], args=(I,), max_step=dt)
            phi1 = sol.y[0]
            phi2 = sol.y[1]
            t = sol.t
            index = np.where(sol.t > t_av_start)[0][0]
            V1 = 1/(sol.t[-1]-sol.t[index]) * (phi1[-1] - phi1[index])
            V2 = 1/(sol.t[-1]-sol.t[index]) * (phi2[-1] - phi2[index])

        else:
            return "Not implemented yet"
        
        return V1, V2, phi1[-1], phi2[-1]
    


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    # define the parameters
    Ic1 = 50e-6
    Ic2 = 40e-6
    R1 = 19
    R2 = 12
    R = 40

    # create the JJs object
    jjs = JJs(Ic1, Ic2, R1, R2, R)

    # define the initial conditions
    x0 = [0, 0]

    # define the time span
    t_max = 500
    t_span = (0, t_max)

    # define the time step
    dt = 0.05

    # define the time to start averaging the voltage
    t_av_start = 100

    # Create DataFrame to store results
    result = pd.DataFrame(columns=['I', 'V1', 'V2'])

    # solve the system of ODEs
    for i in tqdm(range(400)):
        I = i*1e-6

        V1, V2, phi1, phi2 = jjs.solve(t_span, dt, t_av_start, x0, I)

        result = pd.concat([result, pd.DataFrame({'I': [I], 'V1': [V1], 'V2': [V2]})])

        # Set new starting conditions
        x0 = [jjs.phi_adjust(phi1), jjs.phi_adjust(phi2)]


    # Save the resulting table
    result.to_csv('2_coupled_JJs.csv', index=False)
    plt.figure()
    plt.plot(result['V1'],result['I'], label='V1')
    plt.plot(result['V2'],result['I'], label='V2')
    plt.show()

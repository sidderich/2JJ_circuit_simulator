import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import sys


class JJs:
    def __init__(self, Ic1, Ic2, R1, R2, R):
        self.Ic1 = Ic1
        self.Ic2 = Ic2
        self.R1 = R1
        self.R2 = R2
        self.R = R
        
        # initialize the time step
        self.dt = 0.1

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
    
    @staticmethod
    def period_finder(t, phi):
        '''
        Find time period of the oscillation
        '''

        # Find the peaks

        '''peaks = np.where((phi[1:-1] > phi[:-2]) & (phi[1:-1] > phi[2:]))[0]'''

        peaks, _ = find_peaks(np.sin(phi))
        if len(peaks) < 2:
            print('Not enough peaks found to calculate the period of oscillation')
            sys.exit(1)

        period = np.mean(np.diff(t[peaks]))
        # Calculate threshhold for standard derivation
        # threshhold = 0.01 * period
        # Show warning if the period is not stable
        # Use standard derivation of the period if the period is not stable
        '''if np.std(np.diff(t[peaks])) > threshhold:
            print('\n WARNING: The period is not stable')'''

        # Calculate the period
        return period
    
    
    def check_dt(self, t, phi):
        '''
        Check if the time step is small enough
        '''
        period = JJs.period_finder(t, phi)
        if self.dt > period/20:
            print(f'\n WARNING: Time step is too large! Period: {period} timestep: {self.dt} \n Adjusting the time step')
            # increase the time step
            self.dt = self.dt * 0.9
            return False
        else:
            return True


    def solve(self, t_span,t_av_start, x0, I, iterative_av = True, dt = 0.05):
        '''
        Solve the system of ODEs using the solve_ivp function from scipy.integrate
        :param t_span: tuple with the initial and final time
        :param t_av_start: time to start averaging the voltage
        :param x0: initial conditions
        :param I: external current
        :param iterative_av: if True, the voltage is averaged iteratively (more precise, slower)
        :return: V1, V2, and phi1, phi2 which are starting conditions for next iteration
        '''
        methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']

        if iterative_av == False:
            sol = solve_ivp(self.model, t_span, x0, t_eval=None, method=methods[1], args=(I,), max_step=self.dt)
            phi1 = sol.y[0]
            phi2 = sol.y[1]
            t = sol.t
            # check dt
            self.check_dt(t, phi1)
            self.check_dt(t, phi2)
            index = np.where(sol.t > t_av_start)[0][0]
            V1 = 1/(sol.t[-1]-sol.t[index]) * (phi1[-1] - phi1[index])
            V2 = 1/(sol.t[-1]-sol.t[index]) * (phi2[-1] - phi2[index])

        else: # iterative averaging
            epsilon = 1e-3
            t_av = t_av_start # time to start averaging the voltage, will grow if the voltage is not stable
            t_av_factor = 1.2  # factor to increase t_av if the voltage is not stable
            t_av_max = 10000  # maximum time to average the voltage
            # Initial solution
            sol = solve_ivp(self.model, t_span, x0, t_eval=None, method=methods[1], args=(I,), max_step=self.dt)
            phi1 = sol.y[0]
            phi2 = sol.y[1]
            t = sol.t
            

            # Index after t_av
            index = np.where(sol.t > t_av)[0][0]

            successful = False

            while not successful:
                # Calculate the voltage after t_av
                V1 = 1/(t[-1]-t[t_av_start]) * (phi1[-1] - phi1[t_av_start])
                V2 = 1/(t[-1]-t[t_av_start]) * (phi2[-1] - phi2[t_av_start])
               
                # Increase the time to average the voltage
                t_av = t_av_factor * t_av
                
                if t_av > t_av_max:
                    print(f'The voltage is not stable after {t_av_max} timesteps')
                    break
                
                # Calculate the new phi after t_av
                sol_check = solve_ivp(self.model, (t[-1], t[-1]+t_av), [phi1[-1],phi2[-1]], t_eval=None, method=methods[1], args=(I,), max_step=dt)
                phi1_check = sol_check.y[0]
                phi2_check = sol_check.y[1]
                t_check = sol_check.t

                # Check if the timestep is small enough
                if I > self.Ic1 or I > self.Ic2:
                    if self.check_dt(t_check, phi1_check) and self.check_dt(t_check, phi2_check):
                        successful = False
                    else:
                        # Skip the rest of the loop
                        continue

                # Concatenate the new solution with the old one
                phi1 = np.hstack((phi1, phi1_check))
                phi2 = np.hstack((phi2, phi2_check))
                t = np.hstack((t, t_check))
                
                # Index after t_av
                index = np.where(t > t_av)[0][0]

                # Calculate the voltage after t_av
                V1_later = 1/(t[-1]-t[index]) * (phi1[-1] - phi1[index])
                V2_later = 1/(t[-1]-t[index]) * (phi2[-1] - phi2[index])

                # Check if the voltage is stable by checking if voltage after t_av is close to the voltage after t_av_start
                if np.abs(V1-V1_later) < epsilon and np.abs(V2-V2_later) < epsilon:
                    V1 = V1_later
                    V2 = V2_later
                    break
                    
        return V1, V2, phi1, phi2, t
    
  


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

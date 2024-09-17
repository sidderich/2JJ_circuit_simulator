import unittest
import numpy as np
from coupled_JJs import JJs

class TestJJs(unittest.TestCase):

    def setUp(self):
        # Values of the parameters are arbitrary
        self.JJs = JJs(1, 1, 1, 1, 1) 
        self.mV = 1e-2

    def test_period_finder(self):
        '''
        This method is a unit test for the period_finder function in the JJs class.
        It tests whether the period_finder function correctly calculates the period of a given signal.
        Parameters:
        - self: The instance of the test class.
        Returns:
        - None
        '''
        t = np.linspace(0, 10, 1000)
        phi = np.sin(t)
        period = JJs.period_finder(t, phi)
        self.assertAlmostEqual(period, 2 * np.pi, places=2)

    def test_check_dt(self):
        '''
        This method is a unit test for the check_dt function in the JJs class.
        It tests whether the check_dt function correctly checks the time step of a given signal.
        '''
        #number of periods
        n = 2
        t = np.linspace(0, n*2*np.pi, 100)
        phi = np.sin(t)
        dt = 0.05
        try:
            self.JJs.check_dt(t, phi)
        except SystemExit as e:
            self.assertEqual(e.code, 1)


class TestNewModel(unittest.TestCase):

    def setUp(self):
        self.JJs = JJs(1, 1, 1, 1, 1) 
        self.mV = 1e-2

    def test_solve(self):
        '''
        This method is a unit test for the solve function in the JJs class.
        '''
        t_span = (0, 10)
        t_av_start = 0
        x0 = [0, 0, 0]
        I = 1
        sol = self.JJs.solve(t_span, t_av_start, x0, I, dt = 0.05, model = 'ind')
        print(sol)

    
        


if __name__ == '__main__':
    unittest.main()
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
        t = np.linspace(0, 10, 1000)
        phi = np.sin(self.mV * 483e9 * t)
        dt = 0.05
        try:
            self.JJs.check_dt(t, phi)
        except SystemExit as e:
            self.assertEqual(e.code, 1)
    
        


if __name__ == '__main__':
    unittest.main()
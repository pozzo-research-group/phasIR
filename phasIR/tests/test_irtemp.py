import os
import unittest

import irtemp


# def test_name():
#     '''Doc String'''
#     #inputs
#     #running function
#     #asserts
#     return
class TestSimulationTools(unittest.TestCase):
    def test_centikelvin_to_celcius(self):
        '''
        Converts given centikelvin value to Celsius
        '''
        cels = irtemp.centikelvin_to_celsius(100000)
        assert isinstance(cels, float), 'Output is not a float'
        return

    def test_to_fahrenheit(self):
        '''
        Converts given centikelvin reading to fahrenheit
        '''
        fahr = irtemp.to_fahrenheit(100000)
        assert isinstance(fahr, float), 'Output is not a float'
        return

    def test_to_temperature(self):
        '''
        Converts given centikelvin value to both fahrenheit and celcius
        '''
        cels, fahr = irtemp.to_temperature(100000)
        assert isinstance(fahr, float), 'Output is not a float'
        assert isinstance(cels, float), 'Output is not a float'
        return

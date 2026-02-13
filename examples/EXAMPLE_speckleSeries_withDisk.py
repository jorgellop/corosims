import corosims
import matplotlib.pylab as plt
import numpy as np
import os

if __name__ == '__main__':
    
    name_speckleSeries = 'example_speckleSeries_ScienceAndRef'
    obs_obj = corosims.Observation(name=name_speckleSeries, cor_type = 'hlc_band1', bandpass='1')

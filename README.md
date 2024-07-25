# corgisim (https://github.com/roman-corgi/cgisim_sims)
This package includes routines and example scripts for simulating observations with the Roman Coronagraph.
The goal of this package is to provide a modular, accessible way of producing simulated observation scenarios with the Roman Coronagraph.
The code wraps around cgisim (John Krist, JPL, https://sourceforge.net/projects/cgisim) to simulate the instrument, and offers easy ways of defining scenes and observation batches with user-defined sources, arbitrary timeseries of wavefront errors, detector noise (based on emccd_detect, https://github.com/roman-corgi/emccd_detect) 

## Installation
pip install .

# corgisims (https://github.com/roman-corgi/cgisim_sims)
This package includes routines and example scripts for simulating observations with the Roman Coronagraph.
The goal of this package is to provide a modular, accessible way of producing simulated observation scenarios with the Roman Coronagraph.
The code wraps around cgisim (John Krist, JPL, https://sourceforge.net/projects/cgisim) to simulate the instrument, and offers easy ways of defining scenes and observation batches with user-defined sources, arbitrary timeseries of wavefront errors, detector noise (based on emccd_detect, https://github.com/roman-corgi/emccd_detect) 

Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons

## Installation
pip install -e .

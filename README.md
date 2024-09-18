# corgisims (https://github.com/roman-corgi/cgisim_sims)
This package includes routines and example scripts for simulating observations with the Roman Coronagraph.
The goal of this package is to provide a modular, accessible way of producing simulated observation scenarios with the Roman Coronagraph.
The code wraps around cgisim (John Krist, JPL, https://sourceforge.net/projects/cgisim) to simulate the instrument, and offers easy ways of defining scenes and observation batches with user-defined sources, arbitrary timeseries of wavefront errors, detector noise (based on emccd_detect, https://github.com/roman-corgi/emccd_detect) 

## Installation
pip install -e .

## License
Copyright (c) 2023-24 California Institute of Technology (Caltech). U.S. Government sponsorship acknowledged.
THE SOFTWARE IS LICENSED UNDER THE APACHE LICENSE, VERSION 2.0 (THE "LICENSE")
YOU MAY NOT USE THIS FILE EXCEPT IN COMPLIANCE WITH THE LICENSE.
YOU MAY OBTAIN A COPY OF THE LICENSE at http://www.apache.org/licenses/LICENSE-2.0
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

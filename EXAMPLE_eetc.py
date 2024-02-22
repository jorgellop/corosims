#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:39:56 2024

@author: llopsayson
"""

from eetc.cgi_eetc import CGIEETC

sequence_name = 'CGI_SEQ_DATA_TCATTF_PR_INFOCUS'
mag=11+8*2.5
cgi_eetc = CGIEETC(mag=mag, phot='v',spt='g2v')

num_frames, exp_time_frame, gain, snr_out, optflag = cgi_eetc.calc_exp_time(sequence_name=sequence_name, snr=5)
max_time, max_time_status, etc_max_time, etc_max_time_status = cgi_eetc.excam_saturation_time(sequence_name=sequence_name, g=5000, f=0.9, scale_bright=1.)

print('***Emccd:')
print('Mag. Target: {}'.format(mag))
print('num frames: {}'.format(num_frames))
print('exp_time_frame: {}'.format(exp_time_frame))
print('gain: {}'.format(gain))
print('max_time saturation off-axis: {}'.format(max_time))
print('')
print('')

# mag = 5+8*2.5
cgi_eetc_pc = CGIEETC(mag=mag, phot='v',spt='g2v')
num_frames, exp_time_frame, gain, snr_out, optflag = cgi_eetc_pc.calc_pc_exp_time(sequence_name=sequence_name, snr=5)
print('***Photon Counting Mode:')
print('Mag. Target: {}'.format(mag))
print('num frames: {}'.format(num_frames))
print('exp_time_frame: {}'.format(exp_time_frame))
print('gain: {}'.format(gain))

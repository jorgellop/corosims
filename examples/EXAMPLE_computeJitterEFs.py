import corgisims

if __name__ == '__main__':
    # cor_type = 'hlc_band4'
    # bandpass='4'
    cor_type = 'spc-wide_band1'
    bandpass='1'

    corgisim = corgisims.corgisims_core(cor_type=cor_type,bandpass=bandpass)
    
    corgisim.compute_jitter_EFs()
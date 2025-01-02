import corosims

if __name__ == '__main__':
    # cor_type = 'hlc_band4'
    # bandpass='4'
    cor_type = 'hlc_band1'
    bandpass='1'

    corosims = corosims.corosims_core(cor_type=cor_type,bandpass=bandpass)
    
    corosims.compute_jitter_EFs()
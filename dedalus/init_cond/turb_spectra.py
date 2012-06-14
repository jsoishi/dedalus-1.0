def mcwilliams_spec(k, k0, E0):
    """spectrum from McWilliams (1990), JFM 219:361-385

    """
    spec = k**6./(k + 2.*k0)**18.
    # normalize by 1D spectrum
    sl = (slice(0,k.shape[0]-1),slice(0,1))
    spec *= E0/spec[sl].sum()
    return spec

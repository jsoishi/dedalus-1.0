def mcwilliams_spec(k, k0):
    """spectrum from McWilliams (1990), JFM 219:361-385

    """
    spec = k**6./(k + 2.*k0)**18.
    return spec

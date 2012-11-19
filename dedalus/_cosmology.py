"""Old cosmology physics and IC's"""


#####################
## PHYSICS CLASSES ##
#####################

class LinearCollisionlessCosmology(Physics):
    """This class implements linear, collisionless cosmology.

    parameters
    ----------
    Omega_r - energy density in radiation
    Omega_m - energy density in matter
    Omega_l - energy density in dark energy (cosmological constant)
    H0      - hubble constant now (100 if all units are in km/s/Mpc)
    solves
    ------
    d_t d = - div(v_)
    d_t v = -grad(Phi) - H v
    laplacian(Phi) = 3 H^2 d / 2

    """
    def __init__(self, *args, **kwargs):
        Physics.__init__(self, *args, **kwargs)
        self.fields = [('delta', 'ScalarField'),
                       ('u', 'VectorField')]
        self._aux_fields = [('gradphi', 'VectorField')]

        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        params = {'Omega_r': 0.,
                  'Omega_m': 0.,
                  'Omega_l': 0.,
                  'H0': 100.}
        self._setup_parameters(params)
        self._setup_aux_fields(0., self._aux_fields)
        aux_eqn_rhs = a_friedmann
        self._setup_aux_eqns(['a'], [aux_eqn_rhs], [0.002],
                             [self.parameters])

        self._RHS = self.create_fields(0.)

    def RHS(self, data):
        self.density_RHS(data)
        self.vel_RHS(data)
        self._RHS.time = data.time
        return self._RHS

    def density_RHS(self, data):
        a = self.aux_eqns['a'].value
        divu = na.zeros(data['delta'].shape, complex)
        for i in self.dims:
            divu += data['u'][i].deriv(self._trans[i])

        self._RHS['delta']['kspace'] = -divu / a

    def vel_RHS(self, data):
        self.grad_phi(data)
        gradphi = self.aux_fields['gradphi']
        a = self.aux_eqns['a'].value
        adot = self.aux_eqns['a'].RHS(a)
        for i in self.dims:
            self._RHS['u'][i]['kspace'] = (-gradphi[i]['kspace'] -
                                            adot * data['u'][i]['kspace']) / a

    def grad_phi(self, data):
        a  = self.aux_eqns['a'].value
        H  = self.aux_eqns['a'].RHS(a) / a
        H0 = self.parameters['H0']
        Om = self.parameters['Omega_m']

        gradphi = self.aux_fields['gradphi']
        tmp = (-3./2. * H0*H0/a * Om * data['delta']['kspace'] /
                data['delta'].k2(no_zero=True))

        for i in self.dims:
            gradphi[i]['kspace'] = 1j * data['u'][i].k[self._trans[i]] * tmp

class CollisionlessCosmology(LinearCollisionlessCosmology):
    """This class implements collisionless cosmology with nonlinear terms.

    solves
    ------
    d_t d = -(1 + d) div(v_) / a - v dot grad(d) / a
    d_t v = -grad(Phi) / a - H v -(v dot grad) v / a
    laplacian(Phi) = 3 H^2 d / 2
    """

    def __init__(self, *args, **kwargs):
        LinearCollisionlessCosmology.__init__(self, *args, **kwargs)
        self._aux_fields.append(('math_tmp', 'ScalarField'))
        self._aux_fields.append(('ugraddelta', 'ScalarField'))
        self._aux_fields.append(('divu', 'ScalarField'))
        self._aux_fields.append(('deltadivu', 'ScalarField'))
        self._aux_fields.append(('ugradu', 'VectorField'))
        self._setup_aux_fields(0., self._aux_fields) # re-creates gradphi

    def delta_div_u(self, data, compute_divu=False):
        """calculate delta*div(v) in x-space, using 2/3 de-aliasing

        """
        divu = self.aux_fields['divu']
        if compute_divu:
            self.divX(data['u'], divu)
        divu = self.aux_fields['divu']
        deltadivu = self.aux_fields['deltadivu']

        deltadivu.zero()
        deltadivu['xspace'] += divu['xspace'] * data['delta']['xspace']

    def RHS(self, data):
        self.density_RHS(data)
        self.vel_RHS(data)
        self._RHS.time = data.time
        return self._RHS

    def density_RHS(self, data):
        a = self.aux_eqns['a'].value
        divu = self.aux_fields['divu']
        self.divX(data['u'], divu)
        self.XlistgradY([data['u'],], data['delta'],
                        self.aux_fields['math_tmp'],
                        [self.aux_fields['ugraddelta'],])
        self.delta_div_u(data)
        ugraddelta = self.aux_fields['ugraddelta']
        deltadivu = self.aux_fields['deltadivu']

        self._RHS['delta']['kspace'] = -(divu['kspace'] +
                                         deltadivu['kspace'] +
                                         ugraddelta['kspace']) / a

    def vel_RHS(self, data):
        self.grad_phi(data)
        gradphi = self.aux_fields['gradphi']

        a = self.aux_eqns['a'].value
        adot = self.aux_eqns['a'].RHS(a)

        self.XlistgradY([data['u'],], data['u'],
                        self.aux_fields['math_tmp'],
                        [self.aux_fields['ugradu'],])

        ugradu = self.aux_fields['ugradu']

        for i in self.dims:
            self._RHS['u'][i]['kspace'] = -(gradphi[i]['kspace'] +
                                            adot * data['u'][i]['kspace'] +
                                            ugradu[i]['kspace']) / a

class LinearBaryonCDMCosmology(Physics):
    """This class implements linear cosmology for coupled baryon and CDM fluids.

    parameters
    ----------
    Omega_r - energy density in radiation
    Omega_m - energy density in matter
    Omega_b - energy density in baryons
    Omega_c - energy density in CDM
    Omega_l - energy density in dark energy (cosmological constant)
    H0      - hubble constant now

    solves
    ------
    d_t d_c = -div(v_c) / a
    d_t v_c = -grad(Phi) / a - H v_c

    d_t d_b = -div(v_b) / a
    d_t v_b = -grad(Phi) / a - H v_b
              - c_s^2 grad(d_b) / a

    laplacian(Phi) = 3/2 H^2 (Omega_c * d_c + Omega_b * d_b)/Omega_m
    """

    def __init__(self, *args, **kwargs):
        Physics.__init__(self, *args, **kwargs)
        self.fields = [('delta_b', 'ScalarField'),
                       ('u_b', 'VectorField'),
                       ('delta_c', 'ScalarField'),
                       ('u_c', 'VectorField')]
        self._aux_fields = [('gradphi', 'VectorField')]
        self._aux_fields.append(('graddelta_b', 'VectorField'))
        self._aux_fields.append(('divu_b', 'ScalarField'))
        self._aux_fields.append(('divu_c', 'ScalarField'))
        self._setup_aux_fields(0., self._aux_fields)

        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        params = {'Omega_r': 0.,
                  'Omega_m': 0.,
                  'Omega_b': 0.,
                  'Omega_c': 0.,
                  'Omega_l': 0.,
                  'H0': 7.185e-5} # 70.3 km/s/Mpc in Myr^-1
        self._setup_parameters(params)
        self._setup_aux_fields(0., self._aux_fields)
        aux_eqn_rhs = a_friedmann
        self._setup_aux_eqns(['a'], [aux_eqn_rhs], [0.002],
                             [self.parameters])
        self._RHS = self.create_fields(0.)

        self._cs2_a = []
        self._cs2_values = []

    def init_cs2(self, a, cs2):
        self._cs2_values = cs2
        self._cs2_a = a

    def cs2_at(self, a):
        """look up cs2 in internal table at nearest scale factor. Uses
        binary search to find the correct index.

        """
        lower = 0
        upper = len(self._cs2_a)
        i = 0
        while True:
            if a > self._cs2_a[i]:
                lower = i
                i = lower + int(na.ceil((upper - lower)/2.))
            elif a < self._cs2_a[i]:
                upper = i
                i = upper - int(na.ceil((upper - lower)/2.))

            if i == len(self._cs2_a):
                print "warning: scale factor out of cs2 range; using sound speed at a = ", self._cs2_a[-1]
                return self._cs2_values[-1]

            if lower >= upper - 1:
                return self._cs2_values[i]

    def RHS(self, data):
        self.d_b_RHS(data)
        self.d_c_RHS(data)
        self.v_b_RHS(data)
        self.v_c_RHS(data)
        self._RHS.time = data.time
        return self._RHS

    def d_b_RHS(self, data):
        a = self.aux_eqns['a'].value
        u = data['u_b']
        delta = data['delta_b']
        divu = self.aux_fields['divu_b']

        self.divX(u, divu)

        self._RHS['delta_b']['kspace'] = -(divu['kspace']) / a

    def d_c_RHS(self, data):
        a = self.aux_eqns['a'].value
        u = data['u_c']
        delta = data['delta_c']
        divu = self.aux_fields['divu_c']

        self.divX(u, divu)

        self._RHS['delta_c']['kspace'] = -(divu['kspace']) / a

    def grad_phi(self, data):
        a = self.aux_eqns['a'].value
        H = self.aux_eqns['a'].RHS(a) / a
        H0 = self.parameters['H0']

        sampledata = data['delta_c']

        gradphi = self.aux_fields['gradphi']
        tmp = (-3./2. * H0*H0/a *
                ((self.parameters['Omega_c'] * data['delta_c']['kspace'] +
                  self.parameters['Omega_b'] * data['delta_b']['kspace'])) /
                sampledata.k2(no_zero=True))

        for i in self.dims:
            gradphi[i]['kspace'] = 1j * sampledata.k[self._trans[i]] * tmp

    def v_b_RHS(self, data):
        # needs pressure term
        self.grad_phi(data)
        gradphi = self.aux_fields['gradphi']

        a = self.aux_eqns['a'].value
        adot = self.aux_eqns['a'].RHS(a)

        u = data['u_b']
        graddelta = self.aux_fields['graddelta_b'] # calculated in d_b_RHS
        cs2 = self.cs2_at(a)
        for i in self.dims:
            self._RHS['u_b'][i]['kspace'] = -(gradphi[i]['kspace'] +
                                             adot * u[i]['kspace'] +
                                             cs2 * graddelta[i]['kspace']) / a

    def v_c_RHS(self, data):
        self.grad_phi(data)
        gradphi = self.aux_fields['gradphi']

        a = self.aux_eqns['a'].value
        adot = self.aux_eqns['a'].RHS(a)

        u = data['u_c']
        for i in self.dims:
            self._RHS['u_c'][i]['kspace'] = -(gradphi[i]['kspace'] +
                                             adot * u[i]['kspace']) / a

class BaryonCDMCosmology(Physics):
    """This class implements cosmology for coupled baryon and CDM fluids.

    parameters
    ----------
    Omega_r - energy density in radiation
    Omega_m - energy density in matter
    Omega_b - energy density in baryons
    Omega_c - energy density in CDM
    Omega_l - energy density in dark energy (cosmological constant)
    H0      - hubble constant now

    solves
    ------
    d_t d_c = -(1 + d_c) div(v_c) / a - v_c dot grad(d_c) / a
    d_t v_c = -grad(Phi) / a - H v_c -(v_c dot grad) v_c / a

    d_t d_b = -(1 + d_b) div(v_b) / a - v_b dot grad(d_b) / a
    d_t v_b = -grad(Phi) / a - H v_b -(v_b dot grad) v_b / a
              - c_s^2 grad(d_b) / a

    laplacian(Phi) = 3/2 H^2 (Omega_c * d_c + Omega_b * d_b)/Omega_m
    """

    def __init__(self, *args, **kwargs):
        Physics.__init__(self, *args, **kwargs)
        self.fields = [('delta_b', 'ScalarField'),
                       ('u_b', 'VectorField'),
                       ('delta_c', 'ScalarField'),
                       ('u_c', 'VectorField')]
        self._aux_fields = [('gradphi', 'VectorField')]
        self._aux_fields.append(('graddelta_b', 'VectorField'))
        self._aux_fields.append(('ugraddelta_b', 'ScalarField'))
        self._aux_fields.append(('ugraddelta_c', 'ScalarField'))
        self._aux_fields.append(('divu_b', 'ScalarField'))
        self._aux_fields.append(('divu_c', 'ScalarField'))
        self._aux_fields.append(('deltadivu_b', 'ScalarField'))
        self._aux_fields.append(('deltadivu_c', 'ScalarField'))
        self._aux_fields.append(('math_tmp', 'ScalarField'))
        self._aux_fields.append(('ugradu_b', 'VectorField'))
        self._aux_fields.append(('ugradu_c', 'VectorField'))
        self._aux_fields.append(('math_tmp', 'ScalarField'))
        self._setup_aux_fields(0., self._aux_fields)

        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        params = {'Omega_r': 0.,
                  'Omega_m': 0.,
                  'Omega_b': 0.,
                  'Omega_c': 0.,
                  'Omega_l': 0.,
                  'H0': 7.185e-5} # 70.3 km/s/Mpc in Myr^-1
        self._setup_parameters(params)
        self._setup_aux_fields(0., self._aux_fields)
        aux_eqn_rhs = a_friedmann
        self._setup_aux_eqns(['a'], [aux_eqn_rhs], [0.002],
                             [self.parameters])
        self._RHS = self.create_fields(0.)

        self._cs2_a = []
        self._cs2_values = []

    def init_cs2(self, a, cs2):
        self._cs2_values = cs2
        self._cs2_a = a

    def cs2_at(self, a):
        """look up cs2 in internal table at nearest scale factor. Uses
        binary search to find the correct index.

        """
        lower = 0
        upper = len(self._cs2_a)
        i = 0
        while True:
            if a > self._cs2_a[i]:
                lower = i
                i = lower + int(na.ceil((upper - lower)/2.))
            elif a < self._cs2_a[i]:
                upper = i
                i = upper - int(na.ceil((upper - lower)/2.))

            if i == len(self._cs2_a):
                print "warning: scale factor out of cs2 range; using sound speed at a = ", self._cs2_a[-1]
                return self._cs2_values[-1]

            if lower >= upper - 1:
                return self._cs2_values[i]

    def RHS(self, data):
        self.d_b_RHS(data)
        self.d_c_RHS(data)
        self.v_b_RHS(data)
        self.v_c_RHS(data)
        self._RHS.time = data.time
        return self._RHS

    def d_b_RHS(self, data):
        a = self.aux_eqns['a'].value
        u = data['u_b']
        delta = data['delta_b']
        divu = self.aux_fields['divu_b']
        graddelta = self.aux_fields['graddelta_b']
        ugraddelta = self.aux_fields['ugraddelta_b']
        deltadivu = self.aux_fields['deltadivu_b']

        self.divX(u, divu)
        # compute both graddelta and ugraddelta
        self.XgradY(u, delta, graddelta, ugraddelta)
        deltadivu.zero()
        deltadivu['xspace'] += divu['xspace'] * delta['xspace']

        self._RHS['delta_b']['kspace'] = -(divu['kspace'] +
                                           deltadivu['kspace'] +
                                           ugraddelta['kspace']) / a

    def d_c_RHS(self, data):
        a = self.aux_eqns['a'].value
        u = data['u_c']
        delta = data['delta_c']
        divu = self.aux_fields['divu_c']
        math_tmp = self.aux_fields['math_tmp']
        ugraddelta = self.aux_fields['ugraddelta_c']
        deltadivu = self.aux_fields['deltadivu_c']

        self.divX(u, divu)
        self.XlistgradY([u,], delta, math_tmp, [ugraddelta,])
        deltadivu.zero()
        deltadivu['xspace'] += divu['xspace'] * delta['xspace']

        self._RHS['delta_c']['kspace'] = -(divu['kspace'] +
                                           deltadivu['kspace'] +
                                           ugraddelta['kspace']) / a

    def grad_phi(self, data):
        a = self.aux_eqns['a'].value
        H = self.aux_eqns['a'].RHS(a) / a
        H0 = self.parameters['H0']

        sampledata = data['delta_c']

        gradphi = self.aux_fields['gradphi']
        tmp = (-3./2. * H0*H0/a *
                ((self.parameters['Omega_c'] * data['delta_c']['kspace'] +
                  self.parameters['Omega_b'] * data['delta_b']['kspace'])) /
                sampledata.k2(no_zero=True))

        for i in self.dims:
            gradphi[i]['kspace'] = 1j * sampledata.k[self._trans[i]] * tmp

    def v_b_RHS(self, data):
        # needs pressure term
        self.grad_phi(data)
        gradphi = self.aux_fields['gradphi']

        a = self.aux_eqns['a'].value
        adot = self.aux_eqns['a'].RHS(a)

        u = data['u_b']
        math_tmp = self.aux_fields['math_tmp']
        ugradu = self.aux_fields['ugradu_b']
        self.XlistgradY([u,], u, math_tmp, [ugradu,])
        graddelta = self.aux_fields['graddelta_b'] # calculated in d_b_RHS
        cs2 = self.cs2_at(a)
        for i in self.dims:
            self._RHS['u_b'][i]['kspace'] = -(gradphi[i]['kspace'] +
                                             adot * u[i]['kspace'] +
                                             ugradu[i]['kspace'] +
                                             cs2 * graddelta[i]['kspace']) / a

    def v_c_RHS(self, data):
        self.grad_phi(data)
        gradphi = self.aux_fields['gradphi']

        a = self.aux_eqns['a'].value
        adot = self.aux_eqns['a'].RHS(a)

        u = data['u_c']
        math_tmp = self.aux_fields['math_tmp']
        ugradu = self.aux_fields['ugradu_c']
        self.XlistgradY([u,], u, math_tmp, [ugradu,])
        for i in self.dims:
            self._RHS['u_c'][i]['kspace'] = -(gradphi[i]['kspace'] +
                                             adot * u[i]['kspace'] +
                                             ugradu[i]['kspace']) / a


########################
## INITIAL CONDITIONS ##
########################

def zeldovich(data, ampl, A, a_ini, a_cross):
    """velocity wave IC, for testing nonlinear collisionless cosmology
    against the Zeldovich approximation
    """
    k = 2*na.pi/data.length[0]
    N = data['delta'].shape[0]
    volfac = N**1.5
    D = 1.
    #A = a_ini/(a_cross * k) # Only true for EdS
    x = na.array([i - N/2 for i in xrange(N)])*data.length[0]/N
    func = lambda y: na.array(y) + D*A*na.sin(k*na.array(y)) - x
    left = na.array([min(x),]*N)
    right = na.array([max(x),]*N)
    q = find_zero(func, left, right)
    delta1d = (1./(1. + D*A*k*na.cos(k*q)) - 1.)
    for i in xrange(N):
        for j in xrange(N):
            data['delta']['xspace'][i,j,:] = delta1d

    ampl = ampl * volfac
    data['u'][0]['kspace'][0,0,1] = ampl * 1j / 2
    data['u'][0]['kspace'][0,0,-1] = -data['u'][0]['kspace'][0,0,1]

def read_linger_transfer_data(fname, ak_trans, Ttot, Tdc, Tdb, dTvc, dTvb, dTtot, Ttot0):
    """read values from a transfer-mode linger++ output file

    """
    infile = open(fname)
    for i,line in enumerate(infile):
        values = line.split()
        ak_trans.append(float(values[0]))
        Ttot.append(float(values[1]))
        Tdc.append(float(values[2]))
        Tdb.append(float(values[3]))
        dTvc.append(float(values[4]))
        dTvb.append(float(values[5]))
        dTtot.append(float(values[6]))
        Ttot0.append(float(values[7]))
    infile.close()

def sig2_integrand(Ttot0, akoh, nspect):
    """calculate the integrand in the expression for the mean square
    of the overdensity field, smoothed with a top-hat window function W
    at 8 Mpc/h.

    """
    R = 8.
    x = akoh*R
    w = 3. * (na.sin(x) - x*na.cos(x))/(x*x*x)
    Pk = (akoh**nspect) * (Ttot0*Ttot0)
    return (w*w) * Pk * (akoh*akoh)

def get_normalization(Ttot0, ak, sigma_8, nspect, h):
    """calculate the normalization for delta_tot using sigma_8

    """
    akoh = ak/h
    integrand = sig2_integrand(Ttot0, akoh, nspect)
    sig2 = 4.*na.pi*integrate_quad(integrand, akoh)
    ampl = sigma_8/na.sqrt(sig2)
    return ampl

def collisionless_cosmo_fields(delta, u, spec_delta, spec_u, mean=0., stdev=1.):
    """create realization of cosmological initial conditions by
    filling 3-d k-space fields with values sampled from gaussians with
    amplitudes given by spec.

    """
    shape = spec_delta.shape
    rand = (na.random.normal(mean, stdev, shape) + 1j*na.random.normal(mean, stdev, shape))
    delta['kspace'] = spec_delta * rand
    hermitianize.enforce_hermitian(delta['kspace'])
    delta.dealias()

    for i in xrange(3):
        u[i]['kspace'] = rand * spec_u[i]
        hermitianize.enforce_hermitian(u[i]['kspace'])
        u[i].dealias()

    return rand

def cosmo_fields(delta_c, u_c, delta_b, u_b, spec_delta_c, spec_u_c, spec_delta_b, spec_u_b):
    """create realization of baryon and CDM initial conditions

    """
    # use the same random number field as for delta_c
    rand = collisionless_cosmo_fields(delta_c, u_c, spec_delta_c, spec_u_c)
    delta_b['kspace'] = spec_delta_b * rand
    hermitianize.enforce_hermitian(delta_b['kspace'])
    delta_b.dealias()

    for i in xrange(3):
        u_b[i]['kspace'] = rand * spec_u_b[i]
        hermitianize.enforce_hermitian(u_b[i]['kspace'])
        u_b[i].dealias()

def cosmo_spectra(data, norm_fname, a, nspect=0.961, sigma_8=0.811, h=.703, baryons=False, f_nl=None):
    """generate spectra for CDM overdensity and velocity from linger++
    output. Assumes 3-dimensional fields.

    Length: Mpc
    Time:   Myr (linger++ uses Mpc/c)

    """

    Myr_per_Mpc = 0.3063915366 # 1 c/Mpc = 0.306... 1/Myr

    ak    = [] # the k-values that go with transfer-functions
    Ttot  = [] # linear transfer of total spectrum at initial time
    Ttot0 = [] # ... of total spectrum at z = 0, used with sigma_8 for norm
    Tdc   = [] # transfer of delta_CDM
    Tdb   = [] # transfer of delta_baryons
    dTvc  = [] # rate of change of CDM transfer
    dTvb  = [] # rate of change of baryon transfer
    dTtot = [] # ... of total spectrum

    read_linger_transfer_data(norm_fname, ak, Ttot, Tdc, Tdb, dTvc, dTvb, dTtot, Ttot0)
    ak = na.array(ak) * h

    if baryons:
        deltacp = na.array(Tdc)
        thetac  = na.array(dTvc) #* (h/299792.458) # linger multiplies by c/h
    else:
        deltacp = na.array(Ttot)
        thetac  = na.array(dTtot)


    Ttot0 = na.array(Ttot0)

    # ... get sample data object for shape and k-values
    sampledata = data.fields.values()[0][0]
    shape = sampledata.shape

    nk = shape[0]
    k2 = sampledata.k2()
    kzero = (k2==0)
    k2[kzero] = 1. # may cause problems for non-gaussian IC...
    kk = na.sqrt(k2)
    maxkk = na.sqrt(3*na.max(sampledata.kny)**2)
    maxkinput = max(ak)
    if maxkk > maxkinput:
        mylog.warning('cannot interpolate: some grid wavenumbers larger than input wavenumbers; ICs for those modes are wrong')
        # ... any |k| larger than the max input k is replaced by max input k
        kk[:,:,:] = na.minimum(kk, maxkinput*na.ones_like(kk))

    # ... normalize
    ampl = get_normalization(Ttot0, ak, sigma_8, nspect, h)
    ampl = ampl * (2.*na.pi/sampledata.length[0])**1.5 * (sampledata.shape[0])**1.5 # *h**1.5

    f_deltacp = interp_linear(na.log10(ak), na.log10(deltacp))

    # ... calculate spectra
    if f_nl is not None:
        """ Primordial non-Gaussianity:
        phi(k) = -3/2 H^2/c^2 Omega_m a^2 A k^(n_s/2)/k^2
        phi_NG(x) = phi(x) + f_NL(phi)
        delta(k) = (-3/2 H^2/c^2 Omega_m a^2)^-1 * k^2 phi_NG(k) T_delta(k)
        """
        Omega_m = 0.276 # not necessarily...
        H = 0.452997
        c = Myr_per_Mpc
        to_phi = (-3/2.) * H*H / c*c * Omega_m * a*a
        phi = FourierRepresentation(None, kk.shape, sampledata.length, dtype='float128')
        phi['kspace'] = to_phi * ampl * kk**(nspect/2.)/k2
        phi_ng['xspace'] = phi['xspace'] + f_nl(phi['xspace'])
        f_deltacp = interp_linear(ak, deltacp, kind='cubic')
        spec_delta = k2 * phi_ng['kspace'] * f_deltacp(kk) / to_phi
    else:
        # ... delta = delta_transfer * |k|^(n_s/2)
        spec_delta = kk**(nspect/2.)*(10.**f_deltacp(na.log10(kk)))*ampl
    spec_delta[kzero] = 0.

    vunit = 1.02268944e-6 # km/s in Mpc/Myr
    thetac = thetac * ampl * vunit


    # ... calculate spectra
    # u_j = -i * k_j/|k| * theta * |k|^(n_s/2 - 1)
    f_thetac = interp_linear(na.log10(ak), na.log10(thetac), kind='linear')
    spec_vel = 1j*kk**(nspect/2. -1.) * 10.**f_thetac(na.log10(kk)) # isotropic
    spec_vel[kzero] = 0.

    spec_u = [na.zeros_like(spec_vel),]*3
    for i,dim in enumerate(['x','y','z']):
        spec_u[i] = (sampledata.k[dim]/kk) * spec_vel

    if baryons:
        deltabp = na.array(Tdb)
        thetab  = na.array(dTvb)

        deltabp = deltabp * ampl
        thetab  = thetab * ampl * vunit

        f_deltabp = interp_linear(na.log10(ak), na.log10(deltabp), kind='linear')
        spec_delta_b = kk**(nspect/2.)*10.**f_deltabp(na.log10(kk))
        spec_delta_b[kzero] = 0.

        f_thetab = interp_linear(na.log10(ak), na.log10(thetab), kind='linear')
        spec_vel_b = 1j * kk**(nspect/2. - 1.) * 10.**f_thetab(na.log10(kk))
        spec_vel_b[kzero] = 0.

        spec_u_b = [na.zeros_like(spec_vel_b),]*3
        for i,dim in enumerate(['x','y','z']):
            spec_u_b[i] = (sampledata.k[dim]/kk) * spec_vel_b

        return spec_delta, spec_u, spec_delta_b, spec_u_b

    return spec_delta, spec_u



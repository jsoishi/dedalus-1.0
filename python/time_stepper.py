class TimeStepper:
    """copied from pencil code

    """
    def __init__(self,RHS,itorder=3):
        self.RHS = RHS
        self.itorder = itorder
        if self.itorder==1:
            self.alpha_ts=np.array([ 0., 0., 0. ])
            self.beta_ts =np.array([ 1., 0., 0. ])
        elif self.itorder==2:
            self.alpha_ts=np.array([ 0., -1./2., 0. ])
            self.beta_ts=np.array([ 1./2.,  1.,  0. ])
        elif self.itorder==3:
            #alpha_ts=np.array([0., -2./3., -1.])
            #beta_ts=np.array([1./3., 1., 1./2.])
            #  use coefficients of Williamson (1980)
            self.alpha_ts=np.array([  0. ,  -5./9., -153./128. ])
            self.beta_ts=np.array([ 1./3., 15./16.,    8./15.  ])
        else:
            print "not implemented, stupid."
        
    def rk2n(self,t,f,dt):
        for itsub in range(self.itorder):
            llast = (itsub == self.itorder)

            if itsub == 0:
                lfirst = True
                df = 0.
                ds = 0.
            else:
                lfirst = False
                df = self.alpha_ts[itsub]*df
                ds = self.alpha_ts[itsub]*ds
            
            df += self.RHS(f)
            ds = ds + 1.
            dt_beta_ts = dt*self.beta_ts

            f += dt_beta_ts[itsub]*df
            t +=  dt_beta_ts[itsub]*ds
        return t,f

class TimeStepBase(object):
    def __init__(self, RHS, CFL=0.4, int_factor=None):
        """a very simple Runga-Kutta 2 integrator

        """
        self.RHS = RHS
        self.CFL = CFL
        self.int_factor = int_factor
        self._is_ok = True
        self.iter = 0

    @property
    def ok(self):
        if self.iter >= self.stop_iter:
            self._is_ok = False

        return self._is_ok

    def stop_time(self,t):
        self.stop_time = t

    def stop_iter(self, it):
        self.stop_iter = it

    def stop_walltime(self, wt):
        self.stop_wall = wt

    def final_stats(self):
        print "Simulation complete. Status: awesome"


class RK2simple(TimeStepBase):
    def advance(self, data, dt):
        """
        from NR:
          k1 = h * RHS(x_n, y_n)
          k2 = h * RHS(x_n + 1/2*h, y_n + 1/2*k1)
          y_n+1 = y_n + k2 +O(h**3)
        """
        tmp_fields = self.RHS.create_fields(data.time)
        field_dt = self.RHS.create_fields(data.time)

        # first step
        for f in self.RHS.fields:
            field_dt[f] = self.RHS.RHS(data)[f]['kspace']
            tmp_fields[f] = data[f]['kspace'] + dt/2. * field_dt[f]['kspace']

        # second step
        for f in self.RHS.fields:
            field_dt[f] = self.RHS.RHS(tmp_fields)[f]['kspace']
            data[f] = data[f]['kspace'] + dt * field_dt[f]['kspace']
        data.time += dt
        self.iter += 1

    


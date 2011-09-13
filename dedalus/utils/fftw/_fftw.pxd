cdef extern from "complex.h":
    pass

cdef extern from "fftw3.h":
    ctypedef struct fftw_plan_s:
        pass
    ctypedef fftw_plan_s *fftw_plan
    ctypedef struct fftw_iodim:
        int n
        int ins "is"
        int ous "os"

    fftw_plan fftw_plan_dft_2d(int n0,
                               int n1,
                               complex* in_,
                               complex* out_,
                               int sign,
                               unsigned flags)
    fftw_plan fftw_plan_dft_3d(int n0,
                               int n1,
                               int n2,
                               complex* in_,
                               complex* out_,
                               int sign,
                               unsigned flags)
    fftw_plan fftw_plan_many_dft(int rank,
                                 int* n_,
                                 int howmany,
                                 complex* in_,
                                 int* inembed_,
                                 int istride,
                                 int idist,
                                 complex* out_,
                                 int *onembed_,
                                 int ostride,
                                 int odist,
                                 int sign,
                                 unsigned flags)
    fftw_plan fftw_plan_guru_dft(int rank,
                                 fftw_iodim *dims,
                                 int howmany_rank,
                                 fftw_iodim *howmany_dims,
                                 complex *in_,
                                 complex *out,
                                 int sign,
                                 unsigned flags)

    void fftw_execute(fftw_plan plan)
    void fftw_destroy_plan(fftw_plan plan)

cdef enum:
    FFTW_FORWARD = -1
    FFTW_BACKWARD = +1
    FFTW_MEASURE = 0
    FFTW_DESTROY_INPUT =  (1 << 0)
    FFTW_UNALIGNED = (1 << 1)
    FFTW_CONSERVE_MEMORY = (1 << 2)
    FFTW_EXHAUSTIVE = (1 << 3) # /* NO_EXHAUSTIVE is default */
    FFTW_PRESERVE_INPUT = (1 << 4) # /* cancels FFTW_DESTROY_INPUT */
    FFTW_PATIENT = (1 << 5) # /* IMPATIENT is default */
    FFTW_ESTIMATE = (1 << 6)


    

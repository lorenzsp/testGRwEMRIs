import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../include/Utility.hh":
    void KerrGeoCoordinateFrequenciesVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length);
    
    void KerrEqSpinFrequenciesCorrVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length);

    void get_separatrix_vector(double* separatrix, double* a, double* e, double* x, int length);

    void KerrGeoConstantsOfMotionVectorized(double* E_out, double* L_out, double* Q_out, double* a, double* p, double* e, double* x, int n);
    void ELQ_to_pexVectorised(double *p, double *e, double *x, double *a, double *E, double *Lz, double *Q, int length)
    void Y_to_xI_vector(double* x, double* a, double* p, double* e, double* Y, int length);
    void set_threads(int num_threads);
    int get_threads();
    void KerrEquatorialFrequencyDerivative(double *omegaPhi_dp, double *omegaPhi_de, double *omegaR_dp, double *omegaR_de, double a, double p, double e);

def pyKerrGeoCoordinateFrequencies(np.ndarray[ndim=1, dtype=np.float64_t] a,
                                   np.ndarray[ndim=1, dtype=np.float64_t] p,
                                   np.ndarray[ndim=1, dtype=np.float64_t] e,
                                   np.ndarray[ndim=1, dtype=np.float64_t] x):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaPhi = np.zeros(len(p), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaTheta = np.zeros(len(p), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaR = np.zeros(len(p), dtype=np.float64)

    KerrGeoCoordinateFrequenciesVectorized(&OmegaPhi[0], &OmegaTheta[0], &OmegaR[0],
                                &a[0], &p[0], &e[0], &x[0], len(p))
    return (OmegaPhi, OmegaTheta, OmegaR)

def pyKerrEqSpinFrequenciesCorr(np.ndarray[ndim=1, dtype=np.float64_t] a,
                                   np.ndarray[ndim=1, dtype=np.float64_t] p,
                                   np.ndarray[ndim=1, dtype=np.float64_t] e,
                                   np.ndarray[ndim=1, dtype=np.float64_t] x):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaPhi = np.zeros(len(p), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaTheta = np.zeros(len(p), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] OmegaR = np.zeros(len(p), dtype=np.float64)

    KerrEqSpinFrequenciesCorrVectorized(&OmegaPhi[0], &OmegaTheta[0], &OmegaR[0],
                                &a[0], &p[0], &e[0], &x[0], len(p))
    return (OmegaPhi, OmegaTheta, OmegaR)


def pyGetSeparatrix(np.ndarray[ndim=1, dtype=np.float64_t] a,
                    np.ndarray[ndim=1, dtype=np.float64_t] e,
                    np.ndarray[ndim=1, dtype=np.float64_t] x):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] separatrix = np.zeros_like(e)

    get_separatrix_vector(&separatrix[0], &a[0], &e[0], &x[0], len(e))

    return separatrix

def pyKerrGeoConstantsOfMotionVectorized(np.ndarray[ndim=1, dtype=np.float64_t]  a,
                                         np.ndarray[ndim=1, dtype=np.float64_t]  p,
                                         np.ndarray[ndim=1, dtype=np.float64_t]  e,
                                         np.ndarray[ndim=1, dtype=np.float64_t]  x):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] E_out = np.zeros_like(e)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] L_out = np.zeros_like(e)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] Q_out = np.zeros_like(e)

    KerrGeoConstantsOfMotionVectorized(&E_out[0], &L_out[0], &Q_out[0], &a[0], &p[0], &e[0], &x[0], len(e))

    return (E_out, L_out, Q_out)

def pyELQ_to_pex(np.ndarray[ndim=1, dtype=np.float64_t] a,
                                   np.ndarray[ndim=1, dtype=np.float64_t] E,
                                   np.ndarray[ndim=1, dtype=np.float64_t] Lz,
                                   np.ndarray[ndim=1, dtype=np.float64_t] Q):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] p = np.zeros(len(E), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] e = np.zeros(len(E), dtype=np.float64)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] x = np.zeros(len(E), dtype=np.float64)

    ELQ_to_pexVectorised(&p[0], &e[0], &x[0], &a[0], &E[0], &Lz[0], &Q[0], len(E))
    return (p, e, x)

def pyY_to_xI_vector(np.ndarray[ndim=1, dtype=np.float64_t] a,
                     np.ndarray[ndim=1, dtype=np.float64_t] p,
                     np.ndarray[ndim=1, dtype=np.float64_t] e,
                     np.ndarray[ndim=1, dtype=np.float64_t] Y):

    cdef np.ndarray[ndim=1, dtype=np.float64_t] x = np.zeros_like(e)

    Y_to_xI_vector(&x[0], &a[0], &p[0], &e[0], &Y[0], len(e))

    return x

def set_threads_wrap(num_threads):
    set_threads(num_threads)

def get_threads_wrap():
    return get_threads()

def pyKerrEqDerivFrequenciesPhiR(a_in, p_in, e_in):
    cdef double OmegaPhi_dp, OmegaPhi_de, OmegaR_dp, OmegaR_de
    cdef double a = a_in
    cdef double p = p_in
    cdef double e = e_in
    KerrEquatorialFrequencyDerivative(&OmegaPhi_dp, &OmegaPhi_de, &OmegaR_dp, &OmegaR_de, a, p, e)
    return (OmegaPhi_dp, OmegaPhi_de, OmegaR_dp, OmegaR_de)

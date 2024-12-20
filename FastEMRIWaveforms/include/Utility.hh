#ifndef __FUND_FREQS__
#define __FUND_FREQS__

int sanity_check(double a, double p, double e, double Y);

void KerrGeoMinoFrequencies(double* CapitalGamma_, double* CapitalUpsilonPhi_, double* CapitalUpsilonTheta_, double* CapitalUpsilonr_,
                              double a, double p, double e, double x);

void KerrGeoCoordinateFrequencies(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                            double a, double p, double e, double x);

void KerrGeoEquatorialMinoFrequencies(double* CapitalGamma_, double* CapitalUpsilonPhi_, double* CapitalUpsilonTheta_, double* CapitalUpsilonr_,
                              double a, double p, double e, double x);

void KerrGeoEquatorialCoordinateFrequencies(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                            double a, double p, double e, double x);

void KerrGeoCoordinateFrequenciesVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length);

void KerrEqSpinFrequenciesCorrVectorized(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,
                              double* a, double* p, double* e, double* x, int length);

void KerrEqSpinFrequenciesCorrection(double* deltaOmegaR_, double* deltaOmegaPhi_,
                              double a, double p, double e, double x);
void KerrScott(double* OmegaPhi_, double* OmegaTheta_, double* OmegaR_,double a, double p, double e, double xI);
void SchwarzschildGeoCoordinateFrequencies(double* OmegaPhi, double* OmegaR, double p, double e);

double get_separatrix(double a, double e, double x);
void get_separatrix_vector(double* separatrix, double* a, double* e, double* x, int length);

void KerrGeoConstantsOfMotionVectorized(double* E_out, double* L_out, double* Q_out, double* a, double* p, double* e, double* x, int n);
void KerrGeoConstantsOfMotion(double* E_out, double* L_out, double* Q_out, double a, double p, double e, double x);

void ELQ_to_pexVectorised(double* p, double* e, double* x, double* a, double* E, double* Lz, double* Q, int length);
void ELQ_to_pex(double* p, double* e, double* x, double a, double E, double Lz, double Q);

double Y_to_xI(double a, double p, double e, double Y);
void Y_to_xI_vector(double* x, double* a, double* p, double* e, double* Y, int length);

void set_threads(int num_threads);
int get_threads();

double KerrGeoEnergy(double a, double p, double e, double x);
double KerrGeoAngularMomentum(double a, double p, double e, double x, double En);

double separatrix_KerrEquatorial(const double a, const double e);

void KerrEquatorialFrequencyDerivative(double *omegaPhi_dp, double *omegaPhi_de, double *omegaR_dp, double *omegaR_de, double a, double p, double e);
#endif // __FUND_FREQS__
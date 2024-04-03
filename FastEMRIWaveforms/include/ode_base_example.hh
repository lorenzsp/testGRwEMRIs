#define KERR 1
#define SCHWARZSCHILD 2

#include "Interpolant.h"

// Used to pass the interpolants to the ODE solver
struct interp_params{
	double epsilon;
	Interpolant *Edot;
	Interpolant *Ldot;
};

class SchwarzEccFlux{
public:
    interp_params *interps;
    Interpolant *amp_vec_norm_interp;
    double test;
    bool equatorial = true;
    int background = SCHWARZSCHILD;
    bool circular = false;
    bool integrate_constants_of_motion = false;
    bool integrate_phases = true;

    SchwarzEccFlux(std::string few_dir);

    void deriv_func(double ydot[], const double y[], double epsilon, double a, double *additional_args);
    ~SchwarzEccFlux();
};

class KerrEccentricEquatorial{
public:

    bool equatorial = true;
    int background = KERR;
    bool circular = false;
    bool integrate_constants_of_motion = true;
    bool integrate_phases = false;

    KerrEccentricEquatorial(std::string few_dir);

    void deriv_func(double ydot[], const double y[], double epsilon, double a, double *additional_args);
    ~KerrEccentricEquatorial();

};


class KerrEccentricEquatorialAPEX{
public:

    bool equatorial = true;
    int background = KERR;
    bool circular = false;
    bool integrate_constants_of_motion = false;
    bool integrate_phases = true;

    KerrEccentricEquatorialAPEX(std::string few_dir);

    void deriv_func(double ydot[], const double y[], double epsilon, double a, double *additional_args);
    ~KerrEccentricEquatorialAPEX();

};
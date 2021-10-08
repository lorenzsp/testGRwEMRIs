
    #ifndef __ODE__
    #define __ODE__

    #include "global.h"
    #include <cstring>

    #define __deriv__

    #include "Interpolant.h"

// Used to pass the interpolants to the ODE solver
struct interp_params{
	double epsilon;
	Interpolant *Edot;
	Interpolant *Ldot;
    Interpolant *ScalarInt;
};

class SchwarzEccFlux{
public:
    interp_params *interps;
    Interpolant *amp_vec_norm_interp;
    double test;

    SchwarzEccFlux(std::string few_dir);

    void deriv_func(double* pdot, double* edot, double* Ydot,
                      double Omega_phi, double Omega_theta, double Omega_r,
                      double epsilon, double a, double p, double e, double Y, double* additional_args);
    ~SchwarzEccFlux();
};



class KerrCircFlux{
public:
    interp_params *interps;
    Interpolant *amp_vec_norm_interp;
    double test;
    KerrCircFlux(std::string few_dir);


    double EdotPN(double r, double a);


    void deriv_func(double* pdot, double* edot, double* Ydot,
                      double Omega_phi, double Omega_theta, double Omega_r,
                      double epsilon, double a, double p, double e, double Y, double* additional_args);
    ~KerrCircFlux();
};


            class pn5{
            public:
                double test;

                pn5(std::string few_dir);

                void deriv_func(double* pdot, double* edot, double* Ydot,
                                  double Omega_phi, double Omega_theta, double Omega_r,
                                  double epsilon, double a, double p, double e, double Y, double* additional_args);
                ~pn5();
            };

        

            class CircEqLdot5pnAcc{
            public:
                double test;

                CircEqLdot5pnAcc(std::string few_dir);

                void deriv_func(double* pdot, double* edot, double* Ydot,
                                  double Omega_phi, double Omega_theta, double Omega_r,
                                  double epsilon, double a, double p, double e, double Y, double* additional_args);
                ~CircEqLdot5pnAcc();
            };

        

            class CircEqEdot5pn{
            public:
                double test;

                CircEqEdot5pn(std::string few_dir);

                void deriv_func(double* pdot, double* edot, double* Ydot,
                                  double Omega_phi, double Omega_theta, double Omega_r,
                                  double epsilon, double a, double p, double e, double Y, double* additional_args);
                ~CircEqEdot5pn();
            };

        

            class CircEqPdot5pn{
            public:
                double test;

                CircEqPdot5pn(std::string few_dir);

                void deriv_func(double* pdot, double* edot, double* Ydot,
                                  double Omega_phi, double Omega_theta, double Omega_r,
                                  double epsilon, double a, double p, double e, double Y, double* additional_args);
                ~CircEqPdot5pn();
            };

        

    class ODECarrier{
        public:
            std::string func_name;
            std::string few_dir;
            void* func;
            ODECarrier(std::string func_name_, std::string few_dir_);
            ~ODECarrier();
            void get_derivatives(double* pdot, double* edot, double* Ydot,
                              double Omega_phi, double Omega_theta, double Omega_r,
                              double epsilon, double a, double p, double e, double Y, double* additional_args);

    };

    #endif // __ODE__

    
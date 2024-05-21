#ifndef DEDT_CHEB
#define DEDT_CHEB

/*
  Header of the Interpolated fluxes
*/

//! \file KerrEquatorial.h

// Define macro 

// Define type

// Declare prototype 
double edot_Cheby_full(const double a, const double e, const double r);
double pdot_Cheby_full(const double a, const double e, const double r);
double Ldot_SC(const double a, const double e, const double r, const double p);
double Edot_SC(const double a, const double e, const double r, const double p);
double Edot_GR(const double a, const double e, const double r, const double p);
double Ldot_GR(const double a, const double e, const double r, const double p);
void Jac(const double a, const double p, const double ecc, const double xi,
		  const double E, const double Lz, const double Q,
		  const double Edot, const double Lzdot, const double Qdot,
		  double & pdot, double & eccdot, double & xidot);
void pdot_edot_from_fluxes(double & pdot, double & eccdot, const double Edot, const double Ldot, const double a, const double e, const double p);
// equations 267 from https://arxiv.org/pdf/1106.6313 
double delta_E_B2(double eta, double p, double e, double gamma_12_plus_2_gamma_42);
double delta_L_B2(double eta, double p, double e, double theta_tp, double gamma_12_plus_2_gamma_42);
double delta_Q_B2(double eta, double p, double e, double theta_tp, double gamma_12_plus_2_gamma_42);
double delta_E_B3(double eta, double p, double e, double gamma_13_plus_2_gamma_43);
double delta_L_B3(double eta, double p, double e, double theta_tp, double gamma_13_plus_2_gamma_43);
double delta_Q_B3(double eta, double p, double e, double theta_tp, double gamma_13_plus_2_gamma_43);
double delta_E_B4(double eta, double p, double e, double gamma_14_plus_2_gamma_44);
double delta_L_B4(double eta, double p, double e, double theta_tp, double gamma_14_plus_2_gamma_44);
double delta_Q_B4(double eta, double p, double e, double theta_tp, double gamma_14_plus_2_gamma_44);
double delta_E_B5(double eta, double p, double e, double gamma_15_plus_2_gamma_45);
double delta_L_B5(double eta, double p, double e, double theta_tp, double gamma_15_plus_2_gamma_45);
double delta_Q_B5(double eta, double p, double e, double theta_tp, double gamma_15_plus_2_gamma_45);
double delta_E_CS(double eta, double zeta, double a, double p, double e, double theta_tp);
double delta_L_CS(double eta, double zeta, double a, double p, double e, double theta_tp);
double delta_Q_CS(double eta, double zeta, double a, double p, double e, double theta_tp);
#endif // DEDT_CHEB

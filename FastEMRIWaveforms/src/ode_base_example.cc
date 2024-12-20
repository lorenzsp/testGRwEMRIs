#include <math.h>
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_sf_ellint.h>
#include <algorithm>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <complex>
#include <cmath>

#include "Interpolant.h"
#include "global.h"
#include "Utility.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip> // std::setprecision
#include <cstring>

#include "dIdt8H_5PNe10.h"
#include "ode.hh"
#include "KerrEquatorial.h"

#define pn5_Y
#define pn5_citation1 Pn5_citation
__deriv__ void pn5(double ydot[], const double y[], double epsilon, double a, double *additional_args)
{
    // evaluate ODEs
    double p = y[0];
    double e = y[1];
    double Y = y[2];

    double Omega_phi, Omega_theta, Omega_r;

    // the frequency variables are pointers!
    double x = Y_to_xI(a, p, e, Y);
    KerrGeoCoordinateFrequencies(&Omega_phi, &Omega_theta, &Omega_r, a, p, e, x);

    int Nv = 10;
    int ne = 10;
    double pdot = epsilon * dpdt8H_5PNe10(a, p, e, Y, Nv, ne);

    // needs adjustment for validity
    Nv = 10;
    ne = 8;
    double edot = epsilon * dedt8H_5PNe10(a, p, e, Y, Nv, ne);

    Nv = 7;
    ne = 10;
    double Ydot = epsilon * dYdt8H_5PNe10(a, p, e, Y, Nv, ne);

    ydot[0] = pdot;
    ydot[1] = edot;
    ydot[2] = Ydot;
    ydot[3] = Omega_phi;
    ydot[4] = Omega_theta;
    ydot[5] = Omega_r;
}

// Initialize flux data for inspiral calculations
void load_and_interpolate_flux_data(struct interp_params *interps, const std::string &few_dir)
{

    // Load and interpolate the flux data
    std::string fp = "few/files/FluxNewMinusPNScaled_fixed_y_order.dat";
    fp = few_dir + fp;
    ifstream Flux_file(fp);

    if (Flux_file.fail())
    {
        throw std::runtime_error("The file FluxNewMinusPNScaled_fixed_y_order.dat did not open sucessfully. Make sure it is located in the proper directory (Path/to/Installation/few/files/).");
    }

    // Load the flux data into arrays
    string Flux_string;
    vector<double> ys, es, Edots, Ldots;
    double y, e, Edot, Ldot;
    while (getline(Flux_file, Flux_string))
    {

        stringstream Flux_ss(Flux_string);

        Flux_ss >> y >> e >> Edot >> Ldot;

        ys.push_back(y);
        es.push_back(e);
        Edots.push_back(Edot);
        Ldots.push_back(Ldot);
    }

    // Remove duplicate elements (only works if ys are perfectly repeating with no round off errors)
    sort(ys.begin(), ys.end());
    ys.erase(unique(ys.begin(), ys.end()), ys.end());

    sort(es.begin(), es.end());
    es.erase(unique(es.begin(), es.end()), es.end());

    Interpolant *Edot_interp = new Interpolant(ys, es, Edots);
    Interpolant *Ldot_interp = new Interpolant(ys, es, Ldots);

    interps->Edot = Edot_interp;
    interps->Ldot = Ldot_interp;
}

// Class to carry gsl interpolants for the inspiral data
// also executes inspiral calculations
SchwarzEccFlux::SchwarzEccFlux(std::string few_dir)
{
    interps = new interp_params;

    // prepare the data
    // python will download the data if
    // the user does not have it in the correct place
    load_and_interpolate_flux_data(interps, few_dir);
    // load_and_interpolate_amp_vec_norm_data(&amp_vec_norm_interp, few_dir);
}

#define SchwarzEccFlux_num_add_args 0
#define SchwarzEccFlux_spinless
#define SchwarzEccFlux_equatorial
#define SchwarzEccFlux_file1 FluxNewMinusPNScaled_fixed_y_order.dat
__deriv__ void SchwarzEccFlux::deriv_func(double ydot[], const double y[], double epsilon, double a, double *additional_args)
{
    double p = y[0];
    double e = y[1];
    double Y = y[2];

    double Omega_phi, Omega_theta, Omega_r;

    if ((6.0 + 2. * e) > p)
    {
        ydot[0] = 0.0;
        ydot[1] = 0.0;
        ydot[2] = 0.0;
        return;
    }

    SchwarzschildGeoCoordinateFrequencies(&Omega_phi, &Omega_r, p, e);
    Omega_theta = Omega_phi;

    double y1 = log((p - 2. * e - 2.1));

    // evaluate ODEs, starting with PN contribution, then interpolating over remaining flux contribution

    double yPN = pow((Omega_phi), 2. / 3.);

    double EdotPN = (96 + 292 * Power(e, 2) + 37 * Power(e, 4)) / (15. * Power(1 - Power(e, 2), 3.5)) * pow(yPN, 5);
    double LdotPN = (4 * (8 + 7 * Power(e, 2))) / (5. * Power(-1 + Power(e, 2), 2)) * pow(yPN, 7. / 2.);

    double Edot = -epsilon * (interps->Edot->eval(y1, e) * pow(yPN, 6.) + EdotPN);

    double Ldot = -epsilon * (interps->Ldot->eval(y1, e) * pow(yPN, 9. / 2.) + LdotPN);

    double pdot = (-2 * (Edot * Sqrt((4 * Power(e, 2) - Power(-2 + p, 2)) / (3 + Power(e, 2) - p)) * (3 + Power(e, 2) - p) * Power(p, 1.5) + Ldot * Power(-4 + p, 2) * Sqrt(-3 - Power(e, 2) + p))) / (4 * Power(e, 2) - Power(-6 + p, 2));

    double edot;

    // handle e = 0.0
    if (e > 0.)
    {
        edot = -((Edot * Sqrt((4 * Power(e, 2) - Power(-2 + p, 2)) / (3 + Power(e, 2) - p)) * Power(p, 1.5) *
                      (18 + 2 * Power(e, 4) - 3 * Power(e, 2) * (-4 + p) - 9 * p + Power(p, 2)) +
                  (-1 + Power(e, 2)) * Ldot * Sqrt(-3 - Power(e, 2) + p) * (12 + 4 * Power(e, 2) - 8 * p + Power(p, 2))) /
                 (e * (4 * Power(e, 2) - Power(-6 + p, 2)) * p));
    }
    else
    {
        edot = 0.0;
    }

    double xdot = 0.0;

    ydot[0] = pdot;
    ydot[1] = edot;
    ydot[2] = xdot;
    ydot[3] = Omega_phi;
    ydot[4] = Omega_theta;
    ydot[5] = Omega_r;
}

// destructor
SchwarzEccFlux::~SchwarzEccFlux()
{

    delete interps->Edot;
    delete interps->Ldot;
    delete interps;
}
//--------------------------------------------------------------------------------
Vector fill_vector(std::string fp){
	ifstream file_x(fp);

    if (file_x.fail())
    {
        throw std::runtime_error("The file  did not open sucessfully. Make sure it is located in the proper directory.");
    }
    else{
        // cout << "importing " + fp << endl;
    }

	// Load the flux data into arrays
	string string_x;
    Vector xs;
	double x;
	while(getline(file_x, string_x)){
		stringstream ss(string_x);
		ss >> x ;
		xs.push_back(x);
	}
    return xs;

}

KerrEccentricEquatorial::KerrEccentricEquatorial(std::string few_dir)
{
    std::string fp;

    // fp = few_dir + "few/files/sep_x0.dat";
    // Vector sep_x1 = fill_vector(fp);
    // fp = few_dir + "few/files/sep_x1.dat";
    // Vector sep_x2 = fill_vector(fp);
    // fp = few_dir + "few/files/coeff_sep.dat";
    // Vector coeffSep = fill_vector(fp);
    // Sep_interp = new TensorInterpolant2d(sep_x1, sep_x2, coeffSep);    
}

// #define KerrEccentricEquatorial_Y
#define KerrEccentricEquatorial_equatorial
#define KerrEccentricEquatorial_num_add_args 1
__deriv__ void KerrEccentricEquatorial::deriv_func(double ydot[], const double y[], double epsilon, double a, double *additional_args)
{

    double E = y[0];
    double Lz = y[1];
    double Q = y[2];

    double p, e, x;

    ELQ_to_pex(&p, &e, &x, a, E, Lz, Q);
    if (isnan(p)||isnan(e))
    {
        ydot[0] = 0.0;
        ydot[1] = 0.0;
        ydot[2] = 0.0;
        // ydot[3] = 0.0;
        // ydot[4] = 0.0;
        // ydot[5] = 0.0;
        return;
    }
    // double signed_a = a*x; // signed a for interpolants
    // double w = sqrt(e);
    // double ymin = pow(1.-0.998,1./3.);
    // double ymax = pow(1.+0.998,1./3.);
    // double chi2 = (pow(1.-signed_a,1./3.) - ymin) / (ymax - ymin);
    // double p_sep = Sep_interp->eval(chi2, w) * (6. + 2.*e);

    // cout << "beginning" << " a =" << a  << "\t" << "p=" <<  p << "\t" << "e=" << e << "\t" << "x=" << x << endl;
    // cout << "beginning" << " E =" << E  << "\t" << "L=" <<  Lz << "\t" << "Q=" << Q << endl;
    double p_sep = get_separatrix(a, e, x);
    // cout <<" p_sep =" <<p_sep << endl;
    

    // make sure we do not step into separatrix
    if ((e < 0.0) || (p < p_sep))
    {
        ydot[0] = 0.0;
        ydot[1] = 0.0;
        ydot[2] = 0.0;
        // ydot[3] = 0.0;
        // ydot[4] = 0.0;
        // ydot[5] = 0.0;
        return;
    }
    double Omega_phi, Omega_theta, Omega_r;
    double pdot, edot, xdot;

    // evaluate ODEs
    
    // auto start = std::chrono::high_resolution_clock::now();
    // the frequency variables are pointers!
    if (e==0.0){
        Omega_phi = 1.0/ (a*copysign(1.0,x) + pow(p, 1.5) );
    }
    else{
        KerrGeoEquatorialCoordinateFrequencies(&Omega_phi, &Omega_theta, &Omega_r, a, p, e, x); // shift to avoid problem in fundamental frequencies
    }
    
    // KerrScott(Omega_phi, Omega_theta, Omega_r, a, p, e, x);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double>  msec = end-start;
    // std::cout << "elapsed time fund freqs: " << msec.count() << "s\n";
    
    double r,Omega_phi_sep_circ;
    // reference frequency
    
    Omega_phi_sep_circ = 1.0/ (a*copysign(1.0,x) + pow(p_sep/( 1.0 + e ), 1.5) );
    r = pow(abs(Omega_phi)/ Omega_phi_sep_circ, 2.0/3.0 ) * (1.0 + e);
    
    if (isnan(r)){
        cout << " a =" << a  << "\t" << "p=" <<  p << "\t" << "e=" << e <<  "\t" << "x=" << x << "\t" << r << " plso =" <<  p_sep << endl;
        cout << "omegaphi circ " <<  Omega_phi_sep_circ << " omegaphi " <<  Omega_phi << " omegar " <<  Omega_r <<endl;
        throw std::exception();
        pdot = 0.0;
        edot = 0.0;
        xdot = 0.0;
        return;
        }

    
    double risco = get_separatrix(a, 0.0, x);
    double one_minus_e2 = 1. - pow(e,2);

    // Fluxes in p,e from Chebyshev
    // double pdot_cheb, edot_cheb;
    // pdot_cheb = pdot_Cheby_full(a*copysign(1.0,x), e, r) * ((8.*pow(one_minus_e2,1.5)*(8. + 7.*pow(e,2)))/(5.*p*(pow(p - risco,2) - pow(-risco + p_sep,2))));
    // edot_cheb = edot_Cheby_full(a*copysign(1.0,x), e, r) * ((pow(one_minus_e2,1.5)*(304. + 121.*pow(e,2)))/(15.*pow(p,2)*(pow(p - risco,2) - pow(-risco + p_sep,2))));

    double Edot, Ldot, Qdot, pdot_here, edot_here, xdot_here, E_here, L_here, Q_here;
    // KerrGeoConstantsOfMotion(&E_here, &L_here, &Q_here, a, p, e, x);
    
    // Transform to pdot, edot for the scalar fluxes
    
    // Jac(a, p, e, x, E_here, L_here, Q_here, Edot, Ldot, Qdot, pdot_here, edot_here, xdot_here);
    // pdot_edot_from_fluxes(pdot_here, edot_here, Edot, Ldot, a, e, p);
    // Fluxes in E,L from Chebyshev
    double pdot_out, edot_out, xdot_out;
    // sign of function
    double factor = additional_args[0]*additional_args[0]/4.0;
    // cout << factor << endl;
    Edot = additional_args[0]*Edot_SC(a*copysign(1.0,x), e, r, p)+Edot_GR(a*copysign(1.0,x),e,r,p);
    Ldot = additional_args[0]*Ldot_SC(a*copysign(1.0,x), e, r, p)*copysign(1.0,x)+Ldot_GR(a*copysign(1.0,x),e,r,p)*copysign(1.0,x);
    
    Qdot = 0.0;
    // cout << 'Edot \t' << Edot << endl;
    // cout << 'Ldot \t' << Ldot << endl;
    // Jac(a, p, e, x, E, Lz, Q, -Edot, -Ldot, -Qdot, pdot_out, edot_out, xdot_out);
    // pdot_edot_from_fluxes(pdot_out, edot_out, -Edot_GR(a,e,r,p), -Ldot_GR(a,e,r,p), a, e, p);

    // check Jacobiam
    // cout << "ratio " <<  pdot_cheb/pdot_out << endl;
    // cout << "ratio " <<  edot_cheb/edot_out << endl;

    // cout << "Edot, pdot " <<  Edot << "\t" << pdot_out << endl;
    // cout << "Ldot, edot " <<  Ldot << "\t" << edot_out << endl;

    ydot[0] = -epsilon*Edot;
    ydot[1] = -epsilon*Ldot;
    ydot[2] = -epsilon*Qdot;
    // // needs adjustment for validity
    // if (e > 1e-8)
    // {
    //     // the scalar flux is d^2 /4
    //     pdot = epsilon * pdot_out;
    //     edot = epsilon * edot_out;
    // }
    // else{
        
    //     edot = 0.0;
    //     pdot = epsilon * pdot_out;
    //     // cout << "end" << " a =" << a  << "\t" << "p=" <<  p << "\t" << "e=" << e <<  "\t" << "x=" << x << "\t" << r << " plso =" <<  p_sep << endl;
    // }

    // xdot = 0.0;
    // cout << "end" << endl;
    // ydot[0] = pdot;
    // ydot[1] = edot;
    // ydot[2] = xdot;
    // ydot[3] = Omega_phi;
    // ydot[4] = Omega_theta;
    // ydot[5] = Omega_r;
    // delete GKR;
    return;
}

// destructor
KerrEccentricEquatorial::~KerrEccentricEquatorial()
{

    // delete Sep_interp;
}



KerrEccentricEquatorialAPEX::KerrEccentricEquatorialAPEX(std::string few_dir)
{
    std::string fp;
}

// #define KerrEccentricEquatorialAPEX_Y
#define KerrEccentricEquatorialAPEX_equatorial
#define KerrEccentricEquatorialAPEX_num_add_args 1
__deriv__ void KerrEccentricEquatorialAPEX::deriv_func(double ydot[], const double y[], double epsilon, double a, double *additional_args)
{

    double p = y[0];
    double e = y[1];
    double x = y[2];

    double p_sep = get_separatrix(a, e, x);
    // make sure we do not step into separatrix
    if ((e < 0.0) || (p < p_sep+0.1))
    {
        ydot[0] = 0.0;
        ydot[1] = 0.0;
        ydot[2] = 0.0;
        ydot[3] = 0.0;
        ydot[4] = 0.0;
        ydot[5] = 0.0;
        return;
    }
    double Omega_phi, Omega_theta, Omega_r;
    double pdot, edot, xdot;

    // evaluate ODEs
    // cout << "beginning" << " a =" << a  << "\t" << "p=" <<  p << "\t" << "e=" << e << "\t" << "x=" << x << endl;
    // auto start = std::chrono::high_resolution_clock::now();
    // the frequency variables are pointers!

    KerrGeoEquatorialCoordinateFrequencies(&Omega_phi, &Omega_theta, &Omega_r, a, p, e, x); // shift to avoid problem in fundamental frequencies
    // KerrScott(Omega_phi, Omega_theta, Omega_r, a, p, e, x);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double>  msec = end-start;
    // std::cout << "elapsed time fund freqs: " << msec.count() << "s\n";
    
    double r,Omega_phi_sep_circ;
    // reference frequency
    
    Omega_phi_sep_circ = 1.0/ (a*copysign(1.0,x) + pow(p_sep/( 1.0 + e ), 1.5) );
    r = pow(abs(Omega_phi)/ Omega_phi_sep_circ, 2.0/3.0 ) * (1.0 + e);
    
    if (isnan(r)){
        cout << " a =" << a  << "\t" << "p=" <<  p << "\t" << "e=" << e <<  "\t" << "x=" << x << "\t" << r << " plso =" <<  p_sep << endl;
        cout << "omegaphi circ " <<  Omega_phi_sep_circ << " omegaphi " <<  Omega_phi << " omegar " <<  Omega_r <<endl;
        throw std::exception();
        }

    
    double risco = get_separatrix(a, 0.0, x);
    double one_minus_e2 = 1. - pow(e,2);

    // Fluxes in p,e from Chebyshev
    // double pdot_cheb, edot_cheb;
    // pdot_cheb = pdot_Cheby_full(a*copysign(1.0,x), e, r) * ((8.*pow(one_minus_e2,1.5)*(8. + 7.*pow(e,2)))/(5.*p*(pow(p - risco,2) - pow(-risco + p_sep,2))));
    // edot_cheb = edot_Cheby_full(a*copysign(1.0,x), e, r) * ((pow(one_minus_e2,1.5)*(304. + 121.*pow(e,2)))/(15.*pow(p,2)*(pow(p - risco,2) - pow(-risco + p_sep,2))));

    double Edot, Ldot, Qdot, pdot_here, edot_here, xdot_here, E_here, L_here, Q_here;
    KerrGeoConstantsOfMotion(&E_here, &L_here, &Q_here, a, p, e, x);
    
    // Transform to pdot, edot for the scalar fluxes
    
    // Jac(a, p, e, x, E_here, L_here, Q_here, Edot, Ldot, Qdot, pdot_here, edot_here, xdot_here);
    // pdot_edot_from_fluxes(pdot_here, edot_here, Edot, Ldot, a, e, p);
    // Fluxes in E,L from Chebyshev
    double pdot_out, edot_out, xdot_out;
    // sign of function
    // double factor = additional_args[0]*additional_args[0]/4.0;
    // cout << factor << endl;
    
    // Edot = factor*Edot_SC(a*copysign(1.0,x), e, r, p)+Edot_GR(a*copysign(1.0,x),e,r,p);
    // Ldot = factor*Ldot_SC(a*copysign(1.0,x), e, r, p)*copysign(1.0,x)+Ldot_GR(a*copysign(1.0,x),e,r,p)*copysign(1.0,x);
    // Qdot = 0.0;
    Edot = additional_args[0]*Edot_SC(a*copysign(1.0,x), e, r, p)+Edot_GR(a*copysign(1.0,x),e,r,p);
    Ldot = additional_args[0]*Ldot_SC(a*copysign(1.0,x), e, r, p)*copysign(1.0,x)+Ldot_GR(a*copysign(1.0,x),e,r,p)*copysign(1.0,x);
    Qdot = 0.0;
    // cout << 'Edot \t' << Edot << endl;
    // cout << 'Ldot \t' << Ldot << endl;
    Jac(a, p, e, x, E_here, L_here, Q_here, -Edot, -Ldot, Qdot, pdot_out, edot_out, xdot_out);
    // pdot_edot_from_fluxes(pdot_out, edot_out, -Edot_GR(a,e,r,p), -Ldot_GR(a,e,r,p), a, e, p);

    // check Jacobiam
    // cout << "ratio " <<  pdot_cheb/pdot_out << endl;
    // cout << "ratio " <<  edot_cheb/edot_out << endl;

    // cout << "Edot, pdot " <<  Edot << "\t" << pdot_out << endl;
    // cout << "Ldot, edot " <<  Ldot << "\t" << edot_out << endl;

    
    // needs adjustment for validity
    // if (e > 1e-8)
    // {
        // the scalar flux is d^2 /4
        pdot = epsilon * pdot_out;
        edot = epsilon * edot_out;
    // }
    // else{
        
    //     edot = 0.0;
    //     pdot = epsilon * pdot_out;
    //     // cout << "end" << " a =" << a  << "\t" << "p=" <<  p << "\t" << "e=" << e <<  "\t" << "x=" << x << "\t" << r << " plso =" <<  p_sep << endl;
    // }

    xdot = 0.0;
    
    ydot[0] = pdot;
    ydot[1] = edot;
    ydot[2] = xdot;
    ydot[3] = Omega_phi;
    ydot[4] = Omega_theta;
    ydot[5] = Omega_r;
    // delete GKR;
    return;
}

KerrEccentricEquatorialAPEX::~KerrEccentricEquatorialAPEX()
{

    // delete Sep_interp;
}

// -----------------------------------------------------------------------------
// Bumpy Black Holes
KerrEccentricEquatorialBumpy::KerrEccentricEquatorialBumpy(std::string few_dir)
{
    std::string fp;
}

// #define KerrEccentricEquatorialBumpy_Y
#define KerrEccentricEquatorialBumpy_equatorial
#define KerrEccentricEquatorialBumpy_num_add_args 2
__deriv__ void KerrEccentricEquatorialBumpy::deriv_func(double ydot[], const double y[], double epsilon, double a, double *additional_args)
{

    double p = y[0];
    double e = y[1];
    double x = y[2];

    double p_sep = get_separatrix(a, e, x);
    // make sure we do not step into separatrix
    if ((e < 0.0) || (p < p_sep+0.1))
    {
        ydot[0] = 0.0;
        ydot[1] = 0.0;
        ydot[2] = 0.0;
        ydot[3] = 0.0;
        ydot[4] = 0.0;
        ydot[5] = 0.0;
        return;
    }
    double Omega_phi, Omega_theta, Omega_r;
    double pdot, edot, xdot;

    KerrGeoEquatorialCoordinateFrequencies(&Omega_phi, &Omega_theta, &Omega_r, a, p, e, x); // shift to avoid problem in fundamental frequencies

    
    double r,Omega_phi_sep_circ;
    // reference frequency
    
    Omega_phi_sep_circ = 1.0/ (a*copysign(1.0,x) + pow(p_sep/( 1.0 + e ), 1.5) );
    r = pow(abs(Omega_phi)/ Omega_phi_sep_circ, 2.0/3.0 ) * (1.0 + e);
    
    if (isnan(r)){
        cout << " a =" << a  << "\t" << "p=" <<  p << "\t" << "e=" << e <<  "\t" << "x=" << x << "\t" << r << " plso =" <<  p_sep << endl;
        cout << "omegaphi circ " <<  Omega_phi_sep_circ << " omegaphi " <<  Omega_phi << " omegar " <<  Omega_r <<endl;
        throw std::exception();
        }

    
    double risco = get_separatrix(a, 0.0, x);
    double one_minus_e2 = 1. - pow(e,2);
    double Edot, Ldot, Qdot, pdot_here, edot_here, xdot_here, E_here, L_here, Q_here;
    KerrGeoConstantsOfMotion(&E_here, &L_here, &Q_here, a, p, e, x);
    double pdot_out, edot_out, xdot_out;

    double theta_tp = PI/2.0;
    
    double Edot_correction, Ldot_correction;
    if (additional_args[1]==2.0){
        Edot_correction = delta_E_B2(1.0, p, e, additional_args[0]);
        Ldot_correction = delta_L_B2(1.0, p, e, theta_tp, additional_args[0]);
    }
    else if (additional_args[1]==3.0){
        Edot_correction = delta_E_B3(1.0, p, e, additional_args[0]);
        Ldot_correction = delta_L_B3(1.0, p, e, theta_tp, additional_args[0]);
    }
    else if (additional_args[1]==4.0){
        Edot_correction = delta_E_B4(1.0, p, e, additional_args[0]);
        Ldot_correction = delta_L_B4(1.0, p, e, theta_tp, additional_args[0]);
    }
    else if (additional_args[1]==5.0){
        Edot_correction = delta_E_B5(1.0, p, e, additional_args[0]);
        Ldot_correction = delta_L_B5(1.0, p, e, theta_tp, additional_args[0]);
    }
    else{
        
        Edot_correction = delta_E_CS(1.0, additional_args[0], a, p, e, theta_tp);
        Ldot_correction = delta_L_CS(1.0, additional_args[0], a, p, e, theta_tp);
    }

    Edot = Edot_correction+Edot_GR(a*copysign(1.0,x),e,r,p);
    Ldot = Ldot_correction*copysign(1.0,x)+Ldot_GR(a*copysign(1.0,x),e,r,p)*copysign(1.0,x);
    Qdot = 0.0;

    Jac(a, p, e, x, E_here, L_here, Q_here, -Edot, -Ldot, Qdot, pdot_out, edot_out, xdot_out);
    // pdot_edot_from_fluxes(pdot_out, edot_out, -Edot_GR(a,e,r,p), -Ldot_GR(a,e,r,p), a, e, p);

    // needs adjustment for validity
    if (e > 1e-8)
    {
        // the scalar flux is d^2 /4
        pdot = epsilon * pdot_out;
        edot = epsilon * edot_out;
    }
    else{
        
        edot = 0.0;
        pdot = epsilon * pdot_out;
    }

    xdot = 0.0;
    
    ydot[0] = pdot;
    ydot[1] = edot;
    ydot[2] = xdot;
    ydot[3] = Omega_phi;
    ydot[4] = Omega_theta;
    ydot[5] = Omega_r;
    // delete GKR;
    return;
}

KerrEccentricEquatorialBumpy::~KerrEccentricEquatorialBumpy()
{

    // delete Sep_interp;
}


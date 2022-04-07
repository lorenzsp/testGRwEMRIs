#include "Interpolant.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <math.h>  

using namespace std;


// Used to pass the interpolants to the ODE solver
struct interp_params{
	double epsilon;
	Interpolant *Edot;
	Interpolant *Ldot;
};



// Initialize flux data for inspiral calculations
void load_and_interpolate_flux_data(struct interp_params *interps){

	// Load and interpolate the flux data
    std::string fp = "../SSF_ecc_flux_scaled.dat";
	ifstream Flux_file(fp);

    if (Flux_file.fail())
    {
        throw std::runtime_error("The file SSF_ecc_flux_scaled.dat did not open successfully.");
    }

	// Load the flux data into arrays
	string Flux_string;
	vector<double> ys, es, Edots, Ldots;
	double y, e, Edot, Ldot;
	while(getline(Flux_file, Flux_string)){

		stringstream Flux_ss(Flux_string);

		Flux_ss >> y >> e >> Edot >> Ldot;

		ys.push_back(y);
		es.push_back(e);
		Edots.push_back(Edot);
		Ldots.push_back(Ldot);
	}

	// Remove duplicate elements (only works if ys are perfectly repeating with no round off errors)
	sort( ys.begin(), ys.end() );
	ys.erase( unique( ys.begin(), ys.end() ), ys.end() );

	sort( es.begin(), es.end() );
	es.erase( unique( es.begin(), es.end() ), es.end() );

	Interpolant *Edot_interp = new Interpolant(ys, es, Edots);
	Interpolant *Ldot_interp = new Interpolant(ys, es, Ldots);

	interps->Edot = Edot_interp;
	interps->Ldot = Ldot_interp;

}


int main (int argc, char *argv[]){

    interp_params *interps = new interp_params;
	load_and_interpolate_flux_data(interps);
	
	double p = 7.85;
	double e = 0.475;
	cout << "For p = " << p << " and e = " << e << endl;
	cout << "Edot = " << interps->Edot->eval(p-6-2*e,e)/pow(p,4) << endl;
	cout << "Ldot = " << interps->Ldot->eval(p-6-2*e,e)/pow(p,2.5) << endl;
	

	return 0;

}
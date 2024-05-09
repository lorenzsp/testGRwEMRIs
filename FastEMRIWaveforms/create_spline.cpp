// /data/lsperi/miniconda3/envs/bgr_env/bin/x86_64-conda-linux-gnu-c++ -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /data/lsperi/miniconda3/envs/bgr_env/include -fPIC -O2 -isystem /data/lsperi/miniconda3/envs/bgr_env/include -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /data/lsperi/miniconda3/envs/bgr_env/include -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /data/lsperi/miniconda3/envs/bgr_env/include -fPIC -I/data/lsperi/miniconda3/envs/bgr_env/lib/python3.10/site-packages/numpy/core/include -Iinclude -I/data/lsperi/miniconda3/envs/bgr_env/include/python3.10 create_spline.cpp -o create_spline -std=c++11 -lgsl -lgslcblas 
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf_ellint.h>
#include <fstream>
#include <cmath>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

double EllipticK(double k)
{
    gsl_sf_result result;
    // cout << "CHECK1" << endl;
    int status = gsl_sf_ellint_Kcomp_e(sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticK failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticE(double k)
{
    gsl_sf_result result;
    // cout << "CHECK3 " << k << endl;
    int status = gsl_sf_ellint_Ecomp_e(sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticE failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double EllipticPi(double n, double k)
{
    // cout << "CHECK6" << endl;
    gsl_sf_result result;
    int status = gsl_sf_ellint_Pcomp_e(sqrt(k), -n, GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        printf("55: %e\n", k);
        sprintf(str, "EllipticPi failed with arguments (k,n): (%e,%e)", k, n);
        throw std::invalid_argument(str);
    }
    return result.val;
}

void generate_spline(const char* function_name, double (*function)(double), const char* filename) {
    // Step 1: Generate data points for the function
    int num_points = 100;
    double x_vals[num_points];
    double y_vals[num_points];
    for (int i = 0; i < num_points; ++i) {
        double x = 1e-10 + (0.99999 - 1e-10) * i / (num_points - 1);  // x goes from 1e-10 to 0.99999
        x_vals[i] = x;
        gsl_sf_result result;
        function(x);
        y_vals[i] = result.val;
    }

    // Step 2: Use these data points to create a cubic spline
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, num_points);
    gsl_spline_init(spline, x_vals, y_vals, num_points);

    // Step 3: Write the coefficients to a file
    std::ofstream file(filename);
    file << "#include <cmath>\n\n";
    file << "double interpolated" << function_name << "(double x) {\n";
    file << "    if (x < 1e-10 || x > 0.99999) return NAN;\n";  // Return NaN for out-of-range values

    double first_x = x_vals[0];
    double spacing = x_vals[1] - x_vals[0];  // Assuming regular spacing

    file << "    int index = static_cast<int>((x - " << first_x << ") / " << spacing << ");\n";
    file << "    switch (index) {\n";

    for (int i = 0; i < num_points - 1; ++i) {
        double xi = x_vals[i];
        double xi1 = x_vals[i+1];
        double ai = gsl_spline_eval(spline, xi, NULL);
        double bi = gsl_spline_eval_deriv(spline, xi, NULL);
        double ci = gsl_spline_eval_deriv2(spline, xi, NULL) / 2.0;
        double di = (gsl_spline_eval_deriv2(spline, xi1, NULL) - gsl_spline_eval_deriv2(spline, xi, NULL)) / (6.0 * (xi1 - xi));
        file << "        case " << i << ":\n";
        file << "            return " << ai << " + " << bi << " * (x - " << xi << ") + " << ci << " * pow(x - " << xi << ", 2) + " << di << " * pow(x - " << xi << ", 3);\n";
    }

    file << "        default:\n";
    file << "            return NAN;\n";  // Return NaN for out-of-range values
    file << "    }\n";
    file << "}\n";
    file.close();

    // Don't forget to free the spline when you're done with it
    gsl_spline_free(spline);
}

void generate_bicubic_spline(const char* function_name, double (*function)(double, double), const char* filename) {
    // Step 1: Generate data points for the function
    int num_points = 100;
    double x_vals[num_points];
    double y_vals[num_points];
    double z_vals[num_points * num_points];  // z values are stored in a 1D array

    for (int i = 0; i < num_points; ++i) {
        double x = 1e-10 + (0.99999 - 1e-10) * i / (num_points - 1);  // x goes from 1e-10 to 0.99999
        x_vals[i] = x;
        for (int j = 0; j < num_points; ++j) {
            double y = 1e-10 + (0.99999 - 1e-10) * j / (num_points - 1);  // y goes from 1e-10 to 0.99999
            y_vals[j] = y;
            gsl_sf_result result;
            function(x, y);
            z_vals[i * num_points + j] = result.val;
        }
    }

    // Step 2: Use these data points to create a bicubic spline
    gsl_spline2d *spline = gsl_spline2d_alloc(gsl_interp2d_bicubic, num_points, num_points);
    gsl_spline2d_init(spline, x_vals, y_vals, z_vals, num_points, num_points);

    // Step 3: Write the coefficients to a file
    std::ofstream file(filename);
    file << "#include <cmath>\n\n";
    file << "double interpolated" << function_name << "(double x, double y) {\n";
    file << "    if (x < 1e-10 || x > 0.99999 || y < 1e-10 || y > 0.99999) return NAN;\n";  // Return NaN for out-of-range values

    double first_x = x_vals[0];
    double first_y = y_vals[0];
    double spacing_x = x_vals[1] - x_vals[0];  // Assuming regular spacing
    double spacing_y = y_vals[1] - y_vals[0];  // Assuming regular spacing

    file << "    int index_x = static_cast<int>((x - " << first_x << ") / " << spacing_x << ");\n";
    file << "    int index_y = static_cast<int>((y - " << first_y << ") / " << spacing_y << ");\n";
    file << "    switch (index_x * " << num_points << " + index_y) {\n";

    // Create the gsl_interp_accel objects
    gsl_interp_accel *xacc = gsl_interp_accel_alloc();
    gsl_interp_accel *yacc = gsl_interp_accel_alloc();

    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_points; ++j) {
            double xi = x_vals[i];
            double yj = y_vals[j];
            double z = gsl_spline2d_eval(spline, xi, yj, xacc, yacc);
            file << "        case " << (i * num_points + j) << ":\n";
            file << "            return " << z << ";\n";
        }
    }


    file << "        default:\n";
    file << "            return NAN;\n";  // Return NaN for out-of-range values
    file << "    }\n";
    file << "}\n";
    file.close();

    // Don't forget to free the spline when you're done with it
    gsl_spline2d_free(spline);

    // Don't forget to free the gsl_interp_accel objects when you're done with them
    gsl_interp_accel_free(xacc);
    gsl_interp_accel_free(yacc);
}

int main() {
    generate_spline("EllipticK", EllipticK, "spline_coefficients_K.cpp");
    generate_spline("EllipticE", EllipticE, "spline_coefficients_E.cpp");
    generate_bicubic_spline("EllipticPi", EllipticPi, "spline_coefficients_Pi.cpp");

    return 0;
}
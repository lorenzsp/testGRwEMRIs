#include "stdio.h"
#include "math.h"
#include "global.h"
#include <stdexcept>
#include "Utility.hh"
#include <iostream>
#include <chrono>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include "Python.h"

#ifdef __USE_OMP__
#include "omp.h"
#endif

using namespace std;
using namespace std::chrono;
// ----------------------------------------------------------------------------
// define new elliptic pi function from Scott

// Reasonable maximum
template<class NUMBER>
inline NUMBER Max(const NUMBER a, const NUMBER b) {
  return (a > b ? a : b);
};

#define DMAX(a,b) Max(a,b)
#define FMAX(a,b) Max(a,b)
#define LMAX(a,b) Max(a,b)
#define IMAX(a,b) Max(a,b)

// Reasonable minimum
template<class NUMBER>
inline NUMBER Min(const NUMBER a, const NUMBER b) {
  return (a < b ? a : b);
};

#define DMIN(a,b) Min(a,b)
#define FMIN(a,b) Min(a,b)
#define LMIN(a,b) Min(a,b)
#define IMIN(a,b) Min(a,b)

void Die(const char error_text[])
{
  cerr << error_text << endl;
  exit(0);
}

double rf(const double x, const double y, const double z)
{
    // Define constants inside the function
    const double ERRTOL = 0.0025;
    const double TINY = 1.5e-38;
    const double BIG = 3.0e37;
    const double THIRD = 1.0/3.0;
    const double C1 = 1.0/24.0;
    const double C2 = 0.1;
    const double C3 = 3.0/44.0;
    const double C4 = 1.0/14.0;

    double alamb, ave, delx, dely, delz, e2, e3, sqrtx, sqrty, sqrtz,
    xt, yt, zt;

    if (DMIN(DMIN(x, y), z) < 0.0 || DMIN(DMIN(x + y, x + z), y + z) < TINY ||
        DMAX(DMAX(x, y), z) > BIG) Die("invalid arguments in rf");
    xt = x;
    yt = y;
    zt = z;
    do {
        sqrtx = sqrt(xt);
        sqrty = sqrt(yt);
        sqrtz = sqrt(zt);
        alamb = sqrtx*(sqrty + sqrtz) + sqrty*sqrtz;
        xt = 0.25*(xt + alamb);
        yt = 0.25*(yt + alamb);
        zt = 0.25*(zt + alamb);
        ave = THIRD*(xt + yt + zt);
        delx = (ave - xt)/ave;
        dely = (ave - yt)/ave;
        delz = (ave - zt)/ave;
    } while (DMAX(DMAX(fabs(delx), fabs(dely)), fabs(delz)) > ERRTOL);
    e2 = delx*dely - delz*delz;
    e3 = delx*dely*delz;
    return (1.0 + (C1*e2 - C2 - C3*e3)*e2 + C4*e3)/sqrt(ave);
}

double rc(const double x, const double y)
{
    // Define constants inside the function
    const double ERRTOL = 0.0012;
    const double TINY = 1.69e-38;
    const double SQRTNY = 1.3e-19;
    const double BIG = 3.e37;
    const double TNBG = (TINY*BIG);
    const double COMP1 = (2.236/SQRTNY);
    const double COMP2 = (TNBG*TNBG/25.0);
    const double THIRD = 1.0/3.0;
    const double C1 = 0.3;
    const double C2 = 1.0/7.0;
    const double C3 = 0.375;
    const double C4 = 9.0/22.0;

    double alamb, ave, s, w, xt, yt;
    if (x < 0.0 || y == 0.0 || (x+fabs(y)) < TINY || (x + fabs(y)) > BIG ||
        (y < -COMP1 && x > 0.0 && x < COMP2)) Die("invalid arguments in rc");
    if (y > 0.0) {
        xt = x;
        yt = y;
        w = 1.0;
    } else {
        xt = x-y;
        yt = -y;
        w = sqrt(x)/sqrt(xt);
    }
    do {
        alamb = 2.0*sqrt(xt)*sqrt(yt) + yt;
        xt = 0.25*(xt + alamb);
        yt = 0.25*(yt + alamb);
        ave = THIRD*(xt + yt + yt);
        s = (yt - ave)/ave;
    } while (fabs(s) > ERRTOL);
    return w*(1.0 + s*s*(C1 + s*(C2 + s*(C3 + s*C4))))/sqrt(ave);
}

double rj(const double x, const double y, const double z, const double p)
{
    // Define constants inside the function
    const double ERRTOL = 0.0015;
    const double TINY = 2.5e-13;
    const double BIG = 9.0e11;
    const double C1 = 3.0/14.0;
    const double C2 = 1.0/3.0;
    const double C3 = 3.0/22.0;
    const double C4 = 3.0/26.0;
    const double C5 = 0.75*C3;
    const double C6 = 1.5*C4;
    const double C7 = 0.5*C2;
    const double C8 = C3 + C3;

    double a, alamb, alpha, ans, ave, b, beta, delp, delx, dely, delz, 
    ea, eb, ec, ed, ee, fac, pt, rcx, rho, sqrtx, sqrty, sqrtz, sum,
    tau, xt, yt, zt;

    if (DMIN(DMIN(x, y), z) < 0.0 || DMIN(DMIN(x + y, x + z),
                    DMIN(y + z, fabs(p))) < TINY
      || DMAX(DMAX(x, y), DMAX(z, fabs(p))) > BIG)
    Die("invalid arguments in rj");
    sum = 0.0;
    fac = 1.0;
    if (p > 0.0) {
    xt = x;
    yt = y;
    zt = z;
    pt = p;
    } else {
    xt = DMIN(DMIN(x, y), z);
    zt = DMAX(DMAX(x, y), z);
    yt = x + y + z - xt - zt;
    a = 1.0/(yt - p);
    b = a*(zt - yt)*(yt - xt);
    pt = yt + b;
    rho = xt*zt/yt;
    tau = p*pt/yt;
    rcx = rc(rho, tau);
    }
    do {
    sqrtx = sqrt(xt);
    sqrty = sqrt(yt);
    sqrtz = sqrt(zt);
    alamb = sqrtx*(sqrty + sqrtz) + sqrty*sqrtz;
    alpha = (pt*(sqrtx + sqrty + sqrtz) + sqrtx*sqrty*sqrtz)*(pt*(sqrtx + sqrty + sqrtz) + sqrtx*sqrty*sqrtz);
    beta = pt*(pt + alamb)*(pt + alamb);
    sum +=  fac*rc(alpha, beta);
    fac = 0.25*fac;
    xt = 0.25*(xt + alamb);
    yt = 0.25*(yt + alamb);
    zt = 0.25*(zt + alamb);
    pt = 0.25*(pt + alamb);
    ave = 0.2*(xt + yt + zt + pt + pt);
    delx = (ave - xt)/ave;
    dely = (ave - yt)/ave;
    delz = (ave - zt)/ave;
    delp = (ave - pt)/ave;
    } while (DMAX(DMAX(fabs(delx), fabs(dely)), 
            DMAX(fabs(delz), fabs(delp))) > ERRTOL);
    ea = delx*(dely + delz) + dely*delz;
    eb = delx*dely*delz;
    ec = delp*delp;
    ed = ea - 3.0*ec;
    ee = eb + 2.0*delp*(ea - ec);
    ans = 3.0*sum + fac*(1.0 + ed*(-C1 + C5*ed - C6*ee) +
               eb*(C7 + delp*(-C8 + delp*C4))
               + delp*ea*(C2 - delp*C3) - C2*delp*ec)/(ave*sqrt(ave));
    if (p <= 0.0) ans = a*(b*ans + 3.0*(rcx - rf(xt, yt, zt)));
    return ans;
}

double rd(const double x, const double y, const double z)
{
    // Define constants inside the function
    const double ERRTOL = 0.0015;
    const double TINY = 1.0e-25;
    const double BIG = 4.5e21;
    const double C1 = 3.0/14.0;
    const double C2 = 1.0/6.0;
    const double C3 = 9.0/22.0;
    const double C4 = 3.0/26.0;
    const double C5 = 0.25*C3;
    const double C6 = 1.5*C4;

    double alamb, ave, delx, dely, delz, ea, eb, ec, ed, ee, fac,
    sqrtx, sqrty, sqrtz, sum, xt, yt, zt;

    if (DMIN(x, y) < 0.0 || DMIN(x + y, z) < TINY || DMAX(DMAX(x, y), z) > BIG)
    Die("invalid arguments in rd");
    xt = x;
    yt = y;
    zt = z;
    sum = 0.0;
    fac = 1.0;
    do {
    sqrtx = sqrt(xt);
    sqrty = sqrt(yt);
    sqrtz = sqrt(zt);
    alamb = sqrtx*(sqrty + sqrtz) + sqrty*sqrtz;
    sum += fac/(sqrtz*(zt + alamb));
    fac = 0.25*fac;
    xt = 0.25*(xt + alamb);
    yt = 0.25*(yt + alamb);
    zt = 0.25*(zt + alamb);
    ave = 0.2*(xt + yt + 3.0*zt);
    delx = (ave - xt)/ave;
    dely = (ave - yt)/ave;
    delz = (ave - zt)/ave;
    } while (DMAX(DMAX(fabs(delx), fabs(dely)), fabs(delz)) > ERRTOL);
    ea = delx*dely;
    eb = delz*delz;
    ec = ea-eb;
    ed = ea-6.0*eb;
    ee = ed+ec+ec;
    return 3.0*sum + fac*(1.0 + ed*(-C1 + C5*ed - C6*delz*ee)
            + delz*(C2*ee + delz*(-C3*ec + delz*C4*ea)))/(ave*sqrt(ave));
}

double ellpi(const double phi, const double en, const double ak)
{
    double rf(const double x, const double y, const double z);
    double rj(const double x, const double y, const double z, const double p);

    const double cc = cos(phi)*cos(phi);
    const double s = sin(phi);
    const double enss = en * s * s;
    const double q = (1.0 - s * ak) * (1.0 + s * ak);

    return s * (rf(cc, q, 1.0) - enss * rj(cc, q, 1.0, 1.0 + enss) / 3.0);
}

double elle(const double phi, const double ak)
{
    double rd(const double x, const double y, const double z);
    double rf(const double x, const double y, const double z);
  
    const double s = sin(phi);
    const double cc = pow(cos(phi), 2);
    const double q = (1.0 - s * ak) * (1.0 + s * ak);

    return s * (rf(cc, q, 1.0) - pow(s * ak, 2) * rd(cc, q, 1.0) / 3.0);
}

double ellf(const double phi, const double ak)
{
    double rf(const double x, const double y, const double z);
    const double s = sin(phi);

    if (phi > M_PI/2.)
        return (-1.*s*rf(pow(cos(phi), 2), (1.0-s*ak)*(1.0+s*ak), 1.0) + 2.*rf(0.,(1.0-ak)*(1.0+ak), 1.0));
    else
        return s*rf(pow(cos(phi), 2), (1.0-s*ak)*(1.0+s*ak), 1.0);
}
// ----------------------------------------------------------------------------


int sanity_check(double a, double p, double e, double Y)
{
    int res = 0;

    if (p < 0.0)
        return 1;
    if ((e > 1.0) || (e < 0.0))
        return 1;
    if ((Y > 1.0) || (Y < -1.0))
        return 1;
    if ((a > 1.0) || (a < 0.0))
        return 1;

    if (res == 1)
    {
        printf("a, p, e, Y = %f %f %f %f ", a, p, e, Y);
        // throw std::invalid_argument( "Sanity check wrong");
    }
    return res;
}

// Define elliptic integrals that use Mathematica's conventions
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

double EllipticF(double phi, double k)
{
    gsl_sf_result result;
    // cout << "CHECK2" << endl;
    int status = gsl_sf_ellint_F_e(phi, sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticF failed with arguments phi:%e k: %e", phi, k);
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

double EllipticEIncomp(double phi, double k)
{
    gsl_sf_result result;
    // cout << "CHECK4" << endl;
    int status = gsl_sf_ellint_E_e(phi, sqrt(k), GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticEIncomp failed with argument k: %e", k);
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

double EllipticPiIncomp(double n, double phi, double k)
{
    // cout << "CHECK7" << endl;
    gsl_sf_result result;
    int status = gsl_sf_ellint_P_e(phi, sqrt(k), -n, GSL_PREC_DOUBLE, &result);
    if (status != GSL_SUCCESS)
    {
        char str[1000];
        sprintf(str, "EllipticPiIncomp failed with argument k: %e", k);
        throw std::invalid_argument(str);
    }
    return result.val;
}

double CapitalDelta(double r, double a)
{
    return pow(r, 2.) - 2. * r + pow(a, 2.);
}

double f(double r, double a, double zm)
{
    return pow(r, 4) + pow(a, 2) * (r * (r + 2) + pow(zm, 2) * CapitalDelta(r, a));
}

double g(double r, double a, double zm)
{
    return 2 * a * r;
}

double h(double r, double a, double zm)
{
    return r * (r - 2) + pow(zm, 2) / (1 - pow(zm, 2)) * CapitalDelta(r, a);
}

double d(double r, double a, double zm)
{
    return (pow(r, 2) + pow(a, 2) * pow(zm, 2)) * CapitalDelta(r, a);
}

double fdot(double r, double a, double zm)
{
    double zm2 = pow(zm, 2.);
    return 4. * pow(r, 3.) + pow(a, 2.) * (2. * r * (1. + zm2) + 2. * (1 - zm2));
}

double gdot(double r, double a, double zm)
{
    return 2 * a;
}

double hdot(double r, double a, double zm)
{   
    double zm2 = pow(zm, 2.);
    return 2. * (r - 1.)*(1. + zm2 / (1. - zm2));
}

double ddot(double r, double a, double zm)
{   
    double a2 = pow(a, 2.);
    double zm2 = pow(zm, 2.);
    return 4. * pow(r, 3.) - 6. * pow(r, 2.) + 2.*a2*r*(1. + zm2) - 2.*a2*zm2;
}


double KerrGeoEnergy(double a, double p, double e, double x)
{

    double zm = sqrt(1. - pow(x, 2.));
    double Kappa, Epsilon, Rho, Eta, Sigma;
    if (e < 1e-10) {  // switch to spherical formulas A13-A17 (2102.02713) to avoid instability
        double r = p;

        Kappa = d(r, a, zm) * hdot(r, a, zm) - h(r, a, zm) * ddot(r, a, zm);
        Epsilon = d(r, a, zm) * gdot(r, a, zm) - g(r, a, zm) * ddot(r, a, zm);
        Rho = f(r, a, zm) * hdot(r, a, zm) - h(r, a, zm) * fdot(r, a, zm);
        Eta = f(r, a, zm) * gdot(r, a, zm) - g(r, a, zm) * fdot(r, a, zm);
        Sigma = g(r, a, zm) * hdot(r, a, zm) - h(r, a, zm) * gdot(r, a, zm);
    }
    else {
        double r1 = p / (1. - e);
        double r2 = p / (1. + e);

        Kappa = d(r1, a, zm) * h(r2, a, zm) - h(r1, a, zm) * d(r2, a, zm);
        Epsilon = d(r1, a, zm) * g(r2, a, zm) - g(r1, a, zm) * d(r2, a, zm);
        Rho = f(r1, a, zm) * h(r2, a, zm) - h(r1, a, zm) * f(r2, a, zm);
        Eta = f(r1, a, zm) * g(r2, a, zm) - g(r1, a, zm) * f(r2, a, zm);
        Sigma = g(r1, a, zm) * h(r2, a, zm) - h(r1, a, zm) * g(r2, a, zm);
    }

    return sqrt((Kappa * Rho + 2 * Epsilon * Sigma - x * 2 * sqrt(Sigma * (Sigma * pow(Epsilon, 2) + Rho * Epsilon * Kappa - Eta * pow(Kappa, 2)) / pow(x, 2))) / (pow(Rho, 2) + 4 * Eta * Sigma));
}

double KerrGeoAngularMomentum(double a, double p, double e, double x, double En)
{
    double r1 = p / (1 - e);

    double zm = sqrt(1 - pow(x, 2));

    return (-En * g(r1, a, zm) + x * sqrt((-d(r1, a, zm) * h(r1, a, zm) + pow(En, 2) * (pow(g(r1, a, zm), 2) + f(r1, a, zm) * h(r1, a, zm))) / pow(x, 2))) / h(r1, a, zm);
}

double KerrGeoCarterConstant(double a, double p, double e, double x, double En, double L)
{
    double zm = sqrt(1 - pow(x, 2));

    return pow(zm, 2) * (pow(a, 2) * (1 - pow(En, 2)) + pow(L, 2) / (1 - pow(zm, 2)));
}

void KerrGeoConstantsOfMotion(double *E_out, double *L_out, double *Q_out, double a, double p, double e, double x)
{
    *E_out = KerrGeoEnergy(a, p, e, x);
    *L_out = KerrGeoAngularMomentum(a, p, e, x, *E_out);
    *Q_out = KerrGeoCarterConstant(a, p, e, x, *E_out, *L_out);
}

void KerrGeoConstantsOfMotionVectorized(double *E_out, double *L_out, double *Q_out, double *a, double *p, double *e, double *x, int n)
{
#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i += 1)
    {
        KerrGeoConstantsOfMotion(&E_out[i], &L_out[i], &Q_out[i], a[i], p[i], e[i], x[i]);
    }
}

void KerrGeoRadialRoots(double *r1_, double *r2_, double *r3_, double *r4_, double a, double p, double e, double x, double En, double Q)
{
    double M = 1.0;
    double r1 = p / (1 - e);
    double r2 = p / (1 + e);
    double AplusB = (2 * M) / (1 - pow(En, 2)) - (r1 + r2);
    double AB = (pow(a, 2) * Q) / ((1 - pow(En, 2)) * r1 * r2);
    double r3 = (AplusB + sqrt(pow(AplusB, 2) - 4 * AB)) / 2;
    double r4 = AB / r3;

    *r1_ = r1;
    *r2_ = r2;
    *r3_ = r3;
    *r4_ = r4;
}

void KerrGeoMinoFrequencies(double *CapitalGamma_, double *CapitalUpsilonPhi_, double *CapitalUpsilonTheta_, double *CapitalUpsilonr_,
                            double a, double p, double e, double x)
{
    double M = 1.0;

    double En = KerrGeoEnergy(a, p, e, x);
    double L = KerrGeoAngularMomentum(a, p, e, x, En);
    double Q = KerrGeoCarterConstant(a, p, e, x, En, L);

    // get radial roots
    double r1, r2, r3, r4;
    KerrGeoRadialRoots(&r1, &r2, &r3, &r4, a, p, e, x, En, Q);

    double Epsilon0 = pow(a, 2) * (1 - pow(En, 2)) / pow(L, 2);
    double zm = 1 - pow(x, 2);
    double a2zp = (pow(L, 2) + pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm)) / ((-1 + pow(En, 2)) * (-1 + zm));

    double Epsilon0zp = -((pow(L, 2) + pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm)) / (pow(L, 2) * (-1 + zm)));

    double zmOverZp = zm / ((pow(L, 2) + pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm)) / (pow(a, 2) * (-1 + pow(En, 2)) * (-1 + zm)));

    double kr = sqrt((r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4));                                                //(*Eq.(13)*)
    double kTheta = sqrt(zmOverZp);                                                                                 //(*Eq.(13)*)
    double CapitalUpsilonr = (M_PI * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) / (2 * EllipticK(pow(kr, 2))); //(*Eq.(15)*)
    double CapitalUpsilonTheta = (M_PI * L * sqrt(Epsilon0zp)) / (2 * EllipticK(pow(kTheta, 2)));                   //(*Eq.(15)*)

    double rp = M + sqrt(1.0 - pow(a, 2));
    double rm = M - sqrt(1.0 - pow(a, 2));

    double hr = (r1 - r2) / (r1 - r3);
    double hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp));
    double hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm));

    // (*Eq. (21)*)
    double CapitalUpsilonPhi = (2 * CapitalUpsilonTheta) / (M_PI * sqrt(Epsilon0zp)) * EllipticPi(zm, pow(kTheta, 2)) + (2 * a * CapitalUpsilonr) / (M_PI * (rp - rm) * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) * ((2 * M * En * rp - a * L) / (r3 - rp) * (EllipticK(pow(kr, 2)) - (r2 - r3) / (r2 - rp) * EllipticPi(hp, pow(kr, 2))) - (2 * M * En * rm - a * L) / (r3 - rm) * (EllipticK(pow(kr, 2)) - (r2 - r3) / (r2 - rm) * EllipticPi(hm, pow(kr, 2))));

    double CapitalGamma = 4 * 1.0 * En + (2 * a2zp * En * CapitalUpsilonTheta) / (M_PI * L * sqrt(Epsilon0zp)) * (EllipticK(pow(kTheta, 2)) - EllipticE(pow(kTheta, 2))) + (2 * CapitalUpsilonr) / (M_PI * sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4))) * (En / 2 * ((r3 * (r1 + r2 + r3) - r1 * r2) * EllipticK(pow(kr, 2)) + (r2 - r3) * (r1 + r2 + r3 + r4) * EllipticPi(hr, pow(kr, 2)) + (r1 - r3) * (r2 - r4) * EllipticE(pow(kr, 2))) + 2 * M * En * (r3 * EllipticK(pow(kr, 2)) + (r2 - r3) * EllipticPi(hr, pow(kr, 2))) + (2 * M) / (rp - rm) * (((4 * 1.0 * En - a * L) * rp - 2 * M * pow(a, 2) * En) / (r3 - rp) * (EllipticK(pow(kr, 2)) - (r2 - r3) / (r2 - rp) * EllipticPi(hp, pow(kr, 2))) - ((4 * 1.0 * En - a * L) * rm - 2 * M * pow(a, 2) * En) / (r3 - rm) * (EllipticK(pow(kr, 2)) - (r2 - r3) / (r2 - rm) * EllipticPi(hm, pow(kr, 2)))));

    *CapitalGamma_ = CapitalGamma;
    *CapitalUpsilonPhi_ = CapitalUpsilonPhi;
    *CapitalUpsilonTheta_ = CapitalUpsilonTheta;
    *CapitalUpsilonr_ = CapitalUpsilonr;
}

void KerrCircularMinoFrequencies(double *CapitalGamma_, double *CapitalUpsilonPhi_, double *CapitalUpsilonTheta_, double *CapitalUpsilonr_,
                                 double a, double p, double e, double x)
{
    double CapitalUpsilonr = sqrt((p * (-2 * pow(a, 2) + 6 * a * sqrt(p) + (-5 + p) * p + (pow(a - sqrt(p), 2) * (pow(a, 2) - 4 * a * sqrt(p) - (-4 + p) * p)) / abs(pow(a, 2) - 4 * a * sqrt(p) - (-4 + p) * p))) / (2 * a * sqrt(p) + (-3 + p) * p));
    double CapitalUpsilonTheta = abs((pow(p, 0.25) * sqrt(3 * pow(a, 2) - 4 * a * sqrt(p) + pow(p, 2))) / sqrt(2 * a + (-3 + p) * sqrt(p)));
    double CapitalUpsilonPhi = pow(p, 1.25) / sqrt(2 * a + (-3 + p) * sqrt(p));
    double CapitalGamma = (pow(p, 1.25) * (a + pow(p, 1.5))) / sqrt(2 * a + (-3 + p) * sqrt(p));

    *CapitalGamma_ = CapitalGamma;
    *CapitalUpsilonPhi_ = CapitalUpsilonPhi;
    *CapitalUpsilonTheta_ = CapitalUpsilonTheta;
    *CapitalUpsilonr_ = CapitalUpsilonr;
}

void KerrGeoCoordinateFrequencies(double *OmegaPhi_, double *OmegaTheta_, double *OmegaR_,
                                  double a, double p, double e, double x)
{
    // printf("here p e %f %f %f %f \n", a, p, e, x);
    double CapitalGamma, CapitalUpsilonPhi, CapitalUpsilonTheta, CapitalUpsilonR;

    KerrGeoMinoFrequencies(&CapitalGamma, &CapitalUpsilonPhi, &CapitalUpsilonTheta, &CapitalUpsilonR,
                           a, p, e, x);

    if ((CapitalUpsilonPhi != CapitalUpsilonPhi) || (CapitalGamma != CapitalGamma) || (CapitalUpsilonR != CapitalUpsilonR))
    {
        printf("(a, p, e, x) = (%f , %f , %f , %f) \n", a, p, e, x);
        throw std::invalid_argument("Nan in fundamental frequencies");
    }
    // printf("here xhi %f %f\n", CapitalUpsilonPhi, CapitalGamma);
    *OmegaPhi_ = CapitalUpsilonPhi / CapitalGamma;
    *OmegaTheta_ = CapitalUpsilonTheta / CapitalGamma;
    *OmegaR_ = CapitalUpsilonR / CapitalGamma;
}

void KerrGeoEquatorialMinoFrequencies(double *CapitalGamma_, double *CapitalUpsilonPhi_, double *CapitalUpsilonTheta_, double *CapitalUpsilonr_,
                                      double a, double p, double e, double x)
{
    double M = 1.0;

    double En = KerrGeoEnergy(a, p, e, x);
    double L = KerrGeoAngularMomentum(a, p, e, x, En);
    double Q = KerrGeoCarterConstant(a, p, e, x, En, L);

    // get radial roots
    double r1, r2, r3, r4;
    KerrGeoRadialRoots(&r1, &r2, &r3, &r4, a, p, e, x, En, Q);
    double a_squared = a*a;
    double En_squared = En*En;
    double L_squared = L*L;
    double Epsilon0 = a_squared * (1 - En_squared) / L_squared;
    // double zm = 0;
    double a2zp = (L_squared + a_squared * (-1 + En_squared) * (-1)) / ((-1 + En_squared) * (-1));

    double Epsilon0zp = -((L_squared + a_squared * (-1 + En_squared) * (-1)) / (L_squared * (-1)));

    double zp = a_squared * (1 - En_squared) + L_squared;

    double arg_kr = (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4);

    // double kr = sqrt(arg_kr); //(*Eq.(13)*)
    // double kTheta = 0; //(*Eq.(13)*)
    double kr2 = abs(arg_kr);

    if (kr2>1.0){
        printf("kr %e %e \n", arg_kr, (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4));
        printf("r1 r2 r3 r4 %e %e %e %e\n", r1, r2, r3, r4);
        printf("a p e %e %e %e\n", a,p,e);
    }
    double elK = EllipticK(kr2);
    double CapitalUpsilonr = (M_PI * sqrt((1 - En_squared) * (r1 - r3) * (r2))) / (2 * elK); //(*Eq.(15)*)
    double CapitalUpsilonTheta = x * pow(zp, 0.5);                                                             //(*Eq.(15)*)

    double rp = M + sqrt(1.0 - a_squared);
    double rm = M - sqrt(1.0 - a_squared);

    // this check was introduced to avoid round off errors
    // if (r3 - rp==0.0){
    // printf("round off error %e %e %e\n", r3 - rp, L, 2*rp*En/a);
    // }

    double hr = (r1 - r2) / (r1 - r3);
    double hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp));
    double hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm));

    // (*Eq. (21)*)
    // This term is zero when r3 - rp == 0.0
    double elPi = EllipticPi(hp, kr2);
    double elPi_hm = EllipticPi(hm, kr2);
    double elPi_hr = EllipticPi(hr, kr2);
    double prob1 = (2 * M * En * rp - a * L) * (elK - (r2 - r3) / (r2 - rp) * elPi);
    if (abs(prob1) != 0.0)
    {
        prob1 = prob1 / (r3 - rp);
    }
    double CapitalUpsilonPhi = (CapitalUpsilonTheta) / (sqrt(Epsilon0zp)) + (2 * a * CapitalUpsilonr) / (M_PI * (rp - rm) * sqrt((1 - En_squared) * (r1 - r3) * (r2 - r4))) * (prob1 - (2 * M * En * rm - a * L) / (r3 - rm) * (elK - (r2 - r3) / (r2 - rm) * elPi_hm));

    // This term is zero when r3 - rp == 0.0
    double prob2 = ((4 * 1.0 * En - a * L) * rp - 2 * M * a_squared * En) * (elK - (r2 - r3) / (r2 - rp) * elPi);
    if (abs(prob2) != 0.0)
    {
        prob2 = prob2 / (r3 - rp);
    }
    double CapitalGamma = 4 * 1.0 * En + (2 * CapitalUpsilonr) / (M_PI * sqrt((1 - En_squared) * (r1 - r3) * (r2 - r4))) * (En / 2 * ((r3 * (r1 + r2 + r3) - r1 * r2) * elK + (r2 - r3) * (r1 + r2 + r3 + r4) * elPi_hr + (r1 - r3) * (r2 - r4) * EllipticE(kr2)) + 2 * M * En * (r3 * elK + (r2 - r3) * elPi_hr) + (2 * M) / (rp - rm) * (prob2 - ((4 * 1.0 * En - a * L) * rm - 2 * M * a_squared * En) / (r3 - rm) * (elK - (r2 - r3) / (r2 - rm) * elPi_hm)));

    // This check makes sure that the problematic terms are zero when r3-rp is zero
    // if (r3 - rp==0.0){
    // printf("prob %e %e\n", prob1, prob2);
    // diff_r3_rp = 1e10;
    // }

    *CapitalGamma_ = CapitalGamma;
    *CapitalUpsilonPhi_ = CapitalUpsilonPhi;
    *CapitalUpsilonTheta_ = abs(CapitalUpsilonTheta);
    *CapitalUpsilonr_ = CapitalUpsilonr;
}

void KerrGeoEquatorialCoordinateFrequencies(double *OmegaPhi_, double *OmegaTheta_, double *OmegaR_,
                                            double a, double p, double e, double x)
{
    double CapitalGamma, CapitalUpsilonPhi, CapitalUpsilonTheta, CapitalUpsilonR;

    // printf("(a, p, e, x) = (%f , %f , %f , %f) \n", a, p, e, x);
    // if (e=0.0){
    //     KerrCircularMinoFrequencies(&CapitalGamma, &CapitalUpsilonPhi, &CapitalUpsilonTheta, &CapitalUpsilonR,
    //                               a, p, e, x);
    // }
    // else{
    KerrGeoEquatorialMinoFrequencies(&CapitalGamma, &CapitalUpsilonPhi, &CapitalUpsilonTheta, &CapitalUpsilonR,
                                     a, p, e, x);
    // }

    *OmegaPhi_ = CapitalUpsilonPhi / CapitalGamma;
    *OmegaTheta_ = CapitalUpsilonTheta / CapitalGamma;
    *OmegaR_ = CapitalUpsilonR / CapitalGamma;
}

void SchwarzschildGeoCoordinateFrequencies(double *OmegaPhi, double *OmegaR, double p, double e)
{
    // Need to evaluate 4 different elliptic integrals here. Cache them first to avoid repeated calls.
    // cout << "TEMPTEMP " << p << " " << e << endl;
    double EllipE = EllipticE(4 * e / (p - 6.0 + 2 * e));
    double EllipK = EllipticK(4 * e / (p - 6.0 + 2 * e));
    ;
    double EllipPi1 = EllipticPi(16 * e / (12.0 + 8 * e - 4 * e * e - 8 * p + p * p), 4 * e / (p - 6.0 + 2 * e));
    double EllipPi2 = EllipticPi(2 * e * (p - 4) / ((1.0 + e) * (p - 6.0 + 2 * e)), 4 * e / (p - 6.0 + 2 * e));

    *OmegaPhi = (2 * Power(p, 1.5)) / (Sqrt(-4 * Power(e, 2) + Power(-2 + p, 2)) * (8 + ((-2 * EllipPi2 * (6 + 2 * e - p) * (3 + Power(e, 2) - p) * Power(p, 2)) / ((-1 + e) * Power(1 + e, 2)) - (EllipE * (-4 + p) * Power(p, 2) * (-6 + 2 * e + p)) / (-1 + Power(e, 2)) +
                                                                                         (EllipK * Power(p, 2) * (28 + 4 * Power(e, 2) - 12 * p + Power(p, 2))) / (-1 + Power(e, 2)) + (4 * (-4 + p) * p * (2 * (1 + e) * EllipK + EllipPi2 * (-6 - 2 * e + p))) / (1 + e) + 2 * Power(-4 + p, 2) * (EllipK * (-4 + p) + (EllipPi1 * p * (-6 - 2 * e + p)) / (2 + 2 * e - p))) /
                                                                                            (EllipK * Power(-4 + p, 2))));

    *OmegaR = (p * Sqrt((-6 + 2 * e + p) / (-4 * Power(e, 2) + Power(-2 + p, 2))) * Pi) /
              (8 * EllipK + ((-2 * EllipPi2 * (6 + 2 * e - p) * (3 + Power(e, 2) - p) * Power(p, 2)) / ((-1 + e) * Power(1 + e, 2)) - (EllipE * (-4 + p) * Power(p, 2) * (-6 + 2 * e + p)) / (-1 + Power(e, 2)) +
                             (EllipK * Power(p, 2) * (28 + 4 * Power(e, 2) - 12 * p + Power(p, 2))) / (-1 + Power(e, 2)) + (4 * (-4 + p) * p * (2 * (1 + e) * EllipK + EllipPi2 * (-6 - 2 * e + p))) / (1 + e) + 2 * Power(-4 + p, 2) * (EllipK * (-4 + p) + (EllipPi1 * p * (-6 - 2 * e + p)) / (2 + 2 * e - p))) /
                                Power(-4 + p, 2));
}

void KerrGeoCoordinateFrequenciesVectorized(double *OmegaPhi_, double *OmegaTheta_, double *OmegaR_,
                                            double *a, double *p, double *e, double *x, int length)
{

#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < length; i += 1)
    {
        if (a[i] != 0.0)
        {
            if (abs(x[i]) != 1.)
            {
                KerrGeoCoordinateFrequencies(&OmegaPhi_[i], &OmegaTheta_[i], &OmegaR_[i],
                                             a[i], p[i], e[i], x[i]);
            }
            else
            {
                KerrGeoEquatorialCoordinateFrequencies(&OmegaPhi_[i], &OmegaTheta_[i], &OmegaR_[i],
                                                       a[i], p[i], e[i], x[i]);
            }
        }
        else
        {
            SchwarzschildGeoCoordinateFrequencies(&OmegaPhi_[i], &OmegaR_[i], p[i], e[i]);
            OmegaTheta_[i] = OmegaPhi_[i];
        }
    }
}

double periodic_acos(double x) {
    // Ensure x is within the range [-1, 1]
    x = fmod(x, 2.0);
    if (x < -1.0)
        x += 2.0;
    else if (x > 1.0)
        x -= 2.0;

    return acos(x);
}

void solveCubic(double A2, double A1, double A0,double *rp, double *ra, double *r3) {
    // Coefficients
    double a = 1.; // coefficient of r^3
    double b = A2; // coefficient of r^2
    double c = A1; // coefficient of r^1
    double d = A0; // coefficient of r^0
    
    // Calculate p and q
    double p = (3.*a*c - b*b) / (3.*a*a);
    double q = (2.*b*b*b - 9.*a*b*c + 27.*a*a*d) / (27.*a*a*a);

    // Calculate discriminant
    double discriminant = q*q/4. + p*p*p/27.;

    if (discriminant >= 0) {
        // One real root and two complex conjugate roots
        double u = cbrt(-q/2. + sqrt(discriminant));
        double v = cbrt(-q/2. - sqrt(discriminant));
        double root = u + v - b/(3.*a);
        // cout << "Real Root: " << root << endl;

        complex<double> imaginaryPart(-sqrt(3.0) / 2.0 * (u - v), 0.5 * (u + v));
        complex<double> root2 = -0.5 * (u + v) - b / (3. * a) + imaginaryPart;
        complex<double> root3 = -0.5 * (u + v) - b / (3. * a) - imaginaryPart;
        // cout << "Complex Root 1: " << root2 << endl;
        // cout << "Complex Root 2: " << root3 << endl;
        *ra = -0.5 * (u + v) - b / (3. * a);
        *rp = -0.5 * (u + v) - b / (3. * a);
        *r3 = root;
    // } else if (discriminant == 0) {
    //     // All roots are real and at least two are equal
    //     double u = cbrt(-q/2.);
    //     double v = cbrt(-q/2.);
    //     double root = u + v - b/(3.*a);
    //     // cout << "Real Root: " << root << endl;
    //     // cout << "Real Root (equal to above): " << root << endl;
    //     // complex<double> root2 = -0.5 * (u + v) - b / (3 * a);
    //     // cout << "Complex Root: " << root2 << endl;
    //     *ra = -0.5 * (u + v) - b / (3. * a);
    //     *rp = -0.5 * (u + v) - b / (3. * a);
    //     *r3 = root;
    } else {
        // All three roots are real and different
        double r = sqrt(-p/3.);
        double theta = acos(-q/(2.*r*r*r));
        double root1 = 2. * r * cos(theta/3.) - b / (3. * a);
        double root2 = 2. * r * cos((theta + 2.*M_PI) / 3.) - b / (3. * a);
        double root3 = 2. * r * cos((theta - 2.*M_PI) / 3.) - b / (3. * a);
        // ra = -2.*rtQnr*cos((theta + 2.*M_PI)/3.) - A2/3.;
        // rp = -2.*rtQnr*cos((theta - 2.*M_PI)/3.) - A2/3.;
        *ra = root1;
        *rp = root3;
        *r3 = root2;
    }
    // cout << "ra: " << *ra << endl;
    // cout << "rp: " << *rp << endl;
    // cout << "r3: " << *r3 << endl;
}

void ELQ_to_pex(double *p, double *e, double *xI, double a, double E, double Lz, double Q)
//
// pexI_of_aELzQ.cc: implements the mapping from orbit integrals
// (E, Lz, Q) to orbit geometry (p, e, xI).  Also provides the
// roots r3 and r4 of the Kerr radial geodesic function.
//
// Scott A. Hughes (sahughes@mit.edu); code extracted from Gremlin
// and converted to standalone form 13 Jan 2024.
//
{
  if (Q < 1.e-14) { // equatorial
    
    double E2m1 = E*E - 1.;//(E - 1.)*(E + 1.);
    double A2 = 2./E2m1;
    double A1 = a*a - Lz*Lz/E2m1;//(a*a*E2m1 - Lz*Lz)/E2m1;
    double A0 = 2.*(a*E - Lz)*(a*E - Lz)/E2m1;
    double rp,ra,r3;
    solveCubic(A2,A1,A0,&rp,&ra,&r3);
    //
    // double Qnr = (A2*A2 - 3.*A1)/9.;
    // double rtQnr = sqrt(Qnr);
    // double Rnr = (A2*(2.*A2*A2 - 9.*A1) + 27.*A0)/54.;
    // double argacos = Rnr/(rtQnr*rtQnr*rtQnr);
    // double theta = acos(argacos);
    // ra = -2.*rtQnr*cos((theta + 2.*M_PI)/3.) - A2/3.;
    // rp = -2.*rtQnr*cos((theta - 2.*M_PI)/3.) - A2/3.;
    // cout << "Scott ra: " << ra << endl;
    // cout << "Scott rp: " << rp << endl;

    *p = 2.*ra*rp/(ra + rp);
    *e = (ra - rp)/(ra + rp);
    // cout << " p: " << *p << endl;
    // cout << " e: " << *e << endl;
    
    // r3 = -2.*rtQnr*cos(theta/3.) - A2/3.;
    // r4 = 0.;
    //
    
    // if (isnan(*p)||isnan(*e)){
    //     cout << "beginning" << " E =" << E  << "\t" << "L=" <<  Lz << "\t" << "Q=" << Q << endl;
    //     cout << "beginning" << " a =" << a  << "\t" << "p=" <<  *p << "\t" << "e=" << *e << "\t" <<  "arg of acos=" <<Rnr/(rtQnr*rtQnr*rtQnr) << endl;
    //     throw std::exception();
    // }

    if (Lz > 0.) *xI = 1.;
    else *xI = -1.;
  } else { // non-equatorial
    double a2 = a*a;
    double E2m1= (E - 1)*(E + 1.);
    double aEmLz = a*E - Lz;
    //
    // The quartic: r^4 + A3 r^3 + A2 r^2 + A1 r + A0 == 0.
    // Kerr radial function divided by E^2 - 1.
    //
    double A0 = -a2*Q/E2m1;
    double A1 = 2.*(Q + aEmLz*aEmLz)/E2m1;
    double A2 = (a2*E2m1 - Lz*Lz - Q)/E2m1;
    double A3 = 2./E2m1;
    //
    // Definitions following Wolters (https://quarticequations.com)
    //
    double B0 = A0 + A3*(-0.25*A1 + A3*(0.0625*A2 - 0.01171875*A3*A3));
    double B1 = A1 + A3*(-0.5*A2 + 0.125*A3*A3);
    double B2 = A2 - 0.375*A3*A3;
    //
    // Definitions needed for the resolvent cubic: z^3 + C2 z^2 + C1 z + C0 == 0;
    //
    double C0 = -0.015625*B1*B1;
    double C1 = 0.0625*B2*B2 - 0.25*B0;
    double C2 = 0.5*B2;
    //
    double rtQnr = sqrt(C2*C2/9. - C1/3.);
    double Rnr = C2*(C2*C2/27. - C1/6.) + C0/2.;
    double theta = acos(Rnr/(rtQnr*rtQnr*rtQnr));
    //
    // zN = cubic zero N
    //
    double rtz1 = sqrt(-2.*rtQnr*cos((theta + 2.*M_PI)/3.) - C2/3.);
    double z2 = -2.*rtQnr*cos((theta - 2.*M_PI)/3.) - C2/3.;
    double z3 = -2.*rtQnr*cos(theta/3.) - C2/3.;
    double rtz2z3 = sqrt(z2*z3);
    //
    // Now assemble the roots of the quartic.  Note that M/(2(1 - E^2)) = -0.25*A3.
    //
    double sgnB1 = (B1 > 0 ? 1. : -1.);
    double rttermmin = sqrt(z2 + z3 - 2.*sgnB1*rtz2z3);
    double rttermplus = sqrt(z2 + z3 + 2.*sgnB1*rtz2z3);
    double ra = -0.25*A3 + rtz1 + rttermmin;
    double rp = -0.25*A3 + rtz1 - rttermmin;
    // r3 = -0.25*A3 - rtz1 + rttermplus;
    // r4 = -0.25*A3 - rtz1 - rttermplus;
    //
    *p = 2.*ra*rp/(ra + rp);
    *e = (ra - rp)/(ra + rp);
    //
    // Note that omE2 = 1 - E^2 = -E2m1 = -(E^2 - 1)
    //
    double QpLz2ma2omE2 = Q + Lz*Lz + a2*E2m1;
    double denomsqr = QpLz2ma2omE2 + sqrt(QpLz2ma2omE2*QpLz2ma2omE2 - 4.*Lz*Lz*a2*E2m1);
    *xI = sqrt(2.)*Lz/sqrt(denomsqr);
  }
    
}

void ELQ_to_pexVectorised(double *p, double *e, double *x, double *a, double *E, double *Lz, double *Q, int length)
{
#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < length; i += 1)
    {
        ELQ_to_pex(&p[i], &e[i], &x[i], a[i], E[i], Lz[i], Q[i]);
    }
}


struct params_holder
{
    double a, p, e, x, Y;
};

double separatrix_polynomial_full(double p, void *params_in)
{

    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;

    return (-4 * (3 + e) * Power(p, 11) + Power(p, 12) + Power(a, 12) * Power(-1 + e, 4) * Power(1 + e, 8) * Power(-1 + x, 4) * Power(1 + x, 4) - 4 * Power(a, 10) * (-3 + e) * Power(-1 + e, 3) * Power(1 + e, 7) * p * Power(-1 + Power(x, 2), 4) - 4 * Power(a, 8) * (-1 + e) * Power(1 + e, 5) * Power(p, 3) * Power(-1 + x, 3) * Power(1 + x, 3) * (7 - 7 * Power(x, 2) - Power(e, 2) * (-13 + Power(x, 2)) + Power(e, 3) * (-5 + Power(x, 2)) + 7 * e * (-1 + Power(x, 2))) + 8 * Power(a, 6) * (-1 + e) * Power(1 + e, 3) * Power(p, 5) * Power(-1 + Power(x, 2), 2) * (3 + e + 12 * Power(x, 2) + 4 * e * Power(x, 2) + Power(e, 3) * (-5 + 2 * Power(x, 2)) + Power(e, 2) * (1 + 2 * Power(x, 2))) - 8 * Power(a, 4) * Power(1 + e, 2) * Power(p, 7) * (-1 + x) * (1 + x) * (-3 + e + 15 * Power(x, 2) - 5 * e * Power(x, 2) + Power(e, 3) * (-5 + 3 * Power(x, 2)) + Power(e, 2) * (-1 + 3 * Power(x, 2))) + 4 * Power(a, 2) * Power(p, 9) * (-7 - 7 * e + Power(e, 3) * (-5 + 4 * Power(x, 2)) + Power(e, 2) * (-13 + 12 * Power(x, 2))) + 2 * Power(a, 8) * Power(-1 + e, 2) * Power(1 + e, 6) * Power(p, 2) * Power(-1 + Power(x, 2), 3) * (2 * Power(-3 + e, 2) * (-1 + Power(x, 2)) + Power(a, 2) * (Power(e, 2) * (-3 + Power(x, 2)) - 3 * (1 + Power(x, 2)) + 2 * e * (1 + Power(x, 2)))) - 2 * Power(p, 10) * (-2 * Power(3 + e, 2) + Power(a, 2) * (-3 + 6 * Power(x, 2) + Power(e, 2) * (-3 + 2 * Power(x, 2)) + e * (-2 + 4 * Power(x, 2)))) + Power(a, 6) * Power(1 + e, 4) * Power(p, 4) * Power(-1 + Power(x, 2), 2) * (-16 * Power(-1 + e, 2) * (-3 - 2 * e + Power(e, 2)) * (-1 + Power(x, 2)) + Power(a, 2) * (15 + 6 * Power(x, 2) + 9 * Power(x, 4) + Power(e, 2) * (26 + 20 * Power(x, 2) - 2 * Power(x, 4)) + Power(e, 4) * (15 - 10 * Power(x, 2) + Power(x, 4)) + 4 * Power(e, 3) * (-5 - 2 * Power(x, 2) + Power(x, 4)) - 4 * e * (5 + 2 * Power(x, 2) + 3 * Power(x, 4)))) - 4 * Power(a, 4) * Power(1 + e, 2) * Power(p, 6) * (-1 + x) * (1 + x) * (-2 * (11 - 14 * Power(e, 2) + 3 * Power(e, 4)) * (-1 + Power(x, 2)) + Power(a, 2) * (5 - 5 * Power(x, 2) - 9 * Power(x, 4) + 4 * Power(e, 3) * Power(x, 2) * (-2 + Power(x, 2)) + Power(e, 4) * (5 - 5 * Power(x, 2) + Power(x, 4)) + Power(e, 2) * (6 - 6 * Power(x, 2) + 4 * Power(x, 4)))) + Power(a, 2) * Power(p, 8) * (-16 * Power(1 + e, 2) * (-3 + 2 * e + Power(e, 2)) * (-1 + Power(x, 2)) + Power(a, 2) * (15 - 36 * Power(x, 2) + 30 * Power(x, 4) + Power(e, 4) * (15 - 20 * Power(x, 2) + 6 * Power(x, 4)) + 4 * Power(e, 3) * (5 - 12 * Power(x, 2) + 6 * Power(x, 4)) + 4 * e * (5 - 12 * Power(x, 2) + 10 * Power(x, 4)) + Power(e, 2) * (26 - 72 * Power(x, 2) + 44 * Power(x, 4)))));
}

double separatrix_polynomial_polar(double p, void *params_in)
{
    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;

    return (Power(a, 6) * Power(-1 + e, 2) * Power(1 + e, 4) + Power(p, 5) * (-6 - 2 * e + p) + Power(a, 2) * Power(p, 3) * (-4 * (-1 + e) * Power(1 + e, 2) + (3 + e * (2 + 3 * e)) * p) - Power(a, 4) * Power(1 + e, 2) * p * (6 + 2 * Power(e, 3) + 2 * e * (-1 + p) - 3 * p - 3 * Power(e, 2) * (2 + p)));
}

double separatrix_polynomial_equat(double p, void *params_in)
{
    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;

    return (Power(a, 4) * Power(-3 - 2 * e + Power(e, 2), 2) + Power(p, 2) * Power(-6 - 2 * e + p, 2) - 2 * Power(a, 2) * (1 + e) * p * (14 + 2 * Power(e, 2) + 3 * p - e * p));
}

double derivative_polynomial_equat(double p, void *params_in)
{
    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;
    return -2 * Power(a, 2) * (1 + e) * (14 + 2 * Power(e, 2) - e * p + 6 * p) + 4 * p * (18 + 2 * Power(e, 2) - 3 * e * (-4 + p) - 9 * p + Power(p, 2));
}

void eq_pol_fdf(double p, void *params_in, double *y, double *dy)
{
    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double e = params->e;
    double x = params->x;
    *y = (Power(a, 4) * Power(-3 - 2 * e + Power(e, 2), 2) + Power(p, 2) * Power(-6 - 2 * e + p, 2) - 2 * Power(a, 2) * (1 + e) * p * (14 + 2 * Power(e, 2) + 3 * p - e * p));
    *dy = -2 * Power(a, 2) * (1 + e) * (14 + 2 * Power(e, 2) - e * p + 6 * p) + 4 * p * (18 + 2 * Power(e, 2) - 3 * e * (-4 + p) - 9 * p + Power(p, 2));
}

double solver(struct params_holder *params, double (*func)(double, void *), double x_lo, double x_hi)
{
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 1000;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 0, r_expected = sqrt(5.0);
    gsl_function F;

    F.function = func;
    F.params = params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T);
    gsl_root_fsolver_set(s, &F, x_lo, x_hi);

    // printf("-----------START------------------- \n");
    // printf("xlo xhi %f %f\n", x_lo, x_hi);
    // double epsrel=0.001;
    double epsrel = 1e-11; // Decreased tolorance

    do
    {
        iter++;
        status = gsl_root_fsolver_iterate(s);
        r = gsl_root_fsolver_root(s);
        x_lo = gsl_root_fsolver_x_lower(s);
        x_hi = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(x_lo, x_hi, 0.0, epsrel);

        // printf("result %f %f %f \n", r, x_lo, x_hi);
    } while (status == GSL_CONTINUE && iter < max_iter);

    // printf("result %f %f %f \n", r, x_lo, x_hi);
    // printf("stat, iter, GSL_SUCCESS %d %d %d\n", status, iter, GSL_SUCCESS);
    // printf("-----------END------------------- \n");

    if (status != GSL_SUCCESS)
    {
        // warning if it did not converge otherwise throw error
        if (iter == max_iter)
        {
            printf("a, p, e, Y = %e %e %e %e\n", params->a, params->p, params->e, params->Y);
            throw std::invalid_argument("In Utility.cc Brent root solver failed");
            printf("WARNING: Maximum iteration reached in Utility.cc in Brent root solver.\n");
            printf("Result=%f, x_low=%f, x_high=%f \n", r, x_lo, x_hi);
            printf("a, p, e, Y, sep = %f %f %f %f %f\n", params->a, params->p, params->e, params->Y, get_separatrix(params->a, params->e, r));
            
        }
        else
        {
            throw std::invalid_argument("In Utility.cc Brent root solver failed");
        }
    }

    gsl_root_fsolver_free(s);
    return r;
}

double solver_derivative(struct params_holder *params, double x_lo, double x_hi)
{
    gsl_set_error_handler_off();
    int status;
    int iter = 0, max_iter = 100;
    const gsl_root_fdfsolver_type *T;
    gsl_root_fdfsolver *s;
    double x0, x = 0.8 * (x_lo + x_hi);
    gsl_function_fdf FDF;

    FDF.f = &separatrix_polynomial_equat;
    FDF.df = &derivative_polynomial_equat;
    FDF.fdf = &eq_pol_fdf;
    FDF.params = params;

    T = gsl_root_fdfsolver_steffenson;
    s = gsl_root_fdfsolver_alloc(T);
    gsl_root_fdfsolver_set(s, &FDF, x);

    do
    {
        iter++;
        status = gsl_root_fdfsolver_iterate(s);
        x0 = x;
        x = gsl_root_fdfsolver_root(s);
        status = gsl_root_test_delta(x, x0, 0, 1e-7);

    } while (status == GSL_CONTINUE && iter < max_iter);

    if (status != GSL_SUCCESS)
    {
        // warning if it did not converge otherwise throw error
        if (iter == max_iter)
        {
            printf("WARNING: Maximum iteration reached in Utility.cc in Brent root solver.\n");
            printf("Result=%f, x_low=%f, x_high=%f \n", x, x_lo, x_hi);
            printf("a, p, e, Y, sep = %f %f %f %f \n", params->a, params->p, params->e, params->Y);
        }
        else
        {
            throw std::invalid_argument("In Utility.cc Brent root solver failed");
        }
    }

    gsl_root_fdfsolver_free(s);

    return x;
}

double get_separatrix(double a, double e, double x)
{
    double p_sep, z1, z2;
    double sign;
    if (a == 0.0)
    {
        p_sep = 6.0 + 2.0 * e;
        return p_sep;
    }
    else if ((e < 0.0) & (abs(x) == 1.0))
    {
        z1 = 1. + pow((1. - pow(a, 2)), 1. / 3.) * (pow((1. + a), 1. / 3.) + pow((1. - a), 1. / 3.));

        z2 = sqrt(3. * pow(a, 2) + pow(z1, 2));

        // prograde
        if (x > 0.0)
        {
            sign = -1.0;
        }
        // retrograde
        else
        {
            sign = +1.0;
        }

        p_sep = (3. + z2 + sign * sqrt((3. - z1) * (3. + z1 + 2. * z2)));
        return p_sep;
    }
    else if (x == 1.0) // Eccentric Prograde Equatorial
    {
        // fills in p and Y with zeros
        struct params_holder params = {a, 0.0, e, x, x};
        double x_lo, x_hi;

        x_lo = 1.0 + e;
        x_hi = 6 + 2. * e;

        p_sep = solver(&params, &separatrix_polynomial_equat, x_lo, x_hi); // separatrix_KerrEquatorial(a, e);//
        return p_sep;
    }
    else if (x == -1.0) // Eccentric Retrograde Equatorial
    {
        // fills in p and Y with zeros
        struct params_holder params = {a, 0.0, e, x, x};
        double x_lo, x_hi;

        x_lo = 6 + 2. * e;
        x_hi = 5 + e + 4 * Sqrt(1 + e);

        p_sep = solver(&params, &separatrix_polynomial_equat, x_lo, x_hi);
        return p_sep;
    }
    else
    {
        // fills in p and Y with zeros
        struct params_holder params = {a, 0.0, e, x, 0.0};
        double x_lo, x_hi;

        // solve for polar p_sep
        x_lo = 1.0 + sqrt(3.0) + sqrt(3.0 + 2.0 * sqrt(3.0));
        x_hi = 8.0;

        double polar_p_sep = solver(&params, &separatrix_polynomial_polar, x_lo, x_hi);
        if (x == 0.0)
            return polar_p_sep;

        double equat_p_sep;
        if (x > 0.0)
        {
            x_lo = 1.0 + e;
            x_hi = 6 + 2. * e;

            equat_p_sep = solver(&params, &separatrix_polynomial_equat, x_lo, x_hi);

            x_lo = equat_p_sep;
            x_hi = polar_p_sep;
        }
        else
        {
            x_lo = polar_p_sep;
            x_hi = 12.0;
        }

        p_sep = solver(&params, &separatrix_polynomial_full, x_lo, x_hi);

        return p_sep;
    }
}

void get_separatrix_vector(double *separatrix, double *a, double *e, double *x, int length)
{

#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < length; i += 1)
    {
        separatrix[i] = get_separatrix(a[i], e[i], x[i]);
    }
}

double Y_to_xI_eq(double x, void *params_in)
{
    struct params_holder *params = (struct params_holder *)params_in;

    double a = params->a;
    double p = params->p;
    double e = params->e;
    double Y = params->Y;

    double E, L, Q;

    // get constants of motion
    KerrGeoConstantsOfMotion(&E, &L, &Q, a, p, e, x);
    double Y_ = L / sqrt(pow(L, 2) + Q);

    return Y - Y_;
}

#define YLIM 0.998
double Y_to_xI(double a, double p, double e, double Y)
{
    // TODO: check this
    if (abs(Y) > YLIM)
        return Y;
    // fills in x with 0.0
    struct params_holder params = {a, p, e, 0.0, Y};
    double x_lo, x_hi;

    // set limits
    // assume Y is close to x
    x_lo = Y - 0.15;
    x_hi = Y + 0.15;

    x_lo = x_lo > -YLIM ? x_lo : -YLIM;
    x_hi = x_hi < YLIM ? x_hi : YLIM;

    double x = solver(&params, &Y_to_xI_eq, x_lo, x_hi);

    return x;
}

void Y_to_xI_vector(double *x, double *a, double *p, double *e, double *Y, int length)
{

#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < length; i += 1)
    {
        x[i] = Y_to_xI(a[i], p[i], e[i], Y[i]);
    }
}

void set_threads(int num_threads)
{
#ifdef __USE_OMP__
    omp_set_num_threads(num_threads);
#else
    throw std::invalid_argument("Attempting to set threads for openMP, but FEW was not installed with openMP due to the use of the flag --no_omp used during installation.");
#endif
}

int get_threads()
{
#ifdef __USE_OMP__
    int num_threads;
#pragma omp parallel for
    for (int i = 0; i < 1; i += 1)
    {
        num_threads = omp_get_num_threads();
    }

    return num_threads;
#else
    return 0;
#endif // __USE_OMP__
}

double separatrix_KerrEquatorial(const double a, const double e)
{
    double result;
    double Compile_$2 = -0.875 + a;
    double Compile_$3 = pow(Compile_$2, 2);
    double Compile_$9 = pow(Compile_$2, 3);
    double Compile_$15 = pow(Compile_$2, 4);
    double Compile_$21 = pow(Compile_$2, 5);
    double Compile_$27 = pow(Compile_$2, 6);
    double Compile_$38 = pow(Compile_$2, 7);
    double Compile_$45 = pow(Compile_$2, 8);
    double Compile_$55 = pow(Compile_$2, 9);
    double Compile_$63 = pow(Compile_$2, 10);
    double Compile_$5 = 3200. * Compile_$3;
    double Compile_$6 = -1. + Compile_$5;
    double Compile_$92 = -0.4 + e;
    double Compile_$93 = pow(Compile_$92, 2);
    double Compile_$94 = 12.5 * Compile_$93;
    double Compile_$95 = -1. + Compile_$94;
    double Compile_$8 = -120. * Compile_$2;
    double Compile_$10 = 256000. * Compile_$9;
    double Compile_$11 = Compile_$10 + Compile_$8;
    double Compile_$14 = -12800. * Compile_$3;
    double Compile_$16 = 2.048e7 * Compile_$15;
    double Compile_$17 = 1. + Compile_$14 + Compile_$16;
    double Compile_$19 = 200. * Compile_$2;
    double Compile_$20 = -1.28e6 * Compile_$9;
    double Compile_$22 = 1.6384e9 * Compile_$21;
    double Compile_$23 = Compile_$19 + Compile_$20 + Compile_$22;
    double Compile_$25 = 28800. * Compile_$3;
    double Compile_$26 = -1.2288e8 * Compile_$15;
    double Compile_$28 = 1.31072e11 * Compile_$27;
    double Compile_$29 = -1. + Compile_$25 + Compile_$26 + Compile_$28;
    double Compile_$31 = -280. * Compile_$2;
    double Compile_$34 = 3.584e6 * Compile_$9;
    double Compile_$35 = -1.14688e10 * Compile_$21;
    double Compile_$39 = 1.048576e13 * Compile_$38;
    double Compile_$40 = Compile_$31 + Compile_$34 + Compile_$35 + Compile_$39;
    double Compile_$42 = -51200. * Compile_$3;
    double Compile_$43 = 4.096e8 * Compile_$15;
    double Compile_$44 = -1.048576e12 * Compile_$27;
    double Compile_$46 = 8.388608e14 * Compile_$45;
    double Compile_$47 = 1. + Compile_$42 + Compile_$43 + Compile_$44 + Compile_$46;
    double Compile_$51 = 360. * Compile_$2;
    double Compile_$52 = -7.68e6 * Compile_$9;
    double Compile_$53 = 4.42368e10 * Compile_$21;
    double Compile_$54 = -9.437184e13 * Compile_$38;
    double Compile_$56 = 6.7108864e16 * Compile_$55;
    double Compile_$57 = Compile_$51 + Compile_$52 + Compile_$53 + Compile_$54 + Compile_$56;
    double Compile_$59 = 80000. * Compile_$3;
    double Compile_$60 = -1.024e9 * Compile_$15;
    double Compile_$61 = 4.58752e12 * Compile_$27;
    double Compile_$62 = -8.388608e15 * Compile_$45;
    double Compile_$64 = 5.36870912e18 * Compile_$63;
    double Compile_$65 = -1. + Compile_$59 + Compile_$60 + Compile_$61 + Compile_$62 + Compile_$64;
    double Compile_$67 = -440. * Compile_$2;
    double Compile_$70 = 1.408e7 * Compile_$9;
    double Compile_$71 = -1.261568e11 * Compile_$21;
    double Compile_$74 = 4.6137344e14 * Compile_$38;
    double Compile_$75 = -7.38197504e17 * Compile_$55;
    double Compile_$78 = pow(Compile_$2, 11);
    double Compile_$79 = 4.294967296e20 * Compile_$78;
    double Compile_$80 = Compile_$67 + Compile_$70 + Compile_$71 + Compile_$74 + Compile_$75 + Compile_$79;
    double Compile_$82 = -115200. * Compile_$3;
    double Compile_$83 = 2.1504e9 * Compile_$15;
    double Compile_$84 = -1.4680064e13 * Compile_$27;
    double Compile_$85 = 4.52984832e16 * Compile_$45;
    double Compile_$86 = -6.442450944e19 * Compile_$63;
    double Compile_$87 = pow(Compile_$2, 12);
    double Compile_$88 = 3.4359738368e22 * Compile_$87;
    double Compile_$89 = 1. + Compile_$82 + Compile_$83 + Compile_$84 + Compile_$85 + Compile_$86 + Compile_$88;
    double Compile_$109 = -7.5 * Compile_$92;
    double Compile_$110 = pow(Compile_$92, 3);
    double Compile_$111 = 62.5 * Compile_$110;
    double Compile_$112 = Compile_$109 + Compile_$111;
    double Compile_$126 = -50. * Compile_$93;
    double Compile_$127 = pow(Compile_$92, 4);
    double Compile_$128 = 312.5 * Compile_$127;
    double Compile_$129 = 1. + Compile_$126 + Compile_$128;
    double Compile_$143 = 12.5 * Compile_$92;
    double Compile_$144 = -312.5 * Compile_$110;
    double Compile_$145 = pow(Compile_$92, 5);
    double Compile_$146 = 1562.5 * Compile_$145;
    double Compile_$147 = Compile_$143 + Compile_$144 + Compile_$146;
    double Compile_$161 = 112.5 * Compile_$93;
    double Compile_$162 = -1875. * Compile_$127;
    double Compile_$163 = pow(Compile_$92, 6);
    double Compile_$164 = 7812.5 * Compile_$163;
    double Compile_$165 = -1. + Compile_$161 + Compile_$162 + Compile_$164;
    double Compile_$179 = -17.5 * Compile_$92;
    double Compile_$180 = 875. * Compile_$110;
    double Compile_$181 = -10937.5 * Compile_$145;
    double Compile_$182 = pow(Compile_$92, 7);
    double Compile_$183 = 39062.5 * Compile_$182;
    double Compile_$184 = Compile_$179 + Compile_$180 + Compile_$181 + Compile_$183;
    double Compile_$198 = -200. * Compile_$93;
    double Compile_$199 = 6250. * Compile_$127;
    double Compile_$200 = -62500. * Compile_$163;
    double Compile_$201 = pow(Compile_$92, 8);
    double Compile_$202 = 195312.5 * Compile_$201;
    double Compile_$203 = 1. + Compile_$198 + Compile_$199 + Compile_$200 + Compile_$202;
    double Compile_$217 = 22.5 * Compile_$92;
    double Compile_$218 = -1875. * Compile_$110;
    double Compile_$219 = 42187.5 * Compile_$145;
    double Compile_$220 = -351562.5 * Compile_$182;
    double Compile_$221 = pow(Compile_$92, 9);
    double Compile_$222 = 976562.5 * Compile_$221;
    double Compile_$223 = Compile_$217 + Compile_$218 + Compile_$219 + Compile_$220 + Compile_$222;
    double Compile_$237 = 312.5 * Compile_$93;
    double Compile_$238 = -15625. * Compile_$127;
    double Compile_$239 = 273437.5 * Compile_$163;
    double Compile_$240 = -1.953125e6 * Compile_$201;
    double Compile_$241 = pow(Compile_$92, 10);
    double Compile_$242 = 4.8828125e6 * Compile_$241;
    double Compile_$243 = -1. + Compile_$237 + Compile_$238 + Compile_$239 + Compile_$240 + Compile_$242;
    double Compile_$257 = -27.5 * Compile_$92;
    double Compile_$258 = 3437.5 * Compile_$110;
    double Compile_$259 = -120312.5 * Compile_$145;
    double Compile_$260 = 1.71875e6 * Compile_$182;
    double Compile_$261 = -1.07421875e7 * Compile_$221;
    double Compile_$262 = pow(Compile_$92, 11);
    double Compile_$263 = 2.44140625e7 * Compile_$262;
    double Compile_$264 = Compile_$257 + Compile_$258 + Compile_$259 + Compile_$260 + Compile_$261 + Compile_$263;
    double Compile_$278 = -450. * Compile_$93;
    double Compile_$279 = 32812.5 * Compile_$127;
    double Compile_$280 = -875000. * Compile_$163;
    double Compile_$281 = 1.0546875e7 * Compile_$201;
    double Compile_$282 = -5.859375e7 * Compile_$241;
    double Compile_$283 = pow(Compile_$92, 12);
    double Compile_$284 = 1.220703125e8 * Compile_$283;
    double Compile_$285 = 1. + Compile_$278 + Compile_$279 + Compile_$280 + Compile_$281 + Compile_$282 + Compile_$284;
    double Compile_$299 = 32.5 * Compile_$92;
    double Compile_$300 = -5687.5 * Compile_$110;
    double Compile_$301 = 284375. * Compile_$145;
    double Compile_$302 = -6.09375e6 * Compile_$182;
    double Compile_$303 = 6.34765625e7 * Compile_$221;
    double Compile_$304 = -3.173828125e8 * Compile_$262;
    double Compile_$305 = pow(Compile_$92, 13);
    double Compile_$306 = 6.103515625e8 * Compile_$305;
    double Compile_$307 = Compile_$299 + Compile_$300 + Compile_$301 + Compile_$302 + Compile_$303 + Compile_$304 + Compile_$306;
    double Compile_$321 = 612.5 * Compile_$93;
    double Compile_$322 = -61250. * Compile_$127;
    double Compile_$323 = 2.296875e6 * Compile_$163;
    double Compile_$324 = -4.1015625e7 * Compile_$201;
    double Compile_$325 = 3.759765625e8 * Compile_$241;
    double Compile_$326 = -1.708984375e9 * Compile_$283;
    double Compile_$327 = pow(Compile_$92, 14);
    double Compile_$328 = 3.0517578125e9 * Compile_$327;
    double Compile_$329 = -1. + Compile_$321 + Compile_$322 + Compile_$323 + Compile_$324 + Compile_$325 + Compile_$326 + Compile_$328;
    double Compile_$343 = -37.5 * Compile_$92;
    double Compile_$344 = 8750. * Compile_$110;
    double Compile_$345 = -590625. * Compile_$145;
    double Compile_$346 = 1.7578125e7 * Compile_$182;
    double Compile_$347 = -2.685546875e8 * Compile_$221;
    double Compile_$348 = 2.197265625e9 * Compile_$262;
    double Compile_$349 = -9.1552734375e9 * Compile_$305;
    double Compile_$350 = pow(Compile_$92, 15);
    double Compile_$351 = 1.52587890625e10 * Compile_$350;
    double Compile_$352 = Compile_$343 + Compile_$344 + Compile_$345 + Compile_$346 + Compile_$347 + Compile_$348 + Compile_$349 + Compile_$351;
    double Compile_$366 = -800. * Compile_$93;
    double Compile_$367 = 105000. * Compile_$127;
    double Compile_$368 = -5.25e6 * Compile_$163;
    double Compile_$369 = 1.2890625e8 * Compile_$201;
    double Compile_$370 = -1.71875e9 * Compile_$241;
    double Compile_$371 = 1.26953125e10 * Compile_$283;
    double Compile_$372 = -4.8828125e10 * Compile_$327;
    double Compile_$373 = pow(Compile_$92, 16);
    double Compile_$374 = 7.62939453125e10 * Compile_$373;
    double Compile_$375 = 1. + Compile_$366 + Compile_$367 + Compile_$368 + Compile_$369 + Compile_$370 + Compile_$371 + Compile_$372 + Compile_$374;
    double Compile_$389 = 42.5 * Compile_$92;
    double Compile_$390 = -12750. * Compile_$110;
    double Compile_$391 = 1.115625e6 * Compile_$145;
    double Compile_$392 = -4.3828125e7 * Compile_$182;
    double Compile_$393 = 9.130859375e8 * Compile_$221;
    double Compile_$394 = -1.0791015625e10 * Compile_$262;
    double Compile_$395 = 7.26318359375e10 * Compile_$305;
    double Compile_$396 = -2.593994140625e11 * Compile_$350;
    double Compile_$397 = pow(Compile_$92, 17);
    double Compile_$398 = 3.814697265625e11 * Compile_$397;
    double Compile_$399 = Compile_$389 + Compile_$390 + Compile_$391 + Compile_$392 + Compile_$393 + Compile_$394 + Compile_$395 + Compile_$396 + Compile_$398;
    result = 2.91352319406094061986091651 - 0.00016284618369671501891938 * Compile_$11 - 0.00344098151801864926312442 * Compile_$112 - 8.454686467988030343e-7 * Compile_$11 * Compile_$112 + 0.00045975534530061279176401 * Compile_$129 + 3.522747836810062931e-7 * Compile_$11 * Compile_$129 - 0.00005090800216967482739708 * Compile_$147 - 7.96680630602420129e-8 * Compile_$11 * Compile_$147 + 3.90759688820813564179e-6 * Compile_$165 + 9.9464546613873438e-9 * Compile_$11 * Compile_$165 - 0.00001036956920747591485328 * Compile_$17 - 6.05515515618870803e-8 * Compile_$112 * Compile_$17 + 2.95811040821036276e-8 * Compile_$129 * Compile_$17 - 7.6325749584423864e-9 * Compile_$147 * Compile_$17 + 1.1334766240463451e-9 * Compile_$165 * Compile_$17 + 6.75482658581192536e-9 * Compile_$184 + 4.194186368161293e-10 * Compile_$11 * Compile_$184 + 6.3597769255516e-12 * Compile_$17 * Compile_$184 - 6.836183650071831449198908 * Compile_$2 - 0.011281349177524942504919 * Compile_$112 * Compile_$2 + 0.002439547820661392521926 * Compile_$129 * Compile_$2 - 0.000353272121843407841173 * Compile_$147 * Compile_$2 + 0.000027083682671818370053 * Compile_$165 * Compile_$2 + 2.823622629738801065e-6 * Compile_$184 * Compile_$2 - 7.215425898783023565e-8 * Compile_$203 - 5.952739996652805e-10 * Compile_$11 * Compile_$203 - 6.16649499750823e-11 * Compile_$17 * Compile_$203 - 1.56641183425492541e-6 * Compile_$2 * Compile_$203 + 1.562145286028911668e-8 * Compile_$223 + 1.7698887560237489e-10 * Compile_$11 * Compile_$223 + 2.123360153492348e-11 * Compile_$17 * Compile_$223 + 3.44279714744578915e-7 * Compile_$2 * Compile_$223 - 7.3822347684245341617e-7 * Compile_$23 - 4.5919311457885436e-9 * Compile_$112 * Compile_$23 + 2.5486136701661831e-9 * Compile_$129 * Compile_$23 - 7.288413520527012e-10 * Compile_$147 * Compile_$23 + 1.237400953795063e-10 * Compile_$165 * Compile_$23 - 3.0271966594178e-12 * Compile_$184 * Compile_$23 - 6.0820994338773e-12 * Compile_$203 * Compile_$23 + 2.40921404748626e-12 * Compile_$223 * Compile_$23 - 1.90937539524066598e-9 * Compile_$243 - 3.16610644745792e-11 * Compile_$11 * Compile_$243 - 4.2907763119221e-12 * Compile_$17 * Compile_$243 - 4.6980853020381921e-8 * Compile_$2 * Compile_$243 - 5.426687040614e-13 * Compile_$23 * Compile_$243 + 6.092314040983428e-11 * Compile_$264 + 2.6475141631424e-12 * Compile_$11 * Compile_$264 + 4.446061205118e-13 * Compile_$17 * Compile_$264 + 2.328725998042019e-9 * Compile_$2 * Compile_$264 + 6.68602106062e-14 * Compile_$23 * Compile_$264 + 3.755679334800628e-11 * Compile_$285 + 4.85180811662e-13 * Compile_$11 * Compile_$285 + 5.12583218294e-14 * Compile_$17 * Compile_$285 + 8.32367746170678e-10 * Compile_$2 * Compile_$285 + 4.2859071672e-15 * Compile_$23 * Compile_$285 - 5.624451895467586345e-8 * Compile_$29 - 3.615522469933619e-10 * Compile_$112 * Compile_$29 + 2.237066529670028e-10 * Compile_$129 * Compile_$29 - 6.95599024288007e-11 * Compile_$147 * Compile_$29 + 1.31423226871835e-11 * Compile_$165 * Compile_$29 - 6.543769582531e-13 * Compile_$184 * Compile_$29 - 5.799163645474e-13 * Compile_$203 * Compile_$29 + 2.6330235649347e-13 * Compile_$223 * Compile_$29 - 6.53491058985e-14 * Compile_$243 * Compile_$29 + 9.2764160197e-15 * Compile_$264 * Compile_$29 + 2.249016927e-16 * Compile_$285 * Compile_$29 - 1.180067182287779e-11 * Compile_$307 - 2.647460879526e-13 * Compile_$11 * Compile_$307 - 3.76202707199e-14 * Compile_$17 * Compile_$307 - 3.08643897220732e-10 * Compile_$2 * Compile_$307 - 4.8536712587e-15 * Compile_$23 * Compile_$307 - 5.83776826e-16 * Compile_$29 * Compile_$307 + 2.11679433740239e-12 * Compile_$329 + 6.40218339363e-14 * Compile_$11 * Compile_$329 + 1.01707468927e-14 * Compile_$17 * Compile_$329 + 6.0512123410093e-11 * Compile_$2 * Compile_$329 + 1.4671537978e-15 * Compile_$23 * Compile_$329 + 1.973409169e-16 * Compile_$29 * Compile_$329 - 2.3628336763492e-13 * Compile_$352 - 9.4413005311e-15 * Compile_$11 * Compile_$352 - 1.6758649055e-15 * Compile_$17 * Compile_$352 - 7.188180962268e-12 * Compile_$2 * Compile_$352 - 2.683294083e-16 * Compile_$23 * Compile_$352 - 3.9760166e-17 * Compile_$29 * Compile_$352 + 3.51033130781e-15 * Compile_$375 + 3.895051472e-16 * Compile_$11 * Compile_$375 + 1.020133463e-16 * Compile_$17 * Compile_$375 + 8.0325437025e-14 * Compile_$2 * Compile_$375 + 2.1601097e-17 * Compile_$23 * Compile_$375 + 3.9552016e-18 * Compile_$29 * Compile_$375 + 5.53515655667e-15 * Compile_$399 + 2.368402158e-16 * Compile_$11 * Compile_$399 + 3.59615932e-17 * Compile_$17 * Compile_$399 + 2.00624761053e-13 * Compile_$2 * Compile_$399 + 4.629843e-18 * Compile_$23 * Compile_$399 + 5.069367e-19 * Compile_$29 * Compile_$399 - 4.48566934374887613e-9 * Compile_$40 - 2.92345842669529e-11 * Compile_$112 * Compile_$40 + 1.99174687012693e-11 * Compile_$129 * Compile_$40 - 6.6440448007201e-12 * Compile_$147 * Compile_$40 + 1.3707428416706e-12 * Compile_$165 * Compile_$40 - 9.81798450497e-14 * Compile_$184 * Compile_$40 - 5.38553136e-14 * Compile_$203 * Compile_$40 + 2.802307661463e-14 * Compile_$223 * Compile_$40 - 7.5910209293e-15 * Compile_$243 * Compile_$40 + 1.2128758817e-15 * Compile_$264 * Compile_$40 - 9.7656393e-18 * Compile_$285 * Compile_$40 - 6.6495426e-17 * Compile_$307 * Compile_$40 + 2.51737791e-17 * Compile_$329 * Compile_$40 - 5.5482871e-18 * Compile_$352 * Compile_$40 + 6.525777e-19 * Compile_$375 * Compile_$40 + 4.48739e-20 * Compile_$399 * Compile_$40 - 3.6972136561018969e-10 * Compile_$47 - 2.4110616577454e-12 * Compile_$112 * Compile_$47 + 1.7934539240431e-12 * Compile_$129 * Compile_$47 - 6.355168311345e-13 * Compile_$147 * Compile_$47 + 1.412069842755e-13 * Compile_$165 * Compile_$47 - 1.28229706304e-14 * Compile_$184 * Compile_$47 - 4.8874745419e-15 * Compile_$203 * Compile_$47 + 2.92476132954e-15 * Compile_$223 * Compile_$47 - 8.581034423e-16 * Compile_$243 * Compile_$47 + 1.5165262e-16 * Compile_$264 * Compile_$47 - 5.3281875e-18 * Compile_$285 * Compile_$47 - 7.2426115e-18 * Compile_$307 * Compile_$47 + 3.0809479e-18 * Compile_$329 * Compile_$47 - 7.38143e-19 * Compile_$352 * Compile_$47 + 9.96894e-20 * Compile_$375 * Compile_$47 + 2.4049e-21 * Compile_$399 * Compile_$47 - 3.124047100581606e-11 * Compile_$57 - 2.018883439823e-13 * Compile_$112 * Compile_$57 + 1.629821726888e-13 * Compile_$129 * Compile_$57 - 6.08919144825e-14 * Compile_$147 * Compile_$57 + 1.44199305544e-14 * Compile_$165 * Compile_$57 - 1.5562573913e-15 * Compile_$184 * Compile_$57 - 4.336694924e-16 * Compile_$203 * Compile_$57 + 3.0074864819e-16 * Compile_$223 * Compile_$57 - 9.49802385e-17 * Compile_$243 * Compile_$57 + 1.83209563e-17 * Compile_$264 * Compile_$57 - 1.0714238e-18 * Compile_$285 * Compile_$57 - 7.585738e-19 * Compile_$307 * Compile_$57 + 3.648095e-19 * Compile_$329 * Compile_$57 - 9.44454e-20 * Compile_$352 * Compile_$57 + 1.43867e-20 * Compile_$375 * Compile_$57 - 1.176e-22 * Compile_$399 * Compile_$57 - 0.00318656888430662736679277 * Compile_$6 - 0.000013049320300835375528 * Compile_$112 * Compile_$6 + 4.3614564389187135878e-6 * Compile_$129 * Compile_$6 - 8.250286658015210544e-7 * Compile_$147 * Compile_$6 + 8.17852618284975678e-8 * Compile_$165 * Compile_$6 + 7.022679612923859e-9 * Compile_$184 * Compile_$6 - 5.294518727817022e-9 * Compile_$203 * Compile_$6 + 1.34849498326554545e-9 * Compile_$223 * Compile_$6 - 2.108677853260902e-10 * Compile_$243 * Compile_$6 + 1.36235584964043e-11 * Compile_$264 * Compile_$6 + 3.688732220644e-12 * Compile_$285 * Compile_$6 - 1.6125171774961e-12 * Compile_$307 * Compile_$6 + 3.495943851344e-13 * Compile_$329 * Compile_$6 - 4.6040699803e-14 * Compile_$352 * Compile_$6 + 1.0777724061e-15 * Compile_$375 * Compile_$6 + 1.2695985222e-15 * Compile_$399 * Compile_$6 - 2.69154391416394e-12 * Compile_$65 - 1.71074951826e-14 * Compile_$112 * Compile_$65 + 1.49250002201e-14 * Compile_$129 * Compile_$65 - 5.8447704206e-15 * Compile_$147 * Compile_$65 + 1.4632843663e-15 * Compile_$165 * Compile_$65 - 1.805372286e-16 * Compile_$184 * Compile_$65 - 3.75502146e-17 * Compile_$203 * Compile_$65 + 3.056710268e-17 * Compile_$223 * Compile_$65 - 1.03397205e-17 * Compile_$243 * Compile_$65 + 2.1543832e-18 * Compile_$264 * Compile_$65 - 1.702773e-19 * Compile_$285 * Compile_$65 - 7.6653e-20 * Compile_$307 * Compile_$65 + 4.20781e-20 * Compile_$329 * Compile_$65 - 1.15797e-20 * Compile_$352 * Compile_$65 + 2.0288e-21 * Compile_$375 * Compile_$65 - 1.13e-22 * Compile_$399 * Compile_$65 - 2.3552636742849e-13 * Compile_$80 - 1.4633663483e-15 * Compile_$112 * Compile_$80 + 1.375534921e-15 * Compile_$129 * Compile_$80 - 5.619666536e-16 * Compile_$147 * Compile_$80 + 1.477846906e-16 * Compile_$165 * Compile_$80 - 2.03154943e-17 * Compile_$184 * Compile_$80 - 3.1575075e-18 * Compile_$203 * Compile_$80 + 3.07742986e-18 * Compile_$223 * Compile_$80 - 1.1105829e-18 * Compile_$243 * Compile_$80 + 2.478883e-19 * Compile_$264 * Compile_$80 - 2.41396e-20 * Compile_$285 * Compile_$80 - 7.4715e-21 * Compile_$307 * Compile_$80 + 4.7679e-21 * Compile_$329 * Compile_$80 - 1.3123e-21 * Compile_$352 * Compile_$80 + 3.137e-22 * Compile_$375 * Compile_$80 - 3.96e-23 * Compile_$399 * Compile_$80 - 2.070955450059e-14 * Compile_$89 - 1.251908176e-16 * Compile_$112 * Compile_$89 + 1.263900432e-16 * Compile_$129 * Compile_$89 - 5.36302393e-17 * Compile_$147 * Compile_$89 + 1.472689e-17 * Compile_$165 * Compile_$89 - 2.2118417e-18 * Compile_$184 * Compile_$89 - 2.541341e-19 * Compile_$203 * Compile_$89 + 3.0441636e-19 * Compile_$223 * Compile_$89 - 1.16742e-19 * Compile_$243 * Compile_$89 + 2.77259e-20 * Compile_$264 * Compile_$89 - 3.1536e-21 * Compile_$285 * Compile_$89 - 7.157e-22 * Compile_$307 * Compile_$89 + 5.482e-22 * Compile_$329 * Compile_$89 - 1.005e-22 * Compile_$352 * Compile_$89 + 5.35e-23 * Compile_$375 * Compile_$89 - 1.713e-23 * Compile_$399 * Compile_$89 + 1.1514396684417654056222758 * Compile_$92 - 0.0000205468004819699230606 * Compile_$11 * Compile_$92 - 9.959977615612570796e-7 * Compile_$17 * Compile_$92 - 1.44727270401310843905687 * Compile_$2 * Compile_$92 - 5.2972955270696964e-8 * Compile_$23 * Compile_$92 - 2.886813350228537e-9 * Compile_$29 * Compile_$92 - 1.509603862795322e-10 * Compile_$40 * Compile_$92 - 6.6764891137085e-12 * Compile_$47 * Compile_$92 - 1.276317195595e-13 * Compile_$57 * Compile_$92 - 0.0005297205449556790753947 * Compile_$6 * Compile_$92 + 2.30972109316e-14 * Compile_$65 * Compile_$92 + 4.7508181712e-15 * Compile_$80 * Compile_$92 + 6.359512167e-16 * Compile_$89 * Compile_$92 + 0.02256216328824196672745628 * Compile_$95 - 7.965134025673000616e-7 * Compile_$11 * Compile_$95 - 7.87548456943294908e-8 * Compile_$17 * Compile_$95 + 0.025019571872207194992245 * Compile_$2 * Compile_$95 - 7.4839816170359549e-9 * Compile_$23 * Compile_$95 - 7.040293939633263e-10 * Compile_$29 * Compile_$95 - 6.61481821465013e-11 * Compile_$40 * Compile_$95 - 6.2267155461927e-12 * Compile_$47 * Compile_$95 - 5.878718666362e-13 * Compile_$57 * Compile_$95 - 6.6986500531244527943e-6 * Compile_$6 * Compile_$95 - 5.56821074926e-14 * Compile_$65 * Compile_$95 - 5.2906566763e-15 * Compile_$80 * Compile_$95 - 4.996962178e-16 * Compile_$89 * Compile_$95;
    return result;
}

// Secondary Spin functions

double P(double r, double a, double En, double xi)
{
    return En * r * r - a * xi;
}

double deltaP(double r, double a, double En, double xi, double deltaEn, double deltaxi)
{
    return deltaEn * r * r - xi / r - a * deltaxi;
}

double deltaRt(double r, double am1, double a0, double a1, double a2)
{
    return am1 / r + a0 + r * (a1 + r * a2);
}

void KerrEqSpinFrequenciesCorrection(double *deltaOmegaR_, double *deltaOmegaPhi_,
                                     double a, double p, double e, double x)
{
    // printf("a, p, e, x, sep = %f %f %f %f\n", a, p, e, x);
    double M = 1.0;
    double En = KerrGeoEnergy(a, p, e, x);
    double xi = KerrGeoAngularMomentum(a, p, e, x, En) - a * En;

    // get radial roots
    double r1, r2, r3, r4;
    KerrGeoRadialRoots(&r1, &r2, &r3, &r4, a, p, e, x, En, 0.);

    double deltaEn, deltaxi;

    deltaEn = (xi * (-(a * pow(En, 2) * pow(r1, 2) * pow(r2, 2)) - En * pow(r1, 2) * pow(r2, 2) * xi + pow(a, 2) * En * (pow(r1, 2) + r1 * r2 + pow(r2, 2)) * xi +
                     a * (pow(r1, 2) + r1 * (-2 + r2) + (-2 + r2) * r2) * pow(xi, 2))) /
              (pow(r1, 2) * pow(r2, 2) * (a * pow(En, 2) * r1 * r2 * (r1 + r2) + En * (pow(r1, 2) * (-2 + r2) + r1 * (-2 + r2) * r2 - 2 * pow(r2, 2)) * xi + 2 * a * pow(xi, 2)));

    deltaxi = ((pow(r1, 2) + r1 * r2 + pow(r2, 2)) * xi * (En * pow(r2, 2) - a * xi) * (-(En * pow(r1, 2)) + a * xi)) /
              (pow(r1, 2) * pow(r2, 2) * (a * pow(En, 2) * r1 * r2 * (r1 + r2) + En * (pow(r1, 2) * (-2 + r2) + r1 * (-2 + r2) * r2 - 2 * pow(r2, 2)) * xi + 2 * a * pow(xi, 2)));

    double am1, a0, a1, a2;
    am1 = (-2 * a * pow(xi, 2)) / (r1 * r2);
    a0 = -2 * En * (-(a * deltaxi) + deltaEn * pow(r1, 2) + deltaEn * r1 * r2 + deltaEn * pow(r2, 2)) + 2 * (a * deltaEn + deltaxi) * xi;
    a1 = -2 * deltaEn * En * (r1 + r2);
    a2 = -2 * deltaEn * En;

    double kr = (r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4); // convention without the sqrt
    double hr = (r1 - r2) / (r1 - r3);

    double rp = M + sqrt(1.0 - pow(a, 2));
    double rm = M - sqrt(1.0 - pow(a, 2));

    double hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp));
    double hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm));

    double Kkr = EllipticK(kr);         //(* Elliptic integral of the first kind *)
    double Ekr = EllipticE(kr);         //(* Elliptic integral of the second kind *)
    double Pihrkr = EllipticPi(hr, kr); //(* Elliptic integral of the third kind *)
    double Pihmkr = EllipticPi(hm, kr);
    double Pihpkr = EllipticPi(hp, kr);

    double Vtr3 = a * xi + ((pow(a, 2) + pow(r3, 2)) * P(r3, a, En, xi)) / CapitalDelta(r3, a);
    double deltaVtr3 = a * deltaxi + (r3 * r3 + a * a) / CapitalDelta(r3, a) * deltaP(r3, a, En, xi, deltaEn, deltaxi);

    double deltaIt1 = (2 * ((deltaEn * Pihrkr * (r2 - r3) * (4 + r1 + r2 + r3)) / 2. + (Ekr * (r1 - r3) * (deltaEn * r1 * r2 * r3 + 2 * xi)) / (2. * r1 * r3) +
                            ((r2 - r3) * ((Pihmkr * (pow(a, 2) + pow(rm, 2)) * deltaP(rm, a, En, xi, deltaEn, deltaxi)) / ((r2 - rm) * (r3 - rm)) -
                                          (Pihpkr * (pow(a, 2) + pow(rp, 2)) * deltaP(rp, a, En, xi, deltaEn, deltaxi)) / ((r2 - rp) * (r3 - rp)))) /
                                (-rm + rp) +
                            Kkr * (-0.5 * (deltaEn * (r1 - r3) * (r2 - r3)) + deltaVtr3))) /
                      sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4));

    double cK = Kkr * (-0.5 * (a2 * En * (r1 - r3) * (r2 - r3)) + (pow(a, 4) * En * r3 * (-am1 + pow(r3, 2) * (a1 + 2 * a2 * r3)) +
                                                                   2 * pow(a, 2) * En * pow(r3, 2) * (-(am1 * (-2 + r3)) + a0 * r3 + pow(r3, 3) * (a1 - a2 + 2 * a2 * r3)) +
                                                                   En * pow(r3, 5) * (-2 * a0 - am1 + r3 * (a1 * (-4 + r3) + 2 * a2 * (-3 + r3) * r3)) + 2 * pow(a, 3) * (2 * am1 + a0 * r3 - a2 * pow(r3, 3)) * xi +
                                                                   2 * a * r3 * (am1 * (-6 + 4 * r3) + r3 * (2 * a1 * (-1 + r3) * r3 + a2 * pow(r3, 3) + a0 * (-4 + 3 * r3))) * xi) /
                                                                      (pow(r3, 2) * pow(r3 - rm, 2) * pow(r3 - rp, 2)));
    double cEPi = (En * (a2 * Ekr * r2 * (r1 - r3) + Pihrkr * (r2 - r3) * (2 * a1 + a2 * (4 + r1 + r2 + 3 * r3)))) / 2.;
    double cPi = ((-r2 + r3) * ((Pihmkr * (pow(a, 2) + pow(rm, 2)) * P(rm, a, En, xi) * deltaRt(rm, am1, a0, a1, a2)) / ((r2 - rm) * pow(r3 - rm, 2) * rm) -
                                (Pihpkr * (pow(a, 2) + pow(rp, 2)) * P(rp, a, En, xi) * deltaRt(rp, am1, a0, a1, a2)) / ((r2 - rp) * pow(r3 - rp, 2) * rp))) /
                 (-rm + rp);

    double cE = (Ekr * ((2 * am1 * (-r1 + r3) * xi) / (a * r1) + (r2 * Vtr3 * deltaRt(r3, am1, a0, a1, a2)) / (r2 - r3))) / pow(r3, 2);

    double deltaIt2 = -((cE + cEPi + cK + cPi) / (pow(1 - pow(En, 2), 1.5) * sqrt((r1 - r3) * (r2 - r4))));
    double deltaIt = deltaIt1 + deltaIt2;

    double It = (2 * ((En * (Ekr * r2 * (r1 - r3) + Pihrkr * (r2 - r3) * (4 + r1 + r2 + r3))) / 2. +
                      ((r2 - r3) * ((Pihmkr * (pow(a, 2) + pow(rm, 2)) * P(rm, a, En, xi)) / ((r2 - rm) * (r3 - rm)) - (Pihpkr * (pow(a, 2) + pow(rp, 2)) * P(rp, a, En, xi)) / ((r2 - rp) * (r3 - rp)))) /
                          (-rm + rp) +
                      Kkr * (-0.5 * (En * (r1 - r3) * (r2 - r3)) + Vtr3))) /
                sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4));

    double VPhir3 = xi + a / CapitalDelta(r3, a) * P(r3, a, En, xi);
    double deltaVPhir3 = deltaxi + a / CapitalDelta(r3, a) * deltaP(r3, a, En, xi, deltaEn, deltaxi);

    double deltaIPhi1 = (2 * ((Ekr * (r1 - r3) * xi) / (a * r1 * r3) + (a * (r2 - r3) * ((Pihmkr * deltaP(rm, a, En, xi, deltaEn, deltaxi)) / ((r2 - rm) * (r3 - rm)) - (Pihpkr * deltaP(rp, a, En, xi, deltaEn, deltaxi)) / ((r2 - rp) * (r3 - rp)))) / (-rm + rp) + Kkr * deltaVPhir3)) /
                        sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4));

    double dK = (Kkr * (-(a * En * pow(r3, 2) * (2 * a0 * (-1 + r3) * r3 + (a1 + 2 * a2) * pow(r3, 3) + am1 * (-4 + 3 * r3))) - pow(a, 3) * En * r3 * (am1 - pow(r3, 2) * (a1 + 2 * a2 * r3)) -
                        pow(a, 2) * (am1 * (-4 + r3) - 2 * a0 * r3 - (a1 + 2 * a2 * (-1 + r3)) * pow(r3, 3)) * xi - pow(-2 + r3, 2) * r3 * (3 * am1 + r3 * (2 * a0 + a1 * r3)) * xi)) /
                (pow(r3, 2) * pow(r3 - rm, 2) * pow(r3 - rp, 2));

    double dPi = -((a * (r2 - r3) * ((Pihmkr * P(rm, a, En, xi) * deltaRt(rm, am1, a0, a1, a2)) / ((r2 - rm) * pow(r3 - rm, 2) * rm) - (Pihpkr * P(rp, a, En, xi) * deltaRt(rp, am1, a0, a1, a2)) / ((r2 - rp) * pow(r3 - rp, 2) * rp))) / (-rm + rp));
    double dE = (Ekr * ((-2 * am1 * (r1 - r3) * xi) / (pow(a, 2) * r1) + (r2 * VPhir3 * deltaRt(r3, am1, a0, a1, a2)) / (r2 - r3))) / pow(r3, 2);

    double deltaIPhi2 = -((dE + dK + dPi) / (pow(1 - pow(En, 2), 1.5) * sqrt((r1 - r3) * (r2 - r4))));
    double deltaIPhi = deltaIPhi1 + deltaIPhi2;

    double IPhi = (2 * ((a * (r2 - r3) * ((Pihmkr * P(rm, a, En, xi)) / ((r2 - rm) * (r3 - rm)) - (Pihpkr * P(rp, a, En, xi)) / ((r2 - rp) * (r3 - rp)))) / (-rm + rp) + Kkr * VPhir3)) /
                  sqrt((1 - pow(En, 2)) * (r1 - r3) * (r2 - r4));

    double deltaOmegaR = -M_PI / pow(It, 2) * deltaIt;
    double deltaOmegaPhi = deltaIPhi / It - IPhi / pow(It, 2) * deltaIt;

    //                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     cE,                deltaIt2,           It,                deltaIt,            deltaIPhi1, dK, dPi, dE, deltaIPhi2, deltaIPhi, IPhi
    // 0.952696869207406, 2.601147313747245, 11.11111111111111, 9.09090909090909, 1.450341827498306, 0, -0.001534290476164244, -0.1322435748015139, -0.1205695381380546, -0.02411390762761123, 0.0590591407311683, 0.002923427466192832, 0.5641101056459328, 1.435889894354067, 0.03336154445124933, 0.2091139909146586, 0.0217342349165277, 0.0003947869154376093, 1.584149072588183, 1.55761216767624, 1.782193864035892, 1.601724642759611, 1.58446319264442, -112.2728614607676, 1.498105017522236, 111.1816561327114, 1.459858647716873, -7.095639020498573, 133.1766110081966, -7.640013106344259, -0.1508069013343114, -34.72953487758193, 34.00126350567278, 0.7853682931498268, -0.2170281681543273, -0.3678350694886387, 4.044867174992484
    // 0.952697 2.601147                     11.111111          9.090909          1.450342           0.000000 -0.001534        -0.132244            -0.120570            -0.024114             0.059059            0.002923               0.564110           1.435890           0.033362             0.209114            0.021734            0.000395               1.584149           1.557612          1.782194           1.601725           1.584463          -112.272861         1.498105           111.181656         5.642003           -22.992171          -208.799031        -23.536545          -0.150807            -34.729535          34.001264          0.785368            -0.217028            -0.367835            4.044867
    // 0.952697 2.601147 11.111111 9.090909 1.450342 0.000000 -0.001534 -0.132244 -0.120570 -0.024114 0.059059 0.002923 0.564110 1.435890 0.033362 0.209114 0.021734 0.000395 1.584149 1.557612 1.782194 1.601725 1.584463 -112.272861 1.498105 111.181656 1.459859 -7.095639 133.176611 -7.640013 -0.150807 -34.729535 34.001264 0.785368 -0.217028 -0.367835 4.044867
    // 0.952696869207406, 2.601147313747245, 11.11111111111111, 9.09090909090909, 1.450341827498306, 0, -0.001534290476164244, -0.1322435748015139, -0.1205695381380546, -0.02411390762761123, 0.0590591407311683, 0.002923427466192832, 0.5641101056459328, 1.435889894354067, 0.03336154445124933, 0.2091139909146586, 0.0217342349165277, 0.0003947869154376093, 1.584149072588183, 1.55761216767624, 1.782193864035892, 1.601724642759611, 1.58446319264442, -112.2728614607676, 1.498105017522236, 111.1816561327114, 1.459858647716873, -7.095639020498573, 133.1766110081966, -7.640013106344259, -0.1508069013343114, -34.72953487758193, 34.00126350567278, 0.7853682931498268, -0.2170281681543273, -0.3678350694886387, 4.044867174992484
    // printf("En, xi, r1, r2, r3, r4, deltaEn, deltaxi, am1, a0, a1, a2, rm, rp, kr, hr, hm, hp, Kkr, Ekr, Pihrkr, Pihmkr, Pihpkr, cK, cEPi, cPi, cE,deltaIt2, It, deltaIt, deltaPhi1, dK, dPi, dE, deltaPhi2, deltaPhi, IPhi =%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",En, xi, r1, r2, r3, r4, deltaEn, deltaxi, am1, a0, a1, a2, rm, rp, kr, hr, hm, hp, Kkr, Ekr, Pihrkr, Pihmkr, Pihpkr, cK, cEPi, cPi, cE, deltaIt2, It, deltaIt, deltaIPhi1, dK, dPi, dE, deltaIPhi2, deltaIPhi, IPhi);
    *deltaOmegaR_ = deltaOmegaR;
    *deltaOmegaPhi_ = deltaOmegaPhi;
    printf("deltaOmegaR_, deltaOmegaPhi_, = %f %f\n", deltaOmegaR, deltaOmegaPhi);
}

void KerrEqSpinFrequenciesCorrVectorized(double *OmegaPhi_, double *OmegaTheta_, double *OmegaR_,
                                         double *a, double *p, double *e, double *x, int length)
{

#ifdef __USE_OMP__
#pragma omp parallel for
#endif
    for (int i = 0; i < length; i += 1)
    {

        KerrEqSpinFrequenciesCorrection(&OmegaR_[i], &OmegaPhi_[i],
                                        a[i], p[i], e[i], x[i]);
    }
}

/*
int main()
{
    double a = 0.5;
    double p = 10.0;
    double e = 0.2;
    double iota = 0.4;
    double x = cos(iota);

    double temp = KerrGeoMinoFrequencies(a, p, e, x);

    //printf("%e %e %e\n", En, L, C);
}
*/

// create a c++ function with input a,p,e and output the derivative of the omega angle with respect to p
// compile with: g++ -o domegaphi_dp domegaphi_dp.cc -I/usr/local/include -L/usr/local/lib -lgsl -lgslcblas -lm
// run with: ./domegaphi_dp
double DomegaPhi_Dp(double a, double p, double e){
double Compile_$1 = Power(e,2);
double Compile_$5 = -1 + Compile_$1;
double Compile_$6 = Power(a,2);
double Compile_$2 = -Compile_$1;
double Compile_$7 = Power(Compile_$5,2);
double Compile_$4 = 1/p;
double Compile_$45 = 1 + e;
double Compile_$46 = -e;
double Compile_$47 = 1 + Compile_$46;
double Compile_$48 = 1/Compile_$47;
double Compile_$50 = 1/Compile_$45;
double Compile_$51 = -(p*Compile_$50);
double Compile_$3 = 1 + Compile_$2;
double Compile_$8 = -4*Compile_$6*Compile_$7;
double Compile_$9 = -p;
double Compile_$10 = 3 + Compile_$1 + Compile_$9;
double Compile_$11 = Power(Compile_$10,2);
double Compile_$12 = p*Compile_$11;
double Compile_$13 = Compile_$12 + Compile_$8;
double Compile_$14 = 1/Compile_$13;
double Compile_$15 = 3*Compile_$1;
double Compile_$16 = 1 + p + Compile_$15;
double Compile_$17 = Compile_$16*Compile_$6;
double Compile_$18 = Power(p,-3);
double Compile_$19 = Power(a,6);
double Compile_$20 = Compile_$19*Compile_$7;
double Compile_$21 = -4*Compile_$1;
double Compile_$22 = -2 + p;
double Compile_$23 = Power(Compile_$22,2);
double Compile_$24 = Compile_$21 + Compile_$23;
double Compile_$25 = Power(p,2);
double Compile_$26 = Compile_$24*Compile_$25*Compile_$6;
double Compile_$27 = Power(a,4);
double Compile_$28 = 2 + p;
double Compile_$29 = Compile_$1*Compile_$28;
double Compile_$30 = -2 + p + Compile_$29;
double Compile_$31 = 2*p*Compile_$27*Compile_$30;
double Compile_$32 = Compile_$20 + Compile_$26 + Compile_$31;
double Compile_$33 = Compile_$18*Compile_$32;
double Compile_$34 = Sqrt(Compile_$33);
double Compile_$35 = -2*Compile_$34;
double Compile_$36 = -3 + p + Compile_$2 + Compile_$35;
double Compile_$37 = p*Compile_$36;
double Compile_$38 = Compile_$17 + Compile_$37;
double Compile_$39 = Compile_$14*Compile_$38*Compile_$5;
double Compile_$40 = 1 + Compile_$39;
double Compile_$53 = -(p*Compile_$48);
double Compile_$54 = 1/Compile_$3;
double Compile_$55 = 1/Compile_$40;
double Compile_$56 = 2*p*Compile_$54*Compile_$55;
double Compile_$49 = p*Compile_$48;
double Compile_$57 = Compile_$51 + Compile_$53 + Compile_$56;
double Compile_$58 = Power(Compile_$57,2);
double Compile_$59 = Sqrt(Compile_$58);
double Compile_$41 = -(Compile_$3*Compile_$4*Compile_$40);
double Compile_$42 = 1 + Compile_$41;
double Compile_$43 = Sqrt(Compile_$42);
double Compile_$60 = Compile_$51 + Compile_$53 + Compile_$56 + Compile_$59;
double Compile_$52 = Compile_$49 + Compile_$51;
double Compile_$61 = p*Compile_$50;
double Compile_$62 = -2*p*Compile_$54*Compile_$55;
double Compile_$63 = -Compile_$59;
double Compile_$64 = Compile_$49 + Compile_$61 + Compile_$62 + Compile_$63;
double Compile_$65 = Compile_$64/2.;
double Compile_$66 = Compile_$49 + Compile_$65;
double Compile_$67 = 1/Compile_$66;
double Compile_$68 = (Compile_$4*Compile_$45*Compile_$52*Compile_$60*Compile_$67)/2.;
double Compile_$69 = EllipticK(Compile_$68);
double Compile_$72 = Compile_$61 + Compile_$65;
double Compile_$81 = Compile_$60/2.;
double Compile_$82 = Compile_$49 + Compile_$61 + Compile_$81;
double Compile_$73 = Compile_$52*Compile_$67;
double Compile_$74 = EllipticPi(Compile_$73,Compile_$68);
double Compile_$89 = -Compile_$6;
double Compile_$90 = 1 + Compile_$89;
double Compile_$92 = Sqrt(Compile_$90);
double Compile_$44 = 4*Compile_$43;
double Compile_$93 = -Compile_$92;
double Compile_$107 = -1 + Compile_$61 + Compile_$93;
double Compile_$108 = 1/Compile_$107;
double Compile_$94 = -1 + Compile_$81 + Compile_$93;
double Compile_$96 = -2*Compile_$43*Compile_$6;
double Compile_$98 = Compile_$14*Compile_$38;
double Compile_$99 = Sqrt(Compile_$98);
double Compile_$100 = p*Compile_$99;
double Compile_$101 = a*Compile_$43;
double Compile_$102 = Compile_$100 + Compile_$101;
double Compile_$103 = -(a*Compile_$102);
double Compile_$104 = Compile_$103 + Compile_$44;
double Compile_$119 = -1 + Compile_$61 + Compile_$92;
double Compile_$120 = 1/Compile_$119;
double Compile_$114 = -1 + Compile_$81 + Compile_$92;
double Compile_$133 = -2*p*Compile_$10;
double Compile_$134 = Compile_$11 + Compile_$133;
double Compile_$135 = Power(Compile_$13,-2);
double Compile_$137 = 1/Sqrt(Compile_$33);
double Compile_$138 = 1 + Compile_$1;
double Compile_$139 = 2*p*Compile_$138*Compile_$27;
double Compile_$140 = 2*p*Compile_$24*Compile_$6;
double Compile_$141 = 2*Compile_$22*Compile_$25*Compile_$6;
double Compile_$142 = 2*Compile_$27*Compile_$30;
double Compile_$143 = Compile_$139 + Compile_$140 + Compile_$141 + Compile_$142;
double Compile_$144 = Compile_$143*Compile_$18;
double Compile_$145 = Power(p,-4);
double Compile_$146 = -3*Compile_$145*Compile_$32;
double Compile_$147 = Compile_$144 + Compile_$146;
double Compile_$148 = -(Compile_$137*Compile_$147);
double Compile_$149 = 1 + Compile_$148;
double Compile_$150 = p*Compile_$149;
double Compile_$151 = -3 + p + Compile_$150 + Compile_$2 + Compile_$35 + Compile_$6;
double Compile_$91 = 1/Sqrt(Compile_$90);
double Compile_$156 = Power(p,-2);
double Compile_$169 = -Compile_$50;
double Compile_$158 = -(Compile_$134*Compile_$135*Compile_$38*Compile_$5);
double Compile_$159 = Compile_$14*Compile_$151*Compile_$5;
double Compile_$160 = Compile_$158 + Compile_$159;
double Compile_$173 = -Compile_$48;
double Compile_$174 = 2*Compile_$54*Compile_$55;
double Compile_$175 = Power(Compile_$40,-2);
double Compile_$176 = -2*p*Compile_$160*Compile_$175*Compile_$54;
double Compile_$177 = 1/Sqrt(Compile_$58);
double Compile_$178 = Compile_$169 + Compile_$173 + Compile_$174 + Compile_$176;
double Compile_$78 = EllipticE(Compile_$68);
double Compile_$166 = -0.5*(Compile_$4*Compile_$45*Compile_$52*Compile_$60*Compile_$67);
double Compile_$167 = 1 + Compile_$166;
double Compile_$95 = 1/Compile_$94;
double Compile_$97 = 1 + Compile_$92;
double Compile_$109 = Compile_$108*Compile_$52*Compile_$67*Compile_$94;
double Compile_$110 = EllipticPi(Compile_$109,Compile_$68);
double Compile_$111 = -(Compile_$108*Compile_$110*Compile_$72);
double Compile_$112 = Compile_$111 + Compile_$69;
double Compile_$115 = 1/Compile_$114;
double Compile_$116 = 1 + Compile_$93;
double Compile_$121 = Compile_$114*Compile_$120*Compile_$52*Compile_$67;
double Compile_$122 = EllipticPi(Compile_$121,Compile_$68);
double Compile_$123 = -(Compile_$120*Compile_$122*Compile_$72);
double Compile_$124 = Compile_$123 + Compile_$69;
double Compile_$70 = 1/Compile_$69;
double Compile_$179 = Compile_$177*Compile_$178*Compile_$57;
double Compile_$180 = Compile_$169 + Compile_$173 + Compile_$174 + Compile_$176 + Compile_$179;
double Compile_$194 = 2*Compile_$43*Compile_$97;
double Compile_$195 = Compile_$103 + Compile_$194;
double Compile_$155 = 1/Sqrt(Compile_$42);
double Compile_$157 = Compile_$156*Compile_$3*Compile_$40;
double Compile_$161 = -(Compile_$160*Compile_$3*Compile_$4);
double Compile_$162 = Compile_$157 + Compile_$161;
double Compile_$132 = 1/Sqrt(Compile_$98);
double Compile_$136 = -(Compile_$134*Compile_$135*Compile_$38);
double Compile_$152 = Compile_$14*Compile_$151;
double Compile_$153 = Compile_$136 + Compile_$152;
double Compile_$154 = (p*Compile_$132*Compile_$153)/2.;
double Compile_$163 = (a*Compile_$155*Compile_$162)/2.;
double Compile_$164 = 1/Compile_$52;
double Compile_$165 = 1/Compile_$60;
double Compile_$168 = 1/Compile_$167;
double Compile_$170 = Compile_$169 + Compile_$48;
double Compile_$171 = (Compile_$170*Compile_$4*Compile_$45*Compile_$60*Compile_$67)/2.;
double Compile_$172 = -0.5*(Compile_$156*Compile_$45*Compile_$52*Compile_$60*Compile_$67);
double Compile_$181 = (Compile_$180*Compile_$4*Compile_$45*Compile_$52*Compile_$67)/2.;
double Compile_$182 = Power(Compile_$66,-2);
double Compile_$183 = -2*Compile_$54*Compile_$55;
double Compile_$184 = 2*p*Compile_$160*Compile_$175*Compile_$54;
double Compile_$185 = -(Compile_$177*Compile_$178*Compile_$57);
double Compile_$186 = Compile_$183 + Compile_$184 + Compile_$185 + Compile_$48 + Compile_$50;
double Compile_$187 = Compile_$186/2.;
double Compile_$188 = Compile_$187 + Compile_$48;
double Compile_$189 = -0.5*(Compile_$182*Compile_$188*Compile_$4*Compile_$45*Compile_$52*Compile_$60);
double Compile_$190 = Compile_$171 + Compile_$172 + Compile_$181 + Compile_$189;
double Compile_$192 = -(Compile_$167*Compile_$69);
double Compile_$193 = Compile_$192 + Compile_$78;
double Compile_$210 = Power(Compile_$107,-2);
double Compile_$221 = -(Compile_$108*Compile_$52*Compile_$67*Compile_$94);
double Compile_$222 = Compile_$221 + Compile_$68;
double Compile_$197 = 2*Compile_$116*Compile_$43;
double Compile_$198 = Compile_$103 + Compile_$197;
double Compile_$205 = Compile_$154 + Compile_$163 + Compile_$99;
double Compile_$206 = -(a*Compile_$205);
double Compile_$209 = p*Compile_$164*Compile_$165*Compile_$168*Compile_$190*Compile_$193*Compile_$50*Compile_$66;
double Compile_$212 = Compile_$187 + Compile_$50;
double Compile_$216 = -1 + Compile_$68;
double Compile_$217 = 1/Compile_$216;
double Compile_$218 = Compile_$217*Compile_$78;
double Compile_$248 = Power(Compile_$119,-2);
double Compile_$255 = -(Compile_$114*Compile_$120*Compile_$52*Compile_$67);
double Compile_$256 = Compile_$255 + Compile_$68;
double Compile_$232 = Power(Compile_$52,2);
double Compile_$196 = Compile_$112*Compile_$195*Compile_$95;
double Compile_$199 = -(Compile_$115*Compile_$124*Compile_$198);
double Compile_$200 = Compile_$196 + Compile_$199;
double Compile_$71 = (Compile_$60*Compile_$69)/2.;
double Compile_$75 = Compile_$72*Compile_$74;
double Compile_$76 = Compile_$71 + Compile_$75;
double Compile_$77 = 2*Compile_$43*Compile_$76;
double Compile_$79 = p*Compile_$50*Compile_$66*Compile_$78;
double Compile_$80 = -(Compile_$25*Compile_$48*Compile_$50);
double Compile_$83 = (Compile_$60*Compile_$82)/2.;
double Compile_$84 = Compile_$80 + Compile_$83;
double Compile_$85 = Compile_$69*Compile_$84;
double Compile_$86 = Compile_$72*Compile_$74*Compile_$82;
double Compile_$87 = Compile_$79 + Compile_$85 + Compile_$86;
double Compile_$88 = (Compile_$43*Compile_$87)/2.;
double Compile_$105 = Compile_$104*Compile_$97;
double Compile_$106 = Compile_$105 + Compile_$96;
double Compile_$113 = Compile_$106*Compile_$112*Compile_$95;
double Compile_$117 = Compile_$104*Compile_$116;
double Compile_$118 = Compile_$117 + Compile_$96;
double Compile_$125 = -(Compile_$115*Compile_$118*Compile_$124);
double Compile_$126 = Compile_$113 + Compile_$125;
double Compile_$127 = Compile_$126*Compile_$91;
double Compile_$128 = Compile_$127 + Compile_$77 + Compile_$88;
double Compile_$129 = Compile_$128*Compile_$70;
double Compile_$130 = Compile_$129 + Compile_$44;
double Compile_$191 = Power(Compile_$69,-2);
double Compile_$296 = -(Compile_$52*Compile_$67);
double Compile_$297 = Compile_$296 + Compile_$68;
double Compile_$321 = Compile_$180/2.;
double Compile_$322 = Compile_$321 + Compile_$48 + Compile_$50;
double Compile_$290 = Compile_$166 + Compile_$73;
double Compile_$291 = 1/Compile_$290;
double Compile_$292 = Compile_$218 + Compile_$74;
double Compile_$293 = (Compile_$190*Compile_$291*Compile_$292)/2.;
double Compile_$294 = -1 + Compile_$73;
double Compile_$295 = 1/Compile_$294;
double Compile_$298 = 1/Compile_$297;
double Compile_$299 = Compile_$170*Compile_$67;
double Compile_$300 = -(Compile_$182*Compile_$188*Compile_$52);
double Compile_$301 = Compile_$299 + Compile_$300;
double Compile_$302 = Compile_$164*Compile_$297*Compile_$66*Compile_$69;
double Compile_$303 = Compile_$182*Compile_$232;
double Compile_$304 = Compile_$166 + Compile_$303;
double Compile_$305 = Compile_$164*Compile_$304*Compile_$66*Compile_$74;
double Compile_$306 = Compile_$302 + Compile_$305 + Compile_$78;
double Compile_$307 = (Compile_$295*Compile_$298*Compile_$301*Compile_$306)/2.;
double Compile_$308 = Compile_$293 + Compile_$307;
double Compile_$202 = Power(Compile_$94,-2);
double Compile_$283 = 2*Compile_$155*Compile_$162;
double Compile_$211 = Compile_$110*Compile_$210*Compile_$50*Compile_$72;
double Compile_$213 = -(Compile_$108*Compile_$110*Compile_$212);
double Compile_$214 = Compile_$109 + Compile_$166;
double Compile_$215 = 1/Compile_$214;
double Compile_$219 = Compile_$110 + Compile_$218;
double Compile_$220 = (Compile_$190*Compile_$215*Compile_$219)/2.;
double Compile_$223 = 1/Compile_$222;
double Compile_$224 = -1 + Compile_$109;
double Compile_$225 = 1/Compile_$224;
double Compile_$226 = (Compile_$108*Compile_$180*Compile_$52*Compile_$67)/2.;
double Compile_$227 = -(Compile_$210*Compile_$50*Compile_$52*Compile_$67*Compile_$94);
double Compile_$228 = Compile_$108*Compile_$170*Compile_$67*Compile_$94;
double Compile_$229 = -(Compile_$108*Compile_$182*Compile_$188*Compile_$52*Compile_$94);
double Compile_$230 = Compile_$226 + Compile_$227 + Compile_$228 + Compile_$229;
double Compile_$231 = Compile_$107*Compile_$164*Compile_$222*Compile_$66*Compile_$69*Compile_$95;
double Compile_$233 = Power(Compile_$94,2);
double Compile_$234 = Compile_$182*Compile_$210*Compile_$232*Compile_$233;
double Compile_$235 = Compile_$166 + Compile_$234;
double Compile_$236 = Compile_$107*Compile_$110*Compile_$164*Compile_$235*Compile_$66*Compile_$95;
double Compile_$237 = Compile_$231 + Compile_$236 + Compile_$78;
double Compile_$238 = (Compile_$223*Compile_$225*Compile_$230*Compile_$237)/2.;
double Compile_$239 = Compile_$220 + Compile_$238;
double Compile_$240 = -(Compile_$108*Compile_$239*Compile_$72);
double Compile_$241 = Compile_$209 + Compile_$211 + Compile_$213 + Compile_$240;
double Compile_$243 = Power(Compile_$114,-2);
double Compile_$333 = -(Compile_$155*Compile_$162*Compile_$6);
double Compile_$334 = Compile_$206 + Compile_$283;
double Compile_$249 = Compile_$122*Compile_$248*Compile_$50*Compile_$72;
double Compile_$250 = -(Compile_$120*Compile_$122*Compile_$212);
double Compile_$251 = Compile_$121 + Compile_$166;
double Compile_$252 = 1/Compile_$251;
double Compile_$253 = Compile_$122 + Compile_$218;
double Compile_$254 = (Compile_$190*Compile_$252*Compile_$253)/2.;
double Compile_$257 = 1/Compile_$256;
double Compile_$258 = -1 + Compile_$121;
double Compile_$259 = 1/Compile_$258;
double Compile_$260 = (Compile_$120*Compile_$180*Compile_$52*Compile_$67)/2.;
double Compile_$261 = -(Compile_$114*Compile_$248*Compile_$50*Compile_$52*Compile_$67);
double Compile_$262 = Compile_$114*Compile_$120*Compile_$170*Compile_$67;
double Compile_$263 = -(Compile_$114*Compile_$120*Compile_$182*Compile_$188*Compile_$52);
double Compile_$264 = Compile_$260 + Compile_$261 + Compile_$262 + Compile_$263;
double Compile_$265 = Compile_$115*Compile_$119*Compile_$164*Compile_$256*Compile_$66*Compile_$69;
double Compile_$266 = Power(Compile_$114,2);
double Compile_$267 = Compile_$182*Compile_$232*Compile_$248*Compile_$266;
double Compile_$268 = Compile_$166 + Compile_$267;
double Compile_$269 = Compile_$115*Compile_$119*Compile_$122*Compile_$164*Compile_$268*Compile_$66;
double Compile_$270 = Compile_$265 + Compile_$269 + Compile_$78;
double Compile_$271 = (Compile_$257*Compile_$259*Compile_$264*Compile_$270)/2.;
double Compile_$272 = Compile_$254 + Compile_$271;
double Compile_$273 = -(Compile_$120*Compile_$272*Compile_$72);
double Compile_$274 = Compile_$209 + Compile_$249 + Compile_$250 + Compile_$273;
return -(((Compile_$100 + Compile_$101 + (a*Compile_$200*Compile_$70*Compile_$91)/2.)*(Compile_$283 - p*Compile_$128*Compile_$164*Compile_$165*Compile_$168*Compile_$190*Compile_$191*Compile_$193*Compile_$50*Compile_$66 + Compile_$70*(2*Compile_$43*((p*Compile_$164*Compile_$168*Compile_$190*Compile_$193*Compile_$50*Compile_$66)/2. + (Compile_$180*Compile_$69)/2. + Compile_$308*Compile_$72 + Compile_$212*Compile_$74) + Compile_$155*Compile_$162*Compile_$76 + (Compile_$43*(Compile_$322*Compile_$72*Compile_$74 + p*Compile_$188*Compile_$50*Compile_$78 + Compile_$50*Compile_$66*Compile_$78 + (Compile_$164*Compile_$165*Compile_$190*Compile_$25*Power(Compile_$66,2)*(-Compile_$69 + Compile_$78))/Power(Compile_$45,2) + Compile_$308*Compile_$72*Compile_$82 + Compile_$212*Compile_$74*Compile_$82 + Compile_$69*(-2*p*Compile_$48*Compile_$50 + (Compile_$322*Compile_$60)/2. + (Compile_$180*Compile_$82)/2.) + p*Compile_$164*Compile_$165*Compile_$168*Compile_$190*Compile_$193*Compile_$50*Compile_$66*Compile_$84))/2. + (Compile_$155*Compile_$162*Compile_$87)/4. + Compile_$91*(-0.5*(Compile_$106*Compile_$112*Compile_$180*Compile_$202) + (Compile_$118*Compile_$124*Compile_$180*Compile_$243)/2. - Compile_$115*Compile_$118*Compile_$274 - Compile_$115*Compile_$124*(Compile_$333 + Compile_$116*Compile_$334) + Compile_$106*Compile_$241*Compile_$95 + Compile_$112*Compile_$95*(Compile_$333 + Compile_$334*Compile_$97)))))/Power(Compile_$130,2)) + (Compile_$154 + Compile_$163 - (a*p*Compile_$164*Compile_$165*Compile_$168*Compile_$190*Compile_$191*Compile_$193*Compile_$200*Compile_$50*Compile_$66*Compile_$91)/2. + (a*Compile_$70*Compile_$91*(-0.5*(Compile_$112*Compile_$180*Compile_$195*Compile_$202) - Compile_$115*Compile_$124*(Compile_$116*Compile_$155*Compile_$162 + Compile_$206) + (Compile_$124*Compile_$180*Compile_$198*Compile_$243)/2. - Compile_$115*Compile_$198*Compile_$274 + Compile_$195*Compile_$241*Compile_$95 + Compile_$112*Compile_$95*(Compile_$206 + Compile_$155*Compile_$162*Compile_$97)))/2. + Compile_$99)/Compile_$130;
}

double DomegaPhi_De(double a, double p, double e){
double Compile_$131 = Power(e,2);
double Compile_$207 = -1 + Compile_$131;
double Compile_$208 = Power(a,2);
double Compile_$201 = -Compile_$131;
double Compile_$242 = Power(Compile_$207,2);
double Compile_$204 = 1/p;
double Compile_$330 = 1 + e;
double Compile_$331 = -e;
double Compile_$332 = 1 + Compile_$331;
double Compile_$335 = 1/Compile_$332;
double Compile_$337 = 1/Compile_$330;
double Compile_$338 = -(p*Compile_$337);
double Compile_$203 = 1 + Compile_$201;
double Compile_$244 = -4*Compile_$208*Compile_$242;
double Compile_$245 = -p;
double Compile_$246 = 3 + Compile_$131 + Compile_$245;
double Compile_$247 = Power(Compile_$246,2);
double Compile_$275 = p*Compile_$247;
double Compile_$276 = Compile_$244 + Compile_$275;
double Compile_$277 = 1/Compile_$276;
double Compile_$278 = 3*Compile_$131;
double Compile_$279 = 1 + p + Compile_$278;
double Compile_$280 = Compile_$208*Compile_$279;
double Compile_$281 = Power(p,-3);
double Compile_$282 = Power(a,6);
double Compile_$284 = Compile_$242*Compile_$282;
double Compile_$285 = -4*Compile_$131;
double Compile_$286 = -2 + p;
double Compile_$287 = Power(Compile_$286,2);
double Compile_$288 = Compile_$285 + Compile_$287;
double Compile_$289 = Power(p,2);
double Compile_$309 = Compile_$208*Compile_$288*Compile_$289;
double Compile_$310 = Power(a,4);
double Compile_$311 = 2 + p;
double Compile_$312 = Compile_$131*Compile_$311;
double Compile_$313 = -2 + p + Compile_$312;
double Compile_$314 = 2*p*Compile_$310*Compile_$313;
double Compile_$315 = Compile_$284 + Compile_$309 + Compile_$314;
double Compile_$316 = Compile_$281*Compile_$315;
double Compile_$317 = Sqrt(Compile_$316);
double Compile_$318 = -2*Compile_$317;
double Compile_$319 = -3 + p + Compile_$201 + Compile_$318;
double Compile_$320 = p*Compile_$319;
double Compile_$323 = Compile_$280 + Compile_$320;
double Compile_$324 = Compile_$207*Compile_$277*Compile_$323;
double Compile_$325 = 1 + Compile_$324;
double Compile_$340 = -(p*Compile_$335);
double Compile_$341 = 1/Compile_$203;
double Compile_$342 = 1/Compile_$325;
double Compile_$343 = 2*p*Compile_$341*Compile_$342;
double Compile_$336 = p*Compile_$335;
double Compile_$344 = Compile_$338 + Compile_$340 + Compile_$343;
double Compile_$345 = Power(Compile_$344,2);
double Compile_$346 = Sqrt(Compile_$345);
double Compile_$326 = -(Compile_$203*Compile_$204*Compile_$325);
double Compile_$327 = 1 + Compile_$326;
double Compile_$328 = Sqrt(Compile_$327);
double Compile_$347 = Compile_$338 + Compile_$340 + Compile_$343 + Compile_$346;
double Compile_$339 = Compile_$336 + Compile_$338;
double Compile_$348 = p*Compile_$337;
double Compile_$349 = -2*p*Compile_$341*Compile_$342;
double Compile_$350 = -Compile_$346;
double Compile_$351 = Compile_$336 + Compile_$348 + Compile_$349 + Compile_$350;
double Compile_$352 = Compile_$351/2.;
double Compile_$353 = Compile_$336 + Compile_$352;
double Compile_$354 = 1/Compile_$353;
double Compile_$355 = (Compile_$204*Compile_$330*Compile_$339*Compile_$347*Compile_$354)/2.;
double Compile_$356 = EllipticK(Compile_$355);
double Compile_$359 = Compile_$348 + Compile_$352;
double Compile_$368 = Compile_$347/2.;
double Compile_$369 = Compile_$336 + Compile_$348 + Compile_$368;
double Compile_$360 = Compile_$339*Compile_$354;
double Compile_$361 = EllipticPi(Compile_$360,Compile_$355);
double Compile_$376 = -Compile_$208;
double Compile_$377 = 1 + Compile_$376;
double Compile_$379 = Sqrt(Compile_$377);
double Compile_$329 = 4*Compile_$328;
double Compile_$380 = -Compile_$379;
double Compile_$394 = -1 + Compile_$348 + Compile_$380;
double Compile_$395 = 1/Compile_$394;
double Compile_$381 = -1 + Compile_$368 + Compile_$380;
double Compile_$383 = -2*Compile_$208*Compile_$328;
double Compile_$385 = Compile_$277*Compile_$323;
double Compile_$386 = Sqrt(Compile_$385);
double Compile_$387 = p*Compile_$386;
double Compile_$388 = a*Compile_$328;
double Compile_$389 = Compile_$387 + Compile_$388;
double Compile_$390 = -(a*Compile_$389);
double Compile_$391 = Compile_$329 + Compile_$390;
double Compile_$406 = -1 + Compile_$348 + Compile_$379;
double Compile_$407 = 1/Compile_$406;
double Compile_$401 = -1 + Compile_$368 + Compile_$379;
double Compile_$420 = 6*e*Compile_$208;
double Compile_$421 = -2*e;
double Compile_$422 = 4*e*Compile_$207*Compile_$282;
double Compile_$423 = -8*e*Compile_$208*Compile_$289;
double Compile_$424 = 4*e*p*Compile_$310*Compile_$311;
double Compile_$425 = Compile_$422 + Compile_$423 + Compile_$424;
double Compile_$426 = 1/Sqrt(Compile_$316);
double Compile_$427 = -(Compile_$281*Compile_$425*Compile_$426);
double Compile_$428 = Compile_$421 + Compile_$427;
double Compile_$429 = p*Compile_$428;
double Compile_$430 = Compile_$420 + Compile_$429;
double Compile_$432 = -16*e*Compile_$207*Compile_$208;
double Compile_$433 = 4*e*p*Compile_$246;
double Compile_$434 = Compile_$432 + Compile_$433;
double Compile_$435 = Power(Compile_$276,-2);
double Compile_$378 = 1/Sqrt(Compile_$377);
double Compile_$450 = Power(Compile_$332,-2);
double Compile_$451 = p*Compile_$450;
double Compile_$439 = Compile_$207*Compile_$277*Compile_$430;
double Compile_$440 = -(Compile_$207*Compile_$323*Compile_$434*Compile_$435);
double Compile_$441 = 2*e*Compile_$277*Compile_$323;
double Compile_$442 = Compile_$439 + Compile_$440 + Compile_$441;
double Compile_$452 = Power(Compile_$330,-2);
double Compile_$454 = Power(Compile_$325,-2);
double Compile_$456 = Power(Compile_$203,-2);
double Compile_$458 = -(p*Compile_$450);
double Compile_$459 = p*Compile_$452;
double Compile_$460 = -2*p*Compile_$341*Compile_$442*Compile_$454;
double Compile_$461 = 4*e*p*Compile_$342*Compile_$456;
double Compile_$462 = Compile_$458 + Compile_$459 + Compile_$460 + Compile_$461;
double Compile_$463 = 1/Sqrt(Compile_$345);
double Compile_$365 = EllipticE(Compile_$355);
double Compile_$477 = -0.5*(Compile_$204*Compile_$330*Compile_$339*Compile_$347*Compile_$354);
double Compile_$478 = 1 + Compile_$477;
double Compile_$382 = 1/Compile_$381;
double Compile_$384 = 1 + Compile_$379;
double Compile_$396 = Compile_$339*Compile_$354*Compile_$381*Compile_$395;
double Compile_$397 = EllipticPi(Compile_$396,Compile_$355);
double Compile_$398 = -(Compile_$359*Compile_$395*Compile_$397);
double Compile_$399 = Compile_$356 + Compile_$398;
double Compile_$402 = 1/Compile_$401;
double Compile_$403 = 1 + Compile_$380;
double Compile_$408 = Compile_$339*Compile_$354*Compile_$401*Compile_$407;
double Compile_$409 = EllipticPi(Compile_$408,Compile_$355);
double Compile_$410 = -(Compile_$359*Compile_$407*Compile_$409);
double Compile_$411 = Compile_$356 + Compile_$410;
double Compile_$357 = 1/Compile_$356;
double Compile_$443 = -(Compile_$203*Compile_$204*Compile_$442);
double Compile_$444 = 2*e*Compile_$204*Compile_$325;
double Compile_$445 = Compile_$443 + Compile_$444;
double Compile_$446 = 1/Sqrt(Compile_$327);
double Compile_$419 = 1/Sqrt(Compile_$385);
double Compile_$431 = Compile_$277*Compile_$430;
double Compile_$436 = -(Compile_$323*Compile_$434*Compile_$435);
double Compile_$437 = Compile_$431 + Compile_$436;
double Compile_$438 = (p*Compile_$419*Compile_$437)/2.;
double Compile_$447 = (a*Compile_$445*Compile_$446)/2.;
double Compile_$470 = Compile_$344*Compile_$462*Compile_$463;
double Compile_$471 = Compile_$458 + Compile_$459 + Compile_$460 + Compile_$461 + Compile_$470;
double Compile_$483 = 2*Compile_$328*Compile_$384;
double Compile_$484 = Compile_$390 + Compile_$483;
double Compile_$448 = 1/Compile_$339;
double Compile_$449 = 1/Compile_$347;
double Compile_$453 = -(p*Compile_$452);
double Compile_$455 = 2*p*Compile_$341*Compile_$442*Compile_$454;
double Compile_$457 = -4*e*p*Compile_$342*Compile_$456;
double Compile_$464 = -(Compile_$344*Compile_$462*Compile_$463);
double Compile_$465 = Compile_$451 + Compile_$453 + Compile_$455 + Compile_$457 + Compile_$464;
double Compile_$466 = Compile_$465/2.;
double Compile_$467 = Compile_$451 + Compile_$466;
double Compile_$468 = Power(Compile_$353,-2);
double Compile_$469 = -0.5*(Compile_$204*Compile_$330*Compile_$339*Compile_$347*Compile_$467*Compile_$468);
double Compile_$472 = (Compile_$204*Compile_$330*Compile_$339*Compile_$354*Compile_$471)/2.;
double Compile_$473 = Compile_$451 + Compile_$459;
double Compile_$474 = (Compile_$204*Compile_$330*Compile_$347*Compile_$354*Compile_$473)/2.;
double Compile_$475 = (Compile_$204*Compile_$339*Compile_$347*Compile_$354)/2.;
double Compile_$476 = Compile_$469 + Compile_$472 + Compile_$474 + Compile_$475;
double Compile_$479 = 1/Compile_$478;
double Compile_$481 = -(Compile_$356*Compile_$478);
double Compile_$482 = Compile_$365 + Compile_$481;
double Compile_$501 = Power(Compile_$394,-2);
double Compile_$515 = -(Compile_$339*Compile_$354*Compile_$381*Compile_$395);
double Compile_$516 = Compile_$355 + Compile_$515;
double Compile_$492 = Compile_$438 + Compile_$447;
double Compile_$493 = -(a*Compile_$492);
double Compile_$486 = 2*Compile_$328*Compile_$403;
double Compile_$487 = Compile_$390 + Compile_$486;
double Compile_$498 = p*Compile_$337*Compile_$353*Compile_$448*Compile_$449*Compile_$476*Compile_$479*Compile_$482;
double Compile_$499 = Compile_$453 + Compile_$466;
double Compile_$505 = -1 + Compile_$355;
double Compile_$506 = 1/Compile_$505;
double Compile_$507 = Compile_$365*Compile_$506;
double Compile_$538 = Power(Compile_$406,-2);
double Compile_$549 = -(Compile_$339*Compile_$354*Compile_$401*Compile_$407);
double Compile_$550 = Compile_$355 + Compile_$549;
double Compile_$521 = Power(Compile_$339,2);
double Compile_$485 = Compile_$382*Compile_$399*Compile_$484;
double Compile_$488 = -(Compile_$402*Compile_$411*Compile_$487);
double Compile_$489 = Compile_$485 + Compile_$488;
double Compile_$358 = (Compile_$347*Compile_$356)/2.;
double Compile_$362 = Compile_$359*Compile_$361;
double Compile_$363 = Compile_$358 + Compile_$362;
double Compile_$364 = 2*Compile_$328*Compile_$363;
double Compile_$366 = p*Compile_$337*Compile_$353*Compile_$365;
double Compile_$367 = -(Compile_$289*Compile_$335*Compile_$337);
double Compile_$370 = (Compile_$347*Compile_$369)/2.;
double Compile_$371 = Compile_$367 + Compile_$370;
double Compile_$372 = Compile_$356*Compile_$371;
double Compile_$373 = Compile_$359*Compile_$361*Compile_$369;
double Compile_$374 = Compile_$366 + Compile_$372 + Compile_$373;
double Compile_$375 = (Compile_$328*Compile_$374)/2.;
double Compile_$392 = Compile_$384*Compile_$391;
double Compile_$393 = Compile_$383 + Compile_$392;
double Compile_$400 = Compile_$382*Compile_$393*Compile_$399;
double Compile_$404 = Compile_$391*Compile_$403;
double Compile_$405 = Compile_$383 + Compile_$404;
double Compile_$412 = -(Compile_$402*Compile_$405*Compile_$411);
double Compile_$413 = Compile_$400 + Compile_$412;
double Compile_$414 = Compile_$378*Compile_$413;
double Compile_$415 = Compile_$364 + Compile_$375 + Compile_$414;
double Compile_$416 = Compile_$357*Compile_$415;
double Compile_$417 = Compile_$329 + Compile_$416;
double Compile_$480 = Power(Compile_$356,-2);
double Compile_$588 = -(Compile_$339*Compile_$354);
double Compile_$589 = Compile_$355 + Compile_$588;
double Compile_$609 = Compile_$471/2.;
double Compile_$610 = Compile_$451 + Compile_$453 + Compile_$609;
double Compile_$579 = Compile_$360 + Compile_$477;
double Compile_$580 = 1/Compile_$579;
double Compile_$581 = Compile_$361 + Compile_$507;
double Compile_$582 = (Compile_$476*Compile_$580*Compile_$581)/2.;
double Compile_$583 = -(Compile_$339*Compile_$467*Compile_$468);
double Compile_$584 = Compile_$354*Compile_$473;
double Compile_$585 = Compile_$583 + Compile_$584;
double Compile_$586 = -1 + Compile_$360;
double Compile_$587 = 1/Compile_$586;
double Compile_$590 = 1/Compile_$589;
double Compile_$591 = Compile_$353*Compile_$356*Compile_$448*Compile_$589;
double Compile_$592 = Compile_$468*Compile_$521;
double Compile_$593 = Compile_$477 + Compile_$592;
double Compile_$594 = Compile_$353*Compile_$361*Compile_$448*Compile_$593;
double Compile_$595 = Compile_$365 + Compile_$591 + Compile_$594;
double Compile_$596 = (Compile_$585*Compile_$587*Compile_$590*Compile_$595)/2.;
double Compile_$597 = Compile_$582 + Compile_$596;
double Compile_$572 = 2*Compile_$445*Compile_$446;
double Compile_$496 = Power(Compile_$381,-2);
double Compile_$500 = -(Compile_$395*Compile_$397*Compile_$499);
double Compile_$502 = -(p*Compile_$359*Compile_$397*Compile_$452*Compile_$501);
double Compile_$503 = Compile_$396 + Compile_$477;
double Compile_$504 = 1/Compile_$503;
double Compile_$508 = Compile_$397 + Compile_$507;
double Compile_$509 = (Compile_$476*Compile_$504*Compile_$508)/2.;
double Compile_$510 = (Compile_$339*Compile_$354*Compile_$395*Compile_$471)/2.;
double Compile_$511 = -(Compile_$339*Compile_$381*Compile_$395*Compile_$467*Compile_$468);
double Compile_$512 = p*Compile_$339*Compile_$354*Compile_$381*Compile_$452*Compile_$501;
double Compile_$513 = Compile_$354*Compile_$381*Compile_$395*Compile_$473;
double Compile_$514 = Compile_$510 + Compile_$511 + Compile_$512 + Compile_$513;
double Compile_$517 = 1/Compile_$516;
double Compile_$518 = -1 + Compile_$396;
double Compile_$519 = 1/Compile_$518;
double Compile_$520 = Compile_$353*Compile_$356*Compile_$382*Compile_$394*Compile_$448*Compile_$516;
double Compile_$522 = Power(Compile_$381,2);
double Compile_$523 = Compile_$468*Compile_$501*Compile_$521*Compile_$522;
double Compile_$524 = Compile_$477 + Compile_$523;
double Compile_$525 = Compile_$353*Compile_$382*Compile_$394*Compile_$397*Compile_$448*Compile_$524;
double Compile_$526 = Compile_$365 + Compile_$520 + Compile_$525;
double Compile_$527 = (Compile_$514*Compile_$517*Compile_$519*Compile_$526)/2.;
double Compile_$528 = Compile_$509 + Compile_$527;
double Compile_$529 = -(Compile_$359*Compile_$395*Compile_$528);
double Compile_$530 = Compile_$498 + Compile_$500 + Compile_$502 + Compile_$529;
double Compile_$621 = -(Compile_$208*Compile_$445*Compile_$446);
double Compile_$622 = Compile_$493 + Compile_$572;
double Compile_$535 = Power(Compile_$401,-2);
double Compile_$537 = -(Compile_$407*Compile_$409*Compile_$499);
double Compile_$539 = -(p*Compile_$359*Compile_$409*Compile_$452*Compile_$538);
double Compile_$540 = Compile_$408 + Compile_$477;
double Compile_$541 = 1/Compile_$540;
double Compile_$542 = Compile_$409 + Compile_$507;
double Compile_$543 = (Compile_$476*Compile_$541*Compile_$542)/2.;
double Compile_$544 = (Compile_$339*Compile_$354*Compile_$407*Compile_$471)/2.;
double Compile_$545 = -(Compile_$339*Compile_$401*Compile_$407*Compile_$467*Compile_$468);
double Compile_$546 = p*Compile_$339*Compile_$354*Compile_$401*Compile_$452*Compile_$538;
double Compile_$547 = Compile_$354*Compile_$401*Compile_$407*Compile_$473;
double Compile_$548 = Compile_$544 + Compile_$545 + Compile_$546 + Compile_$547;
double Compile_$551 = 1/Compile_$550;
double Compile_$552 = -1 + Compile_$408;
double Compile_$553 = 1/Compile_$552;
double Compile_$554 = Compile_$353*Compile_$356*Compile_$402*Compile_$406*Compile_$448*Compile_$550;
double Compile_$555 = Power(Compile_$401,2);
double Compile_$556 = Compile_$468*Compile_$521*Compile_$538*Compile_$555;
double Compile_$557 = Compile_$477 + Compile_$556;
double Compile_$558 = Compile_$353*Compile_$402*Compile_$406*Compile_$409*Compile_$448*Compile_$557;
double Compile_$559 = Compile_$365 + Compile_$554 + Compile_$558;
double Compile_$560 = (Compile_$548*Compile_$551*Compile_$553*Compile_$559)/2.;
double Compile_$561 = Compile_$543 + Compile_$560;
double Compile_$562 = -(Compile_$359*Compile_$407*Compile_$561);
double Compile_$563 = Compile_$498 + Compile_$537 + Compile_$539 + Compile_$562;
return (Compile_$438 + Compile_$447 - (a*p*Compile_$337*Compile_$353*Compile_$378*Compile_$448*Compile_$449*Compile_$476*Compile_$479*Compile_$480*Compile_$482*Compile_$489)/2. + (a*Compile_$357*Compile_$378*(Compile_$382*Compile_$399*(Compile_$384*Compile_$445*Compile_$446 + Compile_$493) - Compile_$402*Compile_$411*(Compile_$403*Compile_$445*Compile_$446 + Compile_$493) - (Compile_$399*Compile_$471*Compile_$484*Compile_$496)/2. + Compile_$382*Compile_$484*Compile_$530 + (Compile_$411*Compile_$471*Compile_$487*Compile_$535)/2. - Compile_$402*Compile_$487*Compile_$563))/2.)/Compile_$417 - ((Compile_$387 + Compile_$388 + (a*Compile_$357*Compile_$378*Compile_$489)/2.)*(-(p*Compile_$337*Compile_$353*Compile_$415*Compile_$448*Compile_$449*Compile_$476*Compile_$479*Compile_$480*Compile_$482) + Compile_$572 + Compile_$357*(Compile_$363*Compile_$445*Compile_$446 + (Compile_$374*Compile_$445*Compile_$446)/4. + 2*Compile_$328*((Compile_$356*Compile_$471)/2. + (p*Compile_$337*Compile_$353*Compile_$448*Compile_$476*Compile_$479*Compile_$482)/2. + Compile_$361*Compile_$499 + Compile_$359*Compile_$597) + (Compile_$328*(-(p*Compile_$353*Compile_$365*Compile_$452) + p*Compile_$337*Compile_$365*Compile_$467 + Compile_$289*Power(Compile_$353,2)*(-Compile_$356 + Compile_$365)*Compile_$448*Compile_$449*Compile_$452*Compile_$476 + p*Compile_$337*Compile_$353*Compile_$371*Compile_$448*Compile_$449*Compile_$476*Compile_$479*Compile_$482 + Compile_$361*Compile_$369*Compile_$499 + Compile_$359*Compile_$369*Compile_$597 + Compile_$359*Compile_$361*Compile_$610 + Compile_$356*(-(Compile_$289*Compile_$337*Compile_$450) + Compile_$289*Compile_$335*Compile_$452 + (Compile_$369*Compile_$471)/2. + (Compile_$347*Compile_$610)/2.)))/2. + Compile_$378*(-0.5*(Compile_$393*Compile_$399*Compile_$471*Compile_$496) + Compile_$382*Compile_$393*Compile_$530 + (Compile_$405*Compile_$411*Compile_$471*Compile_$535)/2. - Compile_$402*Compile_$405*Compile_$563 + Compile_$382*Compile_$399*(Compile_$621 + Compile_$384*Compile_$622) - Compile_$402*Compile_$411*(Compile_$621 + Compile_$403*Compile_$622)))))/Power(Compile_$417,2);
}

double DomegaR_De(double a, double p, double e)
{
double Compile_$3 = Power(e,2);
double Compile_$6 = -1 + Compile_$3;
double Compile_$7 = Power(a,2);
double Compile_$4 = -Compile_$3;
double Compile_$8 = Power(Compile_$6,2);
double Compile_$42 = -e;
double Compile_$43 = 1 + Compile_$42;
double Compile_$44 = 1/Compile_$43;
double Compile_$45 = p*Compile_$44;
double Compile_$1 = 1 + e;
double Compile_$2 = 1/Compile_$1;
double Compile_$5 = 1 + Compile_$4;
double Compile_$9 = -4*Compile_$7*Compile_$8;
double Compile_$10 = -p;
double Compile_$11 = 3 + Compile_$10 + Compile_$3;
double Compile_$12 = Power(Compile_$11,2);
double Compile_$13 = p*Compile_$12;
double Compile_$14 = Compile_$13 + Compile_$9;
double Compile_$15 = 1/Compile_$14;
double Compile_$16 = 3*Compile_$3;
double Compile_$17 = 1 + p + Compile_$16;
double Compile_$18 = Compile_$17*Compile_$7;
double Compile_$19 = Power(p,-3);
double Compile_$20 = Power(a,6);
double Compile_$21 = Compile_$20*Compile_$8;
double Compile_$22 = -4*Compile_$3;
double Compile_$23 = -2 + p;
double Compile_$24 = Power(Compile_$23,2);
double Compile_$25 = Compile_$22 + Compile_$24;
double Compile_$26 = Power(p,2);
double Compile_$27 = Compile_$25*Compile_$26*Compile_$7;
double Compile_$28 = Power(a,4);
double Compile_$29 = 2 + p;
double Compile_$30 = Compile_$29*Compile_$3;
double Compile_$31 = -2 + p + Compile_$30;
double Compile_$32 = 2*p*Compile_$28*Compile_$31;
double Compile_$33 = Compile_$21 + Compile_$27 + Compile_$32;
double Compile_$34 = Compile_$19*Compile_$33;
double Compile_$35 = Sqrt(Compile_$34);
double Compile_$36 = -2*Compile_$35;
double Compile_$37 = -3 + p + Compile_$36 + Compile_$4;
double Compile_$38 = p*Compile_$37;
double Compile_$39 = Compile_$18 + Compile_$38;
double Compile_$40 = Compile_$15*Compile_$39*Compile_$6;
double Compile_$41 = 1 + Compile_$40;
double Compile_$47 = 1/Compile_$5;
double Compile_$48 = 1/Compile_$41;
double Compile_$62 = Power(Compile_$43,-2);
double Compile_$63 = p*Compile_$62;
double Compile_$64 = Power(Compile_$1,-2);
double Compile_$66 = 6*e*Compile_$7;
double Compile_$67 = -2*e;
double Compile_$68 = 4*e*Compile_$20*Compile_$6;
double Compile_$69 = -8*e*Compile_$26*Compile_$7;
double Compile_$70 = 4*e*p*Compile_$28*Compile_$29;
double Compile_$71 = Compile_$68 + Compile_$69 + Compile_$70;
double Compile_$72 = 1/Sqrt(Compile_$34);
double Compile_$73 = -(Compile_$19*Compile_$71*Compile_$72);
double Compile_$74 = Compile_$67 + Compile_$73;
double Compile_$75 = p*Compile_$74;
double Compile_$76 = Compile_$66 + Compile_$75;
double Compile_$77 = Compile_$15*Compile_$6*Compile_$76;
double Compile_$78 = -16*e*Compile_$6*Compile_$7;
double Compile_$79 = 4*e*p*Compile_$11;
double Compile_$80 = Compile_$78 + Compile_$79;
double Compile_$81 = Power(Compile_$14,-2);
double Compile_$82 = -(Compile_$39*Compile_$6*Compile_$80*Compile_$81);
double Compile_$83 = 2*e*Compile_$15*Compile_$39;
double Compile_$84 = Compile_$77 + Compile_$82 + Compile_$83;
double Compile_$85 = Power(Compile_$41,-2);
double Compile_$87 = Power(Compile_$5,-2);
double Compile_$50 = -(p*Compile_$44);
double Compile_$51 = -(p*Compile_$2);
double Compile_$52 = 2*p*Compile_$47*Compile_$48;
double Compile_$53 = Compile_$50 + Compile_$51 + Compile_$52;
double Compile_$54 = Power(Compile_$53,2);
double Compile_$46 = p*Compile_$2;
double Compile_$49 = -2*p*Compile_$47*Compile_$48;
double Compile_$55 = Sqrt(Compile_$54);
double Compile_$56 = -Compile_$55;
double Compile_$57 = Compile_$45 + Compile_$46 + Compile_$49 + Compile_$56;
double Compile_$58 = Compile_$57/2.;
double Compile_$59 = Compile_$45 + Compile_$58;
double Compile_$104 = 1/p;
double Compile_$105 = Compile_$45 + Compile_$51;
double Compile_$106 = Compile_$50 + Compile_$51 + Compile_$52 + Compile_$55;
double Compile_$107 = 1/Compile_$59;
double Compile_$108 = (Compile_$1*Compile_$104*Compile_$105*Compile_$106*Compile_$107)/2.;
double Compile_$109 = EllipticK(Compile_$108);
double Compile_$110 = 1/Compile_$109;
double Compile_$111 = -(Compile_$104*Compile_$41*Compile_$5);
double Compile_$112 = 1 + Compile_$111;
double Compile_$113 = Sqrt(Compile_$112);
double Compile_$116 = Compile_$46 + Compile_$58;
double Compile_$125 = Compile_$106/2.;
double Compile_$126 = Compile_$125 + Compile_$45 + Compile_$46;
double Compile_$117 = Compile_$105*Compile_$107;
double Compile_$118 = EllipticPi(Compile_$117,Compile_$108);
double Compile_$134 = -Compile_$7;
double Compile_$135 = 1 + Compile_$134;
double Compile_$137 = Sqrt(Compile_$135);
double Compile_$114 = 4*Compile_$113;
double Compile_$138 = -Compile_$137;
double Compile_$152 = -1 + Compile_$138 + Compile_$46;
double Compile_$153 = 1/Compile_$152;
double Compile_$139 = -1 + Compile_$125 + Compile_$138;
double Compile_$141 = -2*Compile_$113*Compile_$7;
double Compile_$143 = Compile_$15*Compile_$39;
double Compile_$144 = Sqrt(Compile_$143);
double Compile_$145 = p*Compile_$144;
double Compile_$146 = a*Compile_$113;
double Compile_$147 = Compile_$145 + Compile_$146;
double Compile_$148 = -(a*Compile_$147);
double Compile_$149 = Compile_$114 + Compile_$148;
double Compile_$164 = -1 + Compile_$137 + Compile_$46;
double Compile_$165 = 1/Compile_$164;
double Compile_$159 = -1 + Compile_$125 + Compile_$137;
double Compile_$60 = Compile_$2*Compile_$41*Compile_$5*Compile_$59;
double Compile_$65 = -(p*Compile_$64);
double Compile_$86 = 2*p*Compile_$47*Compile_$84*Compile_$85;
double Compile_$88 = -4*e*p*Compile_$48*Compile_$87;
double Compile_$89 = -(p*Compile_$62);
double Compile_$90 = p*Compile_$64;
double Compile_$91 = -2*p*Compile_$47*Compile_$84*Compile_$85;
double Compile_$92 = 4*e*p*Compile_$48*Compile_$87;
double Compile_$93 = Compile_$89 + Compile_$90 + Compile_$91 + Compile_$92;
double Compile_$94 = 1/Sqrt(Compile_$54);
double Compile_$95 = -(Compile_$53*Compile_$93*Compile_$94);
double Compile_$96 = Compile_$63 + Compile_$65 + Compile_$86 + Compile_$88 + Compile_$95;
double Compile_$97 = Compile_$96/2.;
double Compile_$98 = Compile_$63 + Compile_$97;
double Compile_$122 = EllipticE(Compile_$108);
double Compile_$190 = -0.5*(Compile_$1*Compile_$104*Compile_$105*Compile_$106*Compile_$107);
double Compile_$191 = 1 + Compile_$190;
double Compile_$115 = (Compile_$106*Compile_$109)/2.;
double Compile_$119 = Compile_$116*Compile_$118;
double Compile_$120 = Compile_$115 + Compile_$119;
double Compile_$121 = 2*Compile_$113*Compile_$120;
double Compile_$123 = p*Compile_$122*Compile_$2*Compile_$59;
double Compile_$124 = -(Compile_$2*Compile_$26*Compile_$44);
double Compile_$127 = (Compile_$106*Compile_$126)/2.;
double Compile_$128 = Compile_$124 + Compile_$127;
double Compile_$129 = Compile_$109*Compile_$128;
double Compile_$130 = Compile_$116*Compile_$118*Compile_$126;
double Compile_$132 = Compile_$123 + Compile_$129 + Compile_$130;
double Compile_$133 = (Compile_$113*Compile_$132)/2.;
double Compile_$136 = 1/Sqrt(Compile_$135);
double Compile_$140 = 1/Compile_$139;
double Compile_$142 = 1 + Compile_$137;
double Compile_$150 = Compile_$142*Compile_$149;
double Compile_$151 = Compile_$141 + Compile_$150;
double Compile_$154 = Compile_$105*Compile_$107*Compile_$139*Compile_$153;
double Compile_$155 = EllipticPi(Compile_$154,Compile_$108);
double Compile_$156 = -(Compile_$116*Compile_$153*Compile_$155);
double Compile_$157 = Compile_$109 + Compile_$156;
double Compile_$158 = Compile_$140*Compile_$151*Compile_$157;
double Compile_$160 = 1/Compile_$159;
double Compile_$161 = 1 + Compile_$138;
double Compile_$162 = Compile_$149*Compile_$161;
double Compile_$163 = Compile_$141 + Compile_$162;
double Compile_$166 = Compile_$105*Compile_$107*Compile_$159*Compile_$165;
double Compile_$167 = EllipticPi(Compile_$166,Compile_$108);
double Compile_$168 = -(Compile_$116*Compile_$165*Compile_$167);
double Compile_$169 = Compile_$109 + Compile_$168;
double Compile_$170 = -(Compile_$160*Compile_$163*Compile_$169);
double Compile_$171 = Compile_$158 + Compile_$170;
double Compile_$172 = Compile_$136*Compile_$171;
double Compile_$173 = Compile_$121 + Compile_$133 + Compile_$172;
double Compile_$174 = Compile_$110*Compile_$173;
double Compile_$175 = Compile_$114 + Compile_$174;
double Compile_$176 = 1/Compile_$175;
double Compile_$180 = Sqrt(Compile_$60);
double Compile_$178 = 1/Compile_$105;
double Compile_$179 = 1/Compile_$106;
double Compile_$181 = Power(Compile_$59,-2);
double Compile_$182 = -0.5*(Compile_$1*Compile_$104*Compile_$105*Compile_$106*Compile_$181*Compile_$98);
double Compile_$183 = Compile_$53*Compile_$93*Compile_$94;
double Compile_$184 = Compile_$183 + Compile_$89 + Compile_$90 + Compile_$91 + Compile_$92;
double Compile_$185 = (Compile_$1*Compile_$104*Compile_$105*Compile_$107*Compile_$184)/2.;
double Compile_$186 = Compile_$63 + Compile_$90;
double Compile_$187 = (Compile_$1*Compile_$104*Compile_$106*Compile_$107*Compile_$186)/2.;
double Compile_$188 = (Compile_$104*Compile_$105*Compile_$106*Compile_$107)/2.;
double Compile_$189 = Compile_$182 + Compile_$185 + Compile_$187 + Compile_$188;
double Compile_$192 = 1/Compile_$191;
double Compile_$193 = Power(Compile_$109,-2);
double Compile_$194 = -(Compile_$109*Compile_$191);
double Compile_$195 = Compile_$122 + Compile_$194;
double Compile_$198 = -(Compile_$104*Compile_$5*Compile_$84);
double Compile_$199 = 2*e*Compile_$104*Compile_$41;
double Compile_$200 = Compile_$198 + Compile_$199;
double Compile_$202 = 1/Sqrt(Compile_$112);
double Compile_$227 = -(Compile_$105*Compile_$107);
double Compile_$228 = Compile_$108 + Compile_$227;
double Compile_$254 = Compile_$184/2.;
double Compile_$255 = Compile_$254 + Compile_$63 + Compile_$65;
double Compile_$213 = Compile_$65 + Compile_$97;
double Compile_$215 = Compile_$117 + Compile_$190;
double Compile_$216 = 1/Compile_$215;
double Compile_$217 = -1 + Compile_$108;
double Compile_$218 = 1/Compile_$217;
double Compile_$219 = Compile_$122*Compile_$218;
double Compile_$220 = Compile_$118 + Compile_$219;
double Compile_$221 = (Compile_$189*Compile_$216*Compile_$220)/2.;
double Compile_$222 = -(Compile_$105*Compile_$181*Compile_$98);
double Compile_$223 = Compile_$107*Compile_$186;
double Compile_$224 = Compile_$222 + Compile_$223;
double Compile_$225 = -1 + Compile_$117;
double Compile_$226 = 1/Compile_$225;
double Compile_$229 = 1/Compile_$228;
double Compile_$230 = Compile_$109*Compile_$178*Compile_$228*Compile_$59;
double Compile_$231 = Power(Compile_$105,2);
double Compile_$232 = Compile_$181*Compile_$231;
double Compile_$233 = Compile_$190 + Compile_$232;
double Compile_$234 = Compile_$118*Compile_$178*Compile_$233*Compile_$59;
double Compile_$235 = Compile_$122 + Compile_$230 + Compile_$234;
double Compile_$236 = (Compile_$224*Compile_$226*Compile_$229*Compile_$235)/2.;
double Compile_$237 = Compile_$221 + Compile_$236;
double Compile_$205 = 2*Compile_$200*Compile_$202;
double Compile_$297 = Power(Compile_$152,-2);
double Compile_$308 = -(Compile_$105*Compile_$107*Compile_$139*Compile_$153);
double Compile_$321 = Compile_$108 + Compile_$308;
double Compile_$266 = -(Compile_$200*Compile_$202*Compile_$7);
double Compile_$267 = 1/Sqrt(Compile_$143);
double Compile_$268 = Compile_$15*Compile_$76;
double Compile_$269 = -(Compile_$39*Compile_$80*Compile_$81);
double Compile_$270 = Compile_$268 + Compile_$269;
double Compile_$271 = (p*Compile_$267*Compile_$270)/2.;
double Compile_$272 = (a*Compile_$200*Compile_$202)/2.;
double Compile_$273 = Compile_$271 + Compile_$272;
double Compile_$274 = -(a*Compile_$273);
double Compile_$283 = Compile_$205 + Compile_$274;
double Compile_$295 = p*Compile_$178*Compile_$179*Compile_$189*Compile_$192*Compile_$195*Compile_$2*Compile_$59;
double Compile_$570 = Power(Compile_$164,-2);
double Compile_$601 = -(Compile_$105*Compile_$107*Compile_$159*Compile_$165);
double Compile_$602 = Compile_$108 + Compile_$601;
return -0.5*(p*Pi*Compile_$176*Compile_$178*Compile_$179*Compile_$180*Compile_$189*Compile_$192*Compile_$193*Compile_$195*Compile_$2*Compile_$59) + (Pi*Compile_$110*Compile_$176*(-2*e*Compile_$2*Compile_$41*Compile_$59 - Compile_$41*Compile_$5*Compile_$59*Compile_$64 + Compile_$2*Compile_$5*Compile_$59*Compile_$84 + Compile_$2*Compile_$41*Compile_$5*Compile_$98))/(4.*Sqrt(Compile_$60)) - (Pi*Compile_$110*Compile_$180*(Compile_$205 - p*Compile_$173*Compile_$178*Compile_$179*Compile_$189*Compile_$192*Compile_$193*Compile_$195*Compile_$2*Compile_$59 + Compile_$110*(Compile_$120*Compile_$200*Compile_$202 + (Compile_$132*Compile_$200*Compile_$202)/4. + 2*Compile_$113*((Compile_$109*Compile_$184)/2. + Compile_$118*Compile_$213 + Compile_$116*Compile_$237 + (p*Compile_$178*Compile_$189*Compile_$192*Compile_$195*Compile_$2*Compile_$59)/2.) + (Compile_$113*(Compile_$118*Compile_$126*Compile_$213 + Compile_$116*Compile_$126*Compile_$237 + Compile_$116*Compile_$118*Compile_$255 + p*Compile_$128*Compile_$178*Compile_$179*Compile_$189*Compile_$192*Compile_$195*Compile_$2*Compile_$59 - p*Compile_$122*Compile_$59*Compile_$64 + (-Compile_$109 + Compile_$122)*Compile_$178*Compile_$179*Compile_$189*Compile_$26*Power(Compile_$59,2)*Compile_$64 + Compile_$109*((Compile_$126*Compile_$184)/2. + (Compile_$106*Compile_$255)/2. - Compile_$2*Compile_$26*Compile_$62 + Compile_$26*Compile_$44*Compile_$64) + p*Compile_$122*Compile_$2*Compile_$98))/2. + Compile_$136*(-0.5*(Compile_$151*Compile_$157*Compile_$184)/Power(Compile_$139,2) + (Compile_$163*Compile_$169*Compile_$184)/(2.*Power(Compile_$159,2)) + Compile_$140*Compile_$157*(Compile_$266 + Compile_$142*Compile_$283) - Compile_$160*Compile_$169*(Compile_$266 + Compile_$161*Compile_$283) + Compile_$140*Compile_$151*(-(Compile_$153*Compile_$155*Compile_$213) + Compile_$295 - p*Compile_$116*Compile_$155*Compile_$297*Compile_$64 - Compile_$116*Compile_$153*((Compile_$189*(Compile_$155 + Compile_$219))/(2.*(Compile_$154 + Compile_$190)) + ((Compile_$122 + Compile_$140*Compile_$152*Compile_$155*Compile_$178*(Compile_$190 + Power(Compile_$139,2)*Compile_$181*Compile_$231*Compile_$297)*Compile_$59 + Compile_$109*Compile_$140*Compile_$152*Compile_$178*Compile_$321*Compile_$59)*((Compile_$105*Compile_$107*Compile_$153*Compile_$184)/2. + Compile_$107*Compile_$139*Compile_$153*Compile_$186 + p*Compile_$105*Compile_$107*Compile_$139*Compile_$297*Compile_$64 - Compile_$105*Compile_$139*Compile_$153*Compile_$181*Compile_$98))/(2.*(-1 + Compile_$154)*Compile_$321))) - Compile_$160*Compile_$163*(-(Compile_$165*Compile_$167*Compile_$213) + Compile_$295 - p*Compile_$116*Compile_$167*Compile_$570*Compile_$64 - Compile_$116*Compile_$165*((Compile_$189*(Compile_$167 + Compile_$219))/(2.*(Compile_$166 + Compile_$190)) + ((Compile_$122 + Compile_$160*Compile_$164*Compile_$167*Compile_$178*(Compile_$190 + Power(Compile_$159,2)*Compile_$181*Compile_$231*Compile_$570)*Compile_$59 + Compile_$109*Compile_$160*Compile_$164*Compile_$178*Compile_$59*Compile_$602)*((Compile_$105*Compile_$107*Compile_$165*Compile_$184)/2. + Compile_$107*Compile_$159*Compile_$165*Compile_$186 + p*Compile_$105*Compile_$107*Compile_$159*Compile_$570*Compile_$64 - Compile_$105*Compile_$159*Compile_$165*Compile_$181*Compile_$98))/(2.*(-1 + Compile_$166)*Compile_$602)))))))/(2.*Power(Compile_$175,2));
}

double DomegaR_Dp(double a, double p, double e)
{
double Compile_$3 = Power(e,2);
double Compile_$6 = -1 + Compile_$3;
double Compile_$7 = Power(a,2);
double Compile_$4 = -Compile_$3;
double Compile_$8 = Power(Compile_$6,2);
double Compile_$42 = -e;
double Compile_$43 = 1 + Compile_$42;
double Compile_$44 = 1/Compile_$43;
double Compile_$45 = p*Compile_$44;
double Compile_$1 = 1 + e;
double Compile_$2 = 1/Compile_$1;
double Compile_$5 = 1 + Compile_$4;
double Compile_$9 = -4*Compile_$7*Compile_$8;
double Compile_$10 = -p;
double Compile_$11 = 3 + Compile_$10 + Compile_$3;
double Compile_$12 = Power(Compile_$11,2);
double Compile_$13 = p*Compile_$12;
double Compile_$14 = Compile_$13 + Compile_$9;
double Compile_$15 = 1/Compile_$14;
double Compile_$16 = 3*Compile_$3;
double Compile_$17 = 1 + p + Compile_$16;
double Compile_$18 = Compile_$17*Compile_$7;
double Compile_$19 = Power(p,-3);
double Compile_$20 = Power(a,6);
double Compile_$21 = Compile_$20*Compile_$8;
double Compile_$22 = -4*Compile_$3;
double Compile_$23 = -2 + p;
double Compile_$24 = Power(Compile_$23,2);
double Compile_$25 = Compile_$22 + Compile_$24;
double Compile_$26 = Power(p,2);
double Compile_$27 = Compile_$25*Compile_$26*Compile_$7;
double Compile_$28 = Power(a,4);
double Compile_$29 = 2 + p;
double Compile_$30 = Compile_$29*Compile_$3;
double Compile_$31 = -2 + p + Compile_$30;
double Compile_$32 = 2*p*Compile_$28*Compile_$31;
double Compile_$33 = Compile_$21 + Compile_$27 + Compile_$32;
double Compile_$34 = Compile_$19*Compile_$33;
double Compile_$35 = Sqrt(Compile_$34);
double Compile_$36 = -2*Compile_$35;
double Compile_$37 = -3 + p + Compile_$36 + Compile_$4;
double Compile_$38 = p*Compile_$37;
double Compile_$39 = Compile_$18 + Compile_$38;
double Compile_$40 = Compile_$15*Compile_$39*Compile_$6;
double Compile_$41 = 1 + Compile_$40;
double Compile_$47 = 1/Compile_$5;
double Compile_$48 = 1/Compile_$41;
double Compile_$46 = p*Compile_$2;
double Compile_$49 = -2*p*Compile_$47*Compile_$48;
double Compile_$50 = -(p*Compile_$44);
double Compile_$51 = -(p*Compile_$2);
double Compile_$52 = 2*p*Compile_$47*Compile_$48;
double Compile_$53 = Compile_$50 + Compile_$51 + Compile_$52;
double Compile_$54 = Power(Compile_$53,2);
double Compile_$55 = Sqrt(Compile_$54);
double Compile_$56 = -Compile_$55;
double Compile_$57 = Compile_$45 + Compile_$46 + Compile_$49 + Compile_$56;
double Compile_$58 = Compile_$57/2.;
double Compile_$59 = Compile_$45 + Compile_$58;
double Compile_$62 = -2*p*Compile_$11;
double Compile_$63 = Compile_$12 + Compile_$62;
double Compile_$64 = Power(Compile_$14,-2);
double Compile_$65 = -(Compile_$39*Compile_$6*Compile_$63*Compile_$64);
double Compile_$66 = 1/Sqrt(Compile_$34);
double Compile_$67 = 1 + Compile_$3;
double Compile_$68 = 2*p*Compile_$28*Compile_$67;
double Compile_$69 = 2*p*Compile_$25*Compile_$7;
double Compile_$70 = 2*Compile_$23*Compile_$26*Compile_$7;
double Compile_$71 = 2*Compile_$28*Compile_$31;
double Compile_$72 = Compile_$68 + Compile_$69 + Compile_$70 + Compile_$71;
double Compile_$73 = Compile_$19*Compile_$72;
double Compile_$74 = Power(p,-4);
double Compile_$75 = -3*Compile_$33*Compile_$74;
double Compile_$76 = Compile_$73 + Compile_$75;
double Compile_$77 = -(Compile_$66*Compile_$76);
double Compile_$78 = 1 + Compile_$77;
double Compile_$79 = p*Compile_$78;
double Compile_$80 = -3 + p + Compile_$36 + Compile_$4 + Compile_$7 + Compile_$79;
double Compile_$81 = Compile_$15*Compile_$6*Compile_$80;
double Compile_$82 = Compile_$65 + Compile_$81;
double Compile_$85 = Power(Compile_$41,-2);
double Compile_$99 = 1/p;
double Compile_$100 = Compile_$45 + Compile_$51;
double Compile_$101 = Compile_$50 + Compile_$51 + Compile_$52 + Compile_$55;
double Compile_$102 = 1/Compile_$59;
double Compile_$103 = (Compile_$1*Compile_$100*Compile_$101*Compile_$102*Compile_$99)/2.;
double Compile_$104 = EllipticK(Compile_$103);
double Compile_$105 = 1/Compile_$104;
double Compile_$106 = -(Compile_$41*Compile_$5*Compile_$99);
double Compile_$107 = 1 + Compile_$106;
double Compile_$108 = Sqrt(Compile_$107);
double Compile_$111 = Compile_$46 + Compile_$58;
double Compile_$120 = Compile_$101/2.;
double Compile_$121 = Compile_$120 + Compile_$45 + Compile_$46;
double Compile_$112 = Compile_$100*Compile_$102;
double Compile_$113 = EllipticPi(Compile_$112,Compile_$103);
double Compile_$128 = -Compile_$7;
double Compile_$129 = 1 + Compile_$128;
double Compile_$132 = Sqrt(Compile_$129);
double Compile_$109 = 4*Compile_$108;
double Compile_$133 = -Compile_$132;
double Compile_$147 = -1 + Compile_$133 + Compile_$46;
double Compile_$148 = 1/Compile_$147;
double Compile_$134 = -1 + Compile_$120 + Compile_$133;
double Compile_$136 = -2*Compile_$108*Compile_$7;
double Compile_$138 = Compile_$15*Compile_$39;
double Compile_$139 = Sqrt(Compile_$138);
double Compile_$140 = p*Compile_$139;
double Compile_$141 = a*Compile_$108;
double Compile_$142 = Compile_$140 + Compile_$141;
double Compile_$143 = -(a*Compile_$142);
double Compile_$144 = Compile_$109 + Compile_$143;
double Compile_$159 = -1 + Compile_$132 + Compile_$46;
double Compile_$160 = 1/Compile_$159;
double Compile_$154 = -1 + Compile_$120 + Compile_$132;
double Compile_$60 = Compile_$2*Compile_$41*Compile_$5*Compile_$59;
double Compile_$89 = -Compile_$2;
double Compile_$88 = -Compile_$44;
double Compile_$90 = 2*Compile_$47*Compile_$48;
double Compile_$91 = -2*p*Compile_$47*Compile_$82*Compile_$85;
double Compile_$87 = 1/Sqrt(Compile_$54);
double Compile_$92 = Compile_$88 + Compile_$89 + Compile_$90 + Compile_$91;
double Compile_$84 = -2*Compile_$47*Compile_$48;
double Compile_$86 = 2*p*Compile_$47*Compile_$82*Compile_$85;
double Compile_$93 = -(Compile_$53*Compile_$87*Compile_$92);
double Compile_$94 = Compile_$2 + Compile_$44 + Compile_$84 + Compile_$86 + Compile_$93;
double Compile_$95 = Compile_$94/2.;
double Compile_$96 = Compile_$44 + Compile_$95;
double Compile_$117 = EllipticE(Compile_$103);
double Compile_$176 = -0.5*(Compile_$1*Compile_$100*Compile_$101*Compile_$102*Compile_$99);
double Compile_$177 = 1 + Compile_$176;
double Compile_$110 = (Compile_$101*Compile_$104)/2.;
double Compile_$114 = Compile_$111*Compile_$113;
double Compile_$115 = Compile_$110 + Compile_$114;
double Compile_$116 = 2*Compile_$108*Compile_$115;
double Compile_$118 = p*Compile_$117*Compile_$2*Compile_$59;
double Compile_$119 = -(Compile_$2*Compile_$26*Compile_$44);
double Compile_$122 = (Compile_$101*Compile_$121)/2.;
double Compile_$123 = Compile_$119 + Compile_$122;
double Compile_$124 = Compile_$104*Compile_$123;
double Compile_$125 = Compile_$111*Compile_$113*Compile_$121;
double Compile_$126 = Compile_$118 + Compile_$124 + Compile_$125;
double Compile_$127 = (Compile_$108*Compile_$126)/2.;
double Compile_$130 = 1/Sqrt(Compile_$129);
double Compile_$135 = 1/Compile_$134;
double Compile_$137 = 1 + Compile_$132;
double Compile_$145 = Compile_$137*Compile_$144;
double Compile_$146 = Compile_$136 + Compile_$145;
double Compile_$149 = Compile_$100*Compile_$102*Compile_$134*Compile_$148;
double Compile_$150 = EllipticPi(Compile_$149,Compile_$103);
double Compile_$151 = -(Compile_$111*Compile_$148*Compile_$150);
double Compile_$152 = Compile_$104 + Compile_$151;
double Compile_$153 = Compile_$135*Compile_$146*Compile_$152;
double Compile_$155 = 1/Compile_$154;
double Compile_$156 = 1 + Compile_$133;
double Compile_$157 = Compile_$144*Compile_$156;
double Compile_$158 = Compile_$136 + Compile_$157;
double Compile_$161 = Compile_$100*Compile_$102*Compile_$154*Compile_$160;
double Compile_$162 = EllipticPi(Compile_$161,Compile_$103);
double Compile_$163 = -(Compile_$111*Compile_$160*Compile_$162);
double Compile_$164 = Compile_$104 + Compile_$163;
double Compile_$165 = -(Compile_$155*Compile_$158*Compile_$164);
double Compile_$166 = Compile_$153 + Compile_$165;
double Compile_$167 = Compile_$130*Compile_$166;
double Compile_$168 = Compile_$116 + Compile_$127 + Compile_$167;
double Compile_$169 = Compile_$105*Compile_$168;
double Compile_$170 = Compile_$109 + Compile_$169;
double Compile_$171 = 1/Compile_$170;
double Compile_$175 = Sqrt(Compile_$60);
double Compile_$181 = Power(p,-2);
double Compile_$173 = 1/Compile_$100;
double Compile_$174 = 1/Compile_$101;
double Compile_$178 = 1/Compile_$177;
double Compile_$179 = Compile_$44 + Compile_$89;
double Compile_$180 = (Compile_$1*Compile_$101*Compile_$102*Compile_$179*Compile_$99)/2.;
double Compile_$182 = -0.5*(Compile_$1*Compile_$100*Compile_$101*Compile_$102*Compile_$181);
double Compile_$183 = Compile_$53*Compile_$87*Compile_$92;
double Compile_$184 = Compile_$183 + Compile_$88 + Compile_$89 + Compile_$90 + Compile_$91;
double Compile_$185 = (Compile_$1*Compile_$100*Compile_$102*Compile_$184*Compile_$99)/2.;
double Compile_$186 = Power(Compile_$59,-2);
double Compile_$187 = -0.5*(Compile_$1*Compile_$100*Compile_$101*Compile_$186*Compile_$96*Compile_$99);
double Compile_$188 = Compile_$180 + Compile_$182 + Compile_$185 + Compile_$187;
double Compile_$189 = Power(Compile_$104,-2);
double Compile_$190 = -(Compile_$104*Compile_$177);
double Compile_$191 = Compile_$117 + Compile_$190;
double Compile_$194 = 1/Sqrt(Compile_$107);
double Compile_$195 = Compile_$181*Compile_$41*Compile_$5;
double Compile_$196 = -(Compile_$5*Compile_$82*Compile_$99);
double Compile_$197 = Compile_$195 + Compile_$196;
double Compile_$220 = -(Compile_$100*Compile_$102);
double Compile_$221 = Compile_$103 + Compile_$220;
double Compile_$209 = Compile_$2 + Compile_$95;
double Compile_$251 = Compile_$184/2.;
double Compile_$252 = Compile_$2 + Compile_$251 + Compile_$44;
double Compile_$211 = Compile_$112 + Compile_$176;
double Compile_$212 = 1/Compile_$211;
double Compile_$213 = -1 + Compile_$103;
double Compile_$214 = 1/Compile_$213;
double Compile_$215 = Compile_$117*Compile_$214;
double Compile_$216 = Compile_$113 + Compile_$215;
double Compile_$217 = (Compile_$188*Compile_$212*Compile_$216)/2.;
double Compile_$218 = -1 + Compile_$112;
double Compile_$219 = 1/Compile_$218;
double Compile_$222 = 1/Compile_$221;
double Compile_$223 = Compile_$102*Compile_$179;
double Compile_$224 = -(Compile_$100*Compile_$186*Compile_$96);
double Compile_$225 = Compile_$223 + Compile_$224;
double Compile_$226 = Compile_$104*Compile_$173*Compile_$221*Compile_$59;
double Compile_$227 = Power(Compile_$100,2);
double Compile_$228 = Compile_$186*Compile_$227;
double Compile_$229 = Compile_$176 + Compile_$228;
double Compile_$230 = Compile_$113*Compile_$173*Compile_$229*Compile_$59;
double Compile_$231 = Compile_$117 + Compile_$226 + Compile_$230;
double Compile_$232 = (Compile_$219*Compile_$222*Compile_$225*Compile_$231)/2.;
double Compile_$233 = Compile_$217 + Compile_$232;
double Compile_$198 = 2*Compile_$194*Compile_$197;
double Compile_$292 = Power(Compile_$147,-2);
double Compile_$299 = -(Compile_$100*Compile_$102*Compile_$134*Compile_$148);
double Compile_$300 = Compile_$103 + Compile_$299;
double Compile_$264 = -(Compile_$194*Compile_$197*Compile_$7);
double Compile_$265 = 1/Sqrt(Compile_$138);
double Compile_$266 = -(Compile_$39*Compile_$63*Compile_$64);
double Compile_$267 = Compile_$15*Compile_$80;
double Compile_$268 = Compile_$266 + Compile_$267;
double Compile_$269 = (p*Compile_$265*Compile_$268)/2.;
double Compile_$270 = (a*Compile_$194*Compile_$197)/2.;
double Compile_$271 = Compile_$139 + Compile_$269 + Compile_$270;
double Compile_$272 = -(a*Compile_$271);
double Compile_$273 = Compile_$198 + Compile_$272;
double Compile_$291 = p*Compile_$173*Compile_$174*Compile_$178*Compile_$188*Compile_$191*Compile_$2*Compile_$59;
double Compile_$565 = Power(Compile_$159,-2);
double Compile_$573 = -(Compile_$100*Compile_$102*Compile_$154*Compile_$160);
double Compile_$574 = Compile_$103 + Compile_$573;
return -0.5*(p*Pi*Compile_$171*Compile_$173*Compile_$174*Compile_$175*Compile_$178*Compile_$188*Compile_$189*Compile_$191*Compile_$2*Compile_$59) + (Pi*Compile_$105*Compile_$171*(Compile_$2*Compile_$5*Compile_$59*Compile_$82 + Compile_$2*Compile_$41*Compile_$5*Compile_$96))/(4.*Sqrt(Compile_$60)) - (Pi*Compile_$105*Compile_$175*(Compile_$198 - p*Compile_$168*Compile_$173*Compile_$174*Compile_$178*Compile_$188*Compile_$189*Compile_$191*Compile_$2*Compile_$59 + Compile_$105*(Compile_$115*Compile_$194*Compile_$197 + (Compile_$126*Compile_$194*Compile_$197)/4. + 2*Compile_$108*((Compile_$104*Compile_$184)/2. + Compile_$113*Compile_$209 + Compile_$111*Compile_$233 + (p*Compile_$173*Compile_$178*Compile_$188*Compile_$191*Compile_$2*Compile_$59)/2.) + (Compile_$108*(Compile_$113*Compile_$121*Compile_$209 + Compile_$111*Compile_$121*Compile_$233 + Compile_$111*Compile_$113*Compile_$252 + Compile_$104*((Compile_$121*Compile_$184)/2. + (Compile_$101*Compile_$252)/2. - 2*p*Compile_$2*Compile_$44) + Compile_$117*Compile_$2*Compile_$59 + p*Compile_$123*Compile_$173*Compile_$174*Compile_$178*Compile_$188*Compile_$191*Compile_$2*Compile_$59 + ((-Compile_$104 + Compile_$117)*Compile_$173*Compile_$174*Compile_$188*Compile_$26*Power(Compile_$59,2))/Power(Compile_$1,2) + p*Compile_$117*Compile_$2*Compile_$96))/2. + Compile_$130*(-0.5*(Compile_$146*Compile_$152*Compile_$184)/Power(Compile_$134,2) + (Compile_$158*Compile_$164*Compile_$184)/(2.*Power(Compile_$154,2)) + Compile_$135*Compile_$152*(Compile_$264 + Compile_$137*Compile_$273) - Compile_$155*Compile_$164*(Compile_$264 + Compile_$156*Compile_$273) + Compile_$135*Compile_$146*(-(Compile_$148*Compile_$150*Compile_$209) + Compile_$291 + Compile_$111*Compile_$150*Compile_$2*Compile_$292 - Compile_$111*Compile_$148*((Compile_$188*(Compile_$150 + Compile_$215))/(2.*(Compile_$149 + Compile_$176)) + ((Compile_$117 + Compile_$135*Compile_$147*Compile_$150*Compile_$173*(Compile_$176 + Power(Compile_$134,2)*Compile_$186*Compile_$227*Compile_$292)*Compile_$59 + Compile_$104*Compile_$135*Compile_$147*Compile_$173*Compile_$300*Compile_$59)*(Compile_$102*Compile_$134*Compile_$148*Compile_$179 + (Compile_$100*Compile_$102*Compile_$148*Compile_$184)/2. - Compile_$100*Compile_$102*Compile_$134*Compile_$2*Compile_$292 - Compile_$100*Compile_$134*Compile_$148*Compile_$186*Compile_$96))/(2.*(-1 + Compile_$149)*Compile_$300))) - Compile_$155*Compile_$158*(-(Compile_$160*Compile_$162*Compile_$209) + Compile_$291 + Compile_$111*Compile_$162*Compile_$2*Compile_$565 - Compile_$111*Compile_$160*((Compile_$188*(Compile_$162 + Compile_$215))/(2.*(Compile_$161 + Compile_$176)) + ((Compile_$117 + Compile_$155*Compile_$159*Compile_$162*Compile_$173*(Compile_$176 + Power(Compile_$154,2)*Compile_$186*Compile_$227*Compile_$565)*Compile_$59 + Compile_$104*Compile_$155*Compile_$159*Compile_$173*Compile_$574*Compile_$59)*(Compile_$102*Compile_$154*Compile_$160*Compile_$179 + (Compile_$100*Compile_$102*Compile_$160*Compile_$184)/2. - Compile_$100*Compile_$102*Compile_$154*Compile_$2*Compile_$565 - Compile_$100*Compile_$154*Compile_$160*Compile_$186*Compile_$96))/(2.*(-1 + Compile_$161)*Compile_$574)))))))/(2.*Power(Compile_$170,2));
}

void KerrEquatorialFrequencyDerivative(double *omegaPhi_dp, double *omegaPhi_de, double *omegaR_dp, double *omegaR_de, double a, double p, double e)
{
    *omegaPhi_dp = DomegaPhi_Dp(a, p, e);
    *omegaPhi_de = DomegaPhi_De(a, p, e);
    *omegaR_dp = DomegaR_Dp(a, p, e);
    *omegaR_de = DomegaR_De(a, p, e);
}


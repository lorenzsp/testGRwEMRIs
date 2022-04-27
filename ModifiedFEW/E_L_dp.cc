double dE_dp = ((1 - Power(e,2))*(1 + ((-1 + Power(e,2))*
          (Power(a,2)*(1 + 3*Power(e,2) + p) + 
            p*(-3 - Power(e,2) + p - (2*
                  Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                    Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                    2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))/Power(p,1.5))))/
        (-4*Power(a,2)*Power(-1 + Power(e,2),2) + Power(3 + Power(e,2) - p,2)*p) + 
       ((-1 + Power(e,2))*p*((3 + Power(e,2) - 3*p)*(3 + Power(e,2) - p)*
             (Power(a,2)*(1 + 3*Power(e,2) + p) + 
               p*(-3 - Power(e,2) + p - 
                  (2*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                       Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                       2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))/Power(p,1.5))) + 
            ((4*Power(a,2)*Power(-1 + Power(e,2),2) - Power(3 + Power(e,2) - p,2)*p)*
               (Power(a,6)*Power(-1 + Power(e,2),2) - 2*Power(a,4)*(1 + Power(e,2))*Power(p,2) + 
                 Power(p,1.5)*(-3 - Power(e,2) + 2*p)*
                  Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                    Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                    2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))) + 
                 Power(a,2)*(4*(-1 + Power(e,2))*Power(p,2) + 8*Power(p,3) - 3*Power(p,4) + 
                    Power(p,1.5)*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                       Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                       2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))))/
             (Power(p,1.5)*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                 Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                 2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))))/
        Power(-4*Power(a,2)*Power(-1 + Power(e,2),2) + Power(3 + Power(e,2) - p,2)*p,2)))/
   (2.*Power(p,2)*Sqrt(1 - ((1 - Power(e,2))*
          (1 + ((-1 + Power(e,2))*(Power(a,2)*(1 + 3*Power(e,2) + p) + 
                 p*(-3 - Power(e,2) + p - 
                    (2*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                         Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                         2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))/Power(p,1.5))))/
             (-4*Power(a,2)*Power(-1 + Power(e,2),2) + Power(3 + Power(e,2) - p,2)*p)))/p));

    //
    double dL_dp = (-(((3 + Power(e,2) - 3*p)*(3 + Power(e,2) - p)*p*
          Sqrt(Power(a,2)*(1 + 3*Power(e,2) + p) + 
            p*(-3 - Power(e,2) + p - (2*
                  Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                    Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                    2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))/Power(p,1.5))))/
        Power(-4*Power(a,2)*Power(-1 + Power(e,2),2) + Power(3 + Power(e,2) - p,2)*p,1.5)) + 
     (2*Sqrt(Power(a,2)*(1 + 3*Power(e,2) + p) + 
          p*(-3 - Power(e,2) + p - (2*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                  Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                  2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))/Power(p,1.5))))/
      Sqrt(-4*Power(a,2)*Power(-1 + Power(e,2),2) + Power(3 + Power(e,2) - p,2)*p) + 
     (Power(a,6)*Power(-1 + Power(e,2),2) - 2*Power(a,4)*(1 + Power(e,2))*Power(p,2) + 
        Power(p,1.5)*(-3 - Power(e,2) + 2*p)*
         Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
           Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
           2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))) + 
        Power(a,2)*(4*(-1 + Power(e,2))*Power(p,2) + 8*Power(p,3) - 3*Power(p,4) + 
           Power(p,1.5)*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
              Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
              2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p)))))/
      (Sqrt(p)*Sqrt(-4*Power(a,2)*Power(-1 + Power(e,2),2) + Power(3 + Power(e,2) - p,2)*p)*
        Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
          Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
          2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p)))*
        Sqrt(Power(a,2)*(1 + 3*Power(e,2) + p) + 
          p*(-3 - Power(e,2) + p - (2*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                  Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                  2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))/Power(p,1.5)))) + 
     (a*(1 - Power(e,2))*(1 + ((-1 + Power(e,2))*
             (Power(a,2)*(1 + 3*Power(e,2) + p) + 
               p*(-3 - Power(e,2) + p - 
                  (2*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                       Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                       2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))/Power(p,1.5))))/
           (-4*Power(a,2)*Power(-1 + Power(e,2),2) + Power(3 + Power(e,2) - p,2)*p) + 
          ((-1 + Power(e,2))*p*((3 + Power(e,2) - 3*p)*(3 + Power(e,2) - p)*
                (Power(a,2)*(1 + 3*Power(e,2) + p) + 
                  p*(-3 - Power(e,2) + p - 
                     (2*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                          Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                          2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))/Power(p,1.5))) + 
               ((4*Power(a,2)*Power(-1 + Power(e,2),2) - Power(3 + Power(e,2) - p,2)*p)*
                  (Power(a,6)*Power(-1 + Power(e,2),2) - 
                    2*Power(a,4)*(1 + Power(e,2))*Power(p,2) + 
                    Power(p,1.5)*(-3 - Power(e,2) + 2*p)*
                     Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                       Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                       2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))) + 
                    Power(a,2)*(4*(-1 + Power(e,2))*Power(p,2) + 8*Power(p,3) - 3*Power(p,4) + 
                       Power(p,1.5)*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                          Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                          2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))))/
                (Power(p,1.5)*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                    Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                    2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))))/
           Power(-4*Power(a,2)*Power(-1 + Power(e,2),2) + Power(3 + Power(e,2) - p,2)*p,2)))/
      (Power(p,2)*Sqrt(1 - ((1 - Power(e,2))*
             (1 + ((-1 + Power(e,2))*(Power(a,2)*(1 + 3*Power(e,2) + p) + 
                    p*(-3 - Power(e,2) + p - 
                       (2*Sqrt(Power(a,6)*Power(-1 + Power(e,2),2) + 
                            Power(a,2)*(-4*Power(e,2) + Power(-2 + p,2))*Power(p,2) + 
                            2*Power(a,4)*p*(-2 + p + Power(e,2)*(2 + p))))/Power(p,1.5))))/
                (-4*Power(a,2)*Power(-1 + Power(e,2),2) + Power(3 + Power(e,2) - p,2)*p)))/p)))/2.;


//double Edot = -(6.4*pow(y,5) - 23.752380952380953*pow(y,6) + 1.6*(50.26548245743669 - 11.*a)*pow(y,6.5) + (-31.54215167548501 + 13.2*pow(a,2))*pow(y,7) + 0.009523809523809525*(-25732.785425553997 - 2646.*a - 504.*pow(a,3))*pow(y,7.5) + 
//        (-649.6614141423464 + 260.32427983539094*a + 163.36281798666926*pow(a,2) - 32.13333333333333*pow(a,3))*pow(y,8.5) + pow(y,8)*(740.6829867239124 - 217.8170906488923*a + 7.758730158730159*pow(a,2) - 52.17523809523809*log(y)) + 
//        pow(y,9)*(-748.828100625135 - 515.5802343491364*a + 69.31499118165785*pow(a,2) + 5.2*pow(a,4) + 3.2*Sqrt(1. - 1.*pow(a,2)) + 41.6*pow(a,2)*Sqrt(1. - 1.*pow(a,2)) + 19.2*pow(a,4)*Sqrt(1. - 1.*pow(a,2)) + 12.8*(a + 3.*pow(a,3))*1.0 + 168.77786848072563*log(y)) + 
//        pow(y,9.5)*(4602.42139029395 - 1328.0603895410009*a + 252.7336489987903*pow(a,2) - 246.65044091710757*pow(a,3) - 1.9428571428571428*pow(a,5) + 0.030476190476190476*(-21513.626491782903 + 6841.*a)*log(y)));

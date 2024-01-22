
#include <Rcpp.h>
using namespace Rcpp;

double rcpp_activation_derivative(double v, int actFun = 0) {
  switch( actFun ) {
  case 1:
    return 1.0;
  case 2:
    return v * (1.0 - v);
  case 3:
    return (0.5 + v) * (0.5 - v);
  case 4:
    if( v > 0.0 ) return 1.0 ;
    else return 0.0;
  case 5:
    return (v - floor(v)) * (1.0 - v + floor(v));
  case 6:
    return v * (1.0 - v);
  default:
    return 1.0;
  }
}

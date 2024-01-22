
#include <Rcpp.h>
using namespace Rcpp;

double rcpp_activation(double z, int actFun = 0) {
  double w = 0.0;
  switch( actFun ) {
  case 1:
    return z;
  case 2:
    w = z;
    if( w >  20.0 ) w =  20.0;
    if( w < -20.0 ) w = -20.0;
    return 1.0 / (1.0 + exp(-w));
  case 3:
    w = z;
    if( w >  20.0 ) w =  20.0;
    if( w < -20.0 ) w = -20.0;
    return 1.0 / (1.0 + exp(-z)) - 0.5;
  case 4:
    if( z > 0.0 ) return z ;
    else return 0;
  case 5:
    return floor(z) + 1.0/(1.0 + exp(-30*(z - floor(z) - 0.5)));
  default:
    return z;
  }
}

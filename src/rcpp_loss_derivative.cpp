
#include <Rcpp.h>
using namespace Rcpp;

NumericVector rcpp_loss_derivative(NumericVector y, NumericVector yhat, int lossFun) {
  int nY = y.length();
  NumericVector d (nY);
  double dv = 0.0;
  double delta = 1.5;
  switch( lossFun ) {
  case 1:
    for(int i = 0; i < nY; i++) {
      if( y[i] > yhat[i] ) d[i] = -1.0;
      else d[i] = 1.0;
    }
    return d;
  case 2:
    return -2.0 * (y - yhat);
  case 3:
    for(int i = 0; i < nY; i++) {
      dv =  abs(y[i] - yhat[i]);
      if( dv < delta ) d[i] = -2 * (y[i] - yhat[i]);
      else {
        if( y[i] > yhat[i] ) d[i] = -2.0 * delta;
        else d[i] = 2.0 * delta;
      }
    }
    return d;
  case 4:
    return -(y - yhat) / (yhat * (1.0 - yhat));
  case 5:
    return -(y - yhat) / (yhat * (1.0 - yhat));
  default:
    return (y - yhat);
  }
}

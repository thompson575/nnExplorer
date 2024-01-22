
#include <Rcpp.h>
using namespace Rcpp;

double rcpp_loss(NumericVector y, NumericVector yhat, int lossFun = 1) {
  double loss = 0.0;
  double dv = 0.0;
  double delta = 1.5;
  double w = 0.0;
  int nY = y.length();
  switch( lossFun ) {
  case 1:
    for(int i = 0; i < nY; i++) loss += abs(y[i] - yhat[i]);
    break;
  case 2:
    for(int i = 0; i < nY; i++) loss += (y[i] - yhat[i])*(y[i] - yhat[i]);
    break;
  case 3:
    for(int i = 0; i < nY; i++) {
      dv =  abs(y[i] - yhat[i]);
      if( dv < delta ) loss += dv * dv;
      else loss += 2.0 * delta * dv - delta * delta;
      loss += dv;
    }
    break;
  case 4:
    for(int i = 0; i < nY; i++) loss -= y[i] * log(yhat[i]) + (1.0 - y[i]) * log(1.0 - yhat[i]);
    break;
  case 5:
    for(int i = 0; i < nY; i++) {
      w = yhat[i];
      if( w < 0.00001 ) w = 0.00001;
      if( w > 0.99999 ) w = 0.99999;
      if( y[i] > 0.5) loss = -log(w);
    }
  }
  return loss;
}


#include <Rcpp.h>
using namespace Rcpp;

double rcpp_activation(double, int);

// [[Rcpp::export]]
NumericVector rcpp_forward_pass(NumericVector v,
                          NumericVector bias,
                          NumericVector weight,
                          IntegerVector from,
                          IntegerVector to,
                          IntegerVector nPtr,
                          IntegerVector wPtr,
                          IntegerVector actFun) {
  // Size of network
  int nLayers = nPtr.length() - 1;
  int nNodes  = v.length();
  double s = 0.0;
  double w = 0.0;
  // z = linear combinations inputs to each node
  // v = value of each node v = activation(z)
  NumericVector z(nNodes);
  NumericVector p(nNodes);
  for(int i = 0; i < nNodes; i++) z[i] = bias[i];
  for(int i = 1; i < nLayers; i++) {
    for(int k = nPtr[i]; k < nPtr[i+1]; k++ ) {
      for(int h = wPtr[i-1]; h < wPtr[i]; h++) {
        if( to[h] == k ) {
          z[k] += weight[h] * v[from[h]];
        }
      }
    }
    // apply activation function
    if( actFun[i-1] == 6 ) {
      // softmax
      s = 0.0;
      for(int k = nPtr[i]; k < nPtr[i+1]; k++ ) {
        w = z[k];
        if( w >  20.0 ) w =  20.0;
        if( w < -20.0 ) w = -20.0;
        p[k] = exp(w);
        s += p[k];
      }
      for(int k = nPtr[i]; k < nPtr[i+1]; k++ ) {
        v[k] = p[k] / s;
      }
    } else{
      for(int k = nPtr[i]; k < nPtr[i+1]; k++ ) {
        v[k] = rcpp_activation( z[k], actFun[i-1]);
      }
    }
  }
  return v;
}


#include <Rcpp.h>
using namespace Rcpp;

NumericVector rcpp_penalty_derivative(NumericVector beta, double lambda, int skip = 0, int penalty = 0) {
  int m = beta.length();
  NumericVector p (m);
  switch( penalty ) {
  case 1:
    for(int i = skip; i < m; i++) {
      if( beta[i] > 0 ) p[i] = lambda;
      else p[i] = -lambda;
    }
    return p;
  case 2:
    for(int i = skip; i < m; i++) p[i] = 2.0 * beta[i] * lambda;
    return p;
  default:
    return p;
  }
}

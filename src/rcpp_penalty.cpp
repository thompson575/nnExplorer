
#include <Rcpp.h>
using namespace Rcpp;

double rcpp_penalty(NumericVector beta, double lambda, int skip = 0, int penalty = 0) {
  int m = beta.length();
  double p = 0.0;
  switch( penalty ) {
  case 1:
    for(int i = skip; i < m; i++) p += abs(beta[i]);
    return p * lambda;
  case 2:
    for(int i = skip; i < m; i++) p += beta[i] * beta[i];
    return p * lambda;
  default:
    return p;
  }
}
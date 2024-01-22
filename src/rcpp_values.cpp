
#include <Rcpp.h>
using namespace Rcpp;

NumericVector rcpp_forward_pass(NumericVector,
                                NumericVector,
                                NumericVector,
                                IntegerVector,
                                IntegerVector,
                                IntegerVector,
                                IntegerVector,
                                IntegerVector);
  
  // ------------------------------------------------------------------
  // Values of each node for a fitted NN
  // inputs
  //    X matrix of test data (predictors)
  //    design list as returned by cpreprare_nn()
  //    actFun coded activation functions for each layer
  // returns 
  //    Y a matrix of nodal values
  //
  // [[Rcpp::export]]
  NumericMatrix rcpp_values( NumericMatrix X, 
                                 List design) {
    // unpack the design
    IntegerVector from = design["from"];
    IntegerVector to   = design["to"];
    IntegerVector nPtr = design["nPtr"];
    IntegerVector wPtr = design["wPtr"];
    NumericVector bias = design["bias"];
    NumericVector weight = design["weight"];
    IntegerVector actFun = design["actFun"];
    // size of the test data
    int nr = X.nrow();
    int nX = X.ncol();
    int nNodes = bias.length();
    int nY = nNodes - nX;
    NumericVector v (nNodes);
    NumericMatrix Y (nr, nY);
    // iterate over the rows of the test data
    for( int d = 0; d < nr; d++) {
      // set the predictors into v
      for(int i = 0; i < nX; i++) v[i] = X(d, i);
      // forward pass
      v = rcpp_forward_pass(v, bias, weight, from, to, nPtr, wPtr, actFun);
      // extract the predictions
      for(int i = 0; i < nY; i++) Y(d, i) = v[nX + i];
    }
    return Y;
  }
  
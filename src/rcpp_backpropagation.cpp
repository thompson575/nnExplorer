
#include <Rcpp.h>
using namespace Rcpp;

NumericVector rcpp_loss_derivative(NumericVector, NumericVector, int);
double rcpp_activation_derivative(double, int);
  
// [[Rcpp::export]]
List rcpp_backpropagation(NumericVector y, 
                  NumericVector v,
                  NumericVector bias,
                  NumericVector weight,
                  IntegerVector from,
                  IntegerVector to,
                  IntegerVector nPtr,
                  IntegerVector wPtr,
                  IntegerVector actFun,
                  int           lossFun) {
  
  // Size of network
  int nLayers = nPtr.length() - 1;
  int nNodes  = bias.length();
  int nWts    = weight.length();
  int nY      = y.length();
  // define structures 
  double wjk  = 0.0;
  NumericVector dbias(nNodes);
  NumericVector dweight(nWts);
  NumericVector dv (nNodes);
  NumericVector df (nNodes);
  NumericVector yhat (nY);
  NumericVector dLoss (nY);
  
  // df = derivatives of activation functions
  for(int i = 1; i < nLayers; i++) {
    for(int j = nPtr[i]; j < nPtr[i+1]; j++) {
      df[j] = rcpp_activation_derivative(v[j], actFun[i-1]);
    }
  }
  // yhat = predicted network outputs
  for(int j = nPtr[nLayers-1]; j < nPtr[nLayers]; j++) {
    yhat[j-nPtr[nLayers-1]] = v[j];
  }
  // derivative of loss wrt to yhat
  dLoss = rcpp_loss_derivative(y, yhat, lossFun);
  // dv derivatives of loss wrt each nodal value 
  // dbias derivatives of loss wrt the bias
  // dweight derivatives of loss wrt the weights
  for(int j = nPtr[nLayers-1]; j < nPtr[nLayers]; j++) {
    dv[j] = dLoss[j-nPtr[nLayers-1]];
  }
  for(int i = nLayers-1; i > 0; i-- ) {
    for(int j = nPtr[i]; j < nPtr[i+1]; j++) {
      dbias[j] = dv[j] * df[j];
      for(int h = wPtr[i-1]; h < wPtr[i]; h++) {
        if( to[h] == j) {
          dweight[h] = dv[j] * df[j] * v[from[h]];
        }
      }
    }
    for(int j = nPtr[i-1]; j < nPtr[i]; j++) {
      dv[j] = 0.0;
      for(int k = nPtr[i]; k < nPtr[i+1]; k++) {
        for(int h = wPtr[i-1]; h < wPtr[i]; h++) {
          if( (from[h] == j) & (to[h] == k) ) wjk = weight[h];
        }
        dv[j] += dv[k] * df[k] * wjk;
      }
    }
  }
  // return derivatives as a named list
  List L = List::create(Named("dbias") = dbias , 
                        _["dweight"] = dweight);
  return L;
}

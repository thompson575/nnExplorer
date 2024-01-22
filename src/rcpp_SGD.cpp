
#include <Rcpp.h>
using namespace Rcpp;

// functions used by rcpp_GD
double rcpp_loss(NumericVector, NumericVector, int);
double rcpp_penalty(NumericVector, double, int, int);
NumericVector rcpp_penalty_derivative(NumericVector, double, int, int);
List rcpp_backpropagation(NumericVector, 
                          NumericVector,
                          NumericVector,
                          NumericVector,
                          IntegerVector,
                          IntegerVector,
                          IntegerVector,
                          IntegerVector,
                          IntegerVector,
                          int);
NumericVector rcpp_forward_pass(NumericVector,
                                NumericVector,
                                NumericVector,
                                IntegerVector,
                                IntegerVector,
                                IntegerVector,
                                IntegerVector,
                                IntegerVector);
// ------------------------------------------------------------------
// Fit a NN by Full Gradient Descent
// inputs
//   X training data predictors
//   Y training data responses
//   design as returned by nnDesign
//   actFun coded activation functions
//   eta the initial step length
//   etaDrop gap between scheduled reductions in eta (0=No drop)
//   penalty 0=no penalty 1=L1 penalty 2=L2 penalty
//   lambdaBias penalty coefficient for bias term
//   lambdaWeight penalty coefficient for weight term
//   nEpoch number of iterations
//   trace number of iterations before reporting progress
//
// outputs
//   List containing
//     bias   the biases with min training loss
//     weight the weights with min training loss
//     lossHistory loss after each iteration
//     penHistory loss+penalty after each iteration
//     eta the final step length
//     finalbias   the biases on completion
//     finalweight the weights on completion
//     dbias  derivative of bias on completion
//     dweight derivative of the weight on completion
//
// [[Rcpp::export]]
List rcpp_SGD(   NumericMatrix X, 
                NumericMatrix Y, 
                List design, 
                double eta = 0.1, 
                int etaDrop = 0,
                int penalty = 0,
                double lambdaBias = 0.0,
                double lambdaWeight = 0.0,
                int batch = 20,
                int nIter = 10,
                double momentum = 0.0,
                double rmsprop = 0.0,
                double beta = 0.99,
                int warmup = 500,
                int nEpoch = 1000, 
                double epsDeriv = 0,
                int trace = 0 ) {
  // unpack the design
  IntegerVector from   = design["from"];
  IntegerVector to     = design["to"];
  IntegerVector nPtr   = design["nPtr"];
  IntegerVector wPtr   = design["wPtr"];
  NumericVector bias   = design["bias"];
  NumericVector weight = design["weight"];
  IntegerVector arch   = design["arch"];
  IntegerVector actFun = design["actFun"];
  int           lossFun = design["lossFun"];
  // size of the problem
  int nr = X.nrow();
  int nX = X.ncol();
  int nY = Y.ncol();
  int nNodes  = bias.length();
  int nWts    = weight.length();
  // working variables
  NumericVector v (nNodes);                      // value of each node
  NumericVector yhat (nY);                       // predicted outputs
  NumericVector y (nY);                          // output values
  NumericVector lossHistory (nEpoch);            // loss at each epoch
  NumericVector penHistory (nEpoch);             // penalised loss at each epoch
  NumericVector magHistory (nEpoch);             // smooth MAG at each epoch
  NumericVector dpenw (nWts);                    // derivative of penaly wrt weight
  NumericVector dpenb (nNodes - nX);             // derivative of penalty wrt bias
  NumericVector dw (nWts);                       // derivative of loss wrt weight
  NumericVector db (nNodes);                     // derivative of loss wrt bias
  NumericVector odw (nWts);
  NumericVector odb (nNodes);
  NumericVector odw2 (nWts, 1.0);
  NumericVector odb2 (nNodes, 1.0);
  NumericVector sdw (nWts);
  NumericVector sdb (nNodes);
  NumericVector fweight (nWts);                  // weight when loss minimised
  NumericVector fbias (nNodes);                  // bias when loss minimised
  NumericVector sweight (nWts);
  NumericVector sbias (nNodes);
  NumericVector startWeight (nWts);
  NumericVector startBias (nNodes);
  double startEta = eta;
  double tloss       = 0.0;
  double minLoss     = 0.0;
  double thisPenalty = 0.0;
  double mag         = 0.0;
  int used           = 0;
  int d              = 0;
  double prop        = 0.0;
  double u           = 0.0;
  int cprint         = 0;
  int converged      = 0;
  double bc1         = 1.0;
  double bc2         = 1.0;
  // reset trace
  if( trace == 0  ) trace   = nEpoch + 1;
  if( etaDrop == 0) etaDrop = nEpoch + 1;
  // record starting state
  startEta = eta;
  for(int i = 0; i < nWts; i++)   startWeight[i] = weight[i];
  for(int i = nX; i < nNodes; i++)  startBias[i] = bias[i];
  // iterate nEpoch times
  for(int epoch = 0; epoch < nEpoch; epoch++) {
    // iterate batch derivatives without calculating the loss
    for( int iter = 0; iter < nIter; iter++) {
      // reset derivatives & loss to zero
      for(int i = 0; i < nWts; i++)    dw[i] = 0.0;
      for(int i = nX; i < nNodes; i++) db[i] = 0.0;
      // iterate over the rows of the training data
      used = 0;
      d    = 0;
      while( (d < nr) & (used < batch) ) {
        // test for inclusion in mini-batch
        u = R::runif(0.0, 1.0);
        prop = (double) (batch - used) / (double) (nr - d);
        if( u < prop ) {
          used++;
          // set the input values into v
          for(int i = 0; i < nX; i++) v[i] = X(d, i);
          // extract the output values
          for(int i = 0; i < nY; i++) y[i] = Y(d, i);
          // forward pass
          v = rcpp_forward_pass(v, bias, weight, from, to, nPtr, wPtr, actFun);
          // back-propagate
          List deriv = rcpp_backpropagation(y, v, bias, weight, from, to, nPtr, wPtr, actFun, lossFun);
          NumericVector dweight = deriv["dweight"];
          NumericVector dbias   = deriv["dbias"];
          // sum the derivatives
          for(int i = 0; i < nWts; i++)    dw[i] += dweight[i];
          for(int i = nX; i < nNodes; i++) db[i] += dbias[i];
          }
        d++;
       }
      // derivatives of the penalty
      if( penalty != 0 ) {
        dpenb = rcpp_penalty_derivative(bias,   lambdaBias,  nX, penalty); 
        dpenw = rcpp_penalty_derivative(weight, lambdaWeight, 0, penalty); 
        // update the weights and biases
        for(int i = 0;  i < nWts; i++)   dw[i] += (dpenw[i] * batch / nr);
        for(int i = nX; i < nNodes; i++) db[i] += (dpenb[i] * batch / nr);
      }
      for(int i = 0; i < nWts; i++)    dw[i]  /= batch;
      for(int i = nX; i < nNodes; i++) db[i]  /= batch;
      // smooth for early stopping
      for(int i = 0; i < nWts; i++)    sdw[i]  = 0.99 * sdw[i] + 0.01 * dw[i];
      for(int i = nX; i < nNodes; i++) sdb[i]  = 0.99 * sdb[i] + 0.01 * db[i];
      // Momentum .. rmsprop .. adam
      if( momentum > 0.0 ) {
        // smoothed gradients
        bc1 *= momentum;
        for(int i = 0; i < nWts; i++)    odw[i]  = momentum * odw[i] + (1.0 - momentum) * dw[i];
        for(int i = nX; i < nNodes; i++) odb[i]  = momentum * odb[i] + (1.0 - momentum) * db[i];
      } 
      if( rmsprop > 0.0 ) {
        // smoothed squared gradients
        bc2 *= rmsprop;
        for(int i = 0; i < nWts; i++)    odw2[i]  = rmsprop * odw2[i] + (1.0 - rmsprop) * dw[i] * dw[i];
        for(int i = nX; i < nNodes; i++) odb2[i]  = rmsprop * odb2[i] + (1.0 - rmsprop) * db[i] * db[i];
      }
      // Updates
      if( (epoch < warmup) | ((momentum == 0.0) & (rmsprop == 0.0))) {
        // simple SGD
        for(int i = 0; i < nWts; i++)  weight[i] -= eta * dw[i];        
        for(int i = nX; i < nNodes; i++) bias[i] -= eta * db[i];      
      } else if( ((momentum > 0.0) & (rmsprop == 0.0)) ) {
        // Momentum
        for(int i = 0; i < nWts; i++)  weight[i] -= eta * odw[i];        
        for(int i = nX; i < nNodes; i++) bias[i] -= eta * odb[i];      
      } else if( ((momentum == 0.0) & (rmsprop > 0.0)) ) {
        // RMSProp
        for(int i = 0; i < nWts; i++)  weight[i] -= eta * dw[i] / sqrt(odw2[i]);        
        for(int i = nX; i < nNodes; i++) bias[i] -= eta * db[i] / sqrt(odb2[i]);      
      } else {
        // Adam
        for(int i = 0; i < nWts; i++)  weight[i] -= eta * (odw[i]/(1.0-bc1)) / (sqrt(odw2[i]/(1.0-bc2)));        
        for(int i = nX; i < nNodes; i++) bias[i] -= eta * (odb[i]/(1.0-bc1)) / (sqrt(odb2[i]/(1.0-bc2)));      
      } 
      // smoothed parameter estimates
      for(int i = 0; i < nWts; i++)  sweight[i] = beta * sweight[i] + (1.0 - beta) * weight[i];        
      for(int i = nX; i < nNodes; i++) sbias[i] = beta * sbias[i] + (1.0 - beta) * bias[i];      
    } 
    // ONCE PER EPOCH evaluate the loss
    tloss = 0.0;
    for( int d = 0; d < nr; d++) {
      // set the predictors into v
      for(int i = 0; i < nX; i++) v[i] = X(d, i);
      // forward pass
      v = rcpp_forward_pass(v, bias, weight, from, to, nPtr, wPtr, actFun);
      // extract the predictions
      for(int i = 0; i < nY; i++) {
        yhat[i] = v[nNodes - nY + i];
        y[i]    = Y(d, i);
      }
      // calculate the loss 
      tloss += rcpp_loss(y, yhat, lossFun);
    }
    // save loss 
    lossHistory[epoch] = tloss / nr;
    // add penalty 
    if( penalty != 0 ) thisPenalty = rcpp_penalty(weight, lambdaWeight, 0, penalty) + 
        rcpp_penalty(bias, lambdaBias, nX, penalty);
    penHistory[epoch] = lossHistory[epoch] + thisPenalty / nr;
    // save parameters if loss is an improvement
    if( (epoch == 0) | (penHistory[epoch] < minLoss) ) {
      minLoss = penHistory[epoch];
      for(int i = 0;  i < nWts;   i++) fweight[i] = weight[i];
      for(int i = nX; i < nNodes; i++) fbias[i]   = bias[i];
    }
    // smoothed mean absolute gradient
    mag = 0.0;
    for(int i = 0;  i < nWts;   i++) mag += abs(sdw[i]);
    for(int i = nX; i < nNodes; i++) mag += abs(sdb[i]);
    mag /= (nWts + nNodes - nX);
    magHistory[epoch] = mag;
    // consider early stopping
    if( (epoch >= warmup) & (mag < epsDeriv) ) converged = 1;
    // reduce learning rate if appropriate
    if( (epoch+1) % etaDrop == 0)  eta *= 0.9;
    // report progress if appropriate
    if( ((epoch+1) % trace == 0) | ((trace < nEpoch) & (converged == 1))) {
      if( cprint % 10 == 0 ) Rprintf("  epoch      loss    penalty    minLoss     step       SMAG\n");
      Rprintf("%6i %10.5f %10.5f %10.5f %8.5f %10.6f\n", 
              epoch+1, lossHistory[epoch], penHistory[epoch], minLoss, eta, mag); 
      cprint++;
    }
    // finished
    if( converged == 1 ) epoch = nEpoch;
  }
  // pack options into a list
  List  Opt = List::create(Named("startEta")  = startEta , 
                           _["etaDrop"]       = etaDrop,
                           _["penalty"]       = penalty,
                           _["lambdaBias"]    = lambdaBias,
                           _["lambdaWeight"]  = lambdaWeight,
                           _["nEpoch"]        = nEpoch,
                           _["epsDeriv"]      = epsDeriv,
                           _["batch"]         = batch,
                           _["nIter"]         = nIter,
                           _["momentum"]      = momentum,
                           _["rmsprop"]       = rmsprop,
                           _["nEpoch"]        = nEpoch,
                           _["trace"]         = trace,
                           _["arch"]          = arch,
                           _["actFun"]        = actFun,
                           _["lossFun"]       = lossFun,
                           _["startWeight"]   = startWeight,
                           _["startBias"]     = startBias);
  // pack results into a list
  List  L = List::create(Named("bias")      = fbias , 
                         _["weight"]        = fweight,
                         _["lossHistory"]   = lossHistory,
                         _["penHistory"]    = penHistory,
                         _["magHistory"]    = magHistory,
                         _["eta"]           = eta,
                         _["finalbias"]     = bias,
                         _["finalweight"]   = weight,
                         _["smoothbias"]    = sbias,
                         _["smoothweight"]  = sweight,
                         _["dbias"]         = db,
                         _["dweight"]       = dw,
                         _["smoothdbias"]   = sdb,
                         _["smoothdweight"] = sdw,
                         _["options"]      = Opt);
  // return the results
  return L;
}
  

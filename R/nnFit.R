#' fit a neural network
#'
#' @param X A matrix of features.
#' @param Y a matrix of outputs.
#' @param design encoded version of the network design as created by nnDesign
#' @param XV optional matrix of validation features
#' @param YV optional matrix of validation features
#' @param control named list of hyperparameters
#' 
#' @return A list containing the results.
#' 
#' Control parameters
#' \itemize {
#'   \item algorithm   GD, GDV, SGD, SGDV. Default GD
#'   \item eta         the step length (learning rate). Default 0.1
#'   \item etaDrop     Epochs between 10% drop in eta. Default 0 (no drop)
#'   \item epochs      Maximum number of epochs (iterations for GD). Default 1000
#'   \item penalty     Regularisation penalty L1 or L2. Default no penalty
#'   \item lambda      Regularisation parameter for weight & bias. Default 0.01
#'   \item lambdabias  lambda to apply to the bias. Default 0.01
#'   \item lambdaweight lambda to apply to the weight. Default 0.01
#'   \item momentum    momentum parameter for SGD. Default 0 (no momentum)
#'   \item rmsprop     rmsprop parameter for SGD. Default 0 (no rmsprop)
#'   \item warmup      period before momentum and rmsprop start. Default 50 epochs
#'   \item batch       batch size for SGD. Default 10
#'   \item nIter       iterations within each epoch. Default 10
#'   \item beta        exponential smoothing parameter for the SGD parameters. Default 0.99
#'   \item epsDeriv    threshold for early stopping based on MAG (mean absolute gradient). Default 0 (no early stopping)
#'   \item maxvrise    maximum number of consecutive rises in the validation loss before early stopping. Default 0 (no early stopping)
#'   \item validGap    epochs between evaluation of the validation loss. Default 1 (every epoch)   
#'   \item trace       epochs between printing current status to screen. Default 0 (no status information)
#'   }
#'   With SGD the loss is evaluated once per epoch but the parameters are updated
#'   nIer times per epoch 
#'   
#' @examples
#' R <- (X, Y, design, control=list(algorithm="GD", eta=0.1))
#' 
nnFit <- function(
    X,
    Y,
    design,   # list(arch, actFun, lossFun)
    XV = NULL,
    YV = NULL,
    control   # list(algorithm, eta, etaDrop, etc.)
) {
  # ---------------------------------------
  # Check Data
  #
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  if( is.vector(Y) ) Y <- matrix(Y, ncol=1)
  if( !is.matrix(X) ) stop("X must be coercable into a matrix")
  if( !is.matrix(Y) ) stop("Y must be coercable into a matrix")
  nrx <- nrow(X)
  ncx <- ncol(X)
  nry <- nrow(Y)
  ncy <- ncol(Y)
  if( nrx != nry ) stop("mismatch between rows of X and Y")
  nv = is.null(XV) + is.null(YV)
  if( nv == 1 ) stop("must include both XV and YV")
  else if( nv == 0 ) {
    XV <- as.matrix(XV)
    YV <- as.matrix(YV)
    if( is.vector(YV) ) YV <- matrix(YV, ncol=1)
    if( !is.matrix(XV) ) stop("XV must be coercable into a matrix")
    if( !is.matrix(YV) ) stop("YV must be coercable into a matrix")
    nrxv <- nrow(X)
    ncxv <- ncol(X)
    nryv <- nrow(Y)
    ncyv <- ncol(Y)
    if( nrx != nry ) stop("mismatch between rows of XV and YV")
    if( ncx != ncxv ) stop("mismatch between X and XV")
    if( ncy != ncyv ) stop("mismatch between Y and YV")
  }
  # ---------------------------------------
  # Check Design
  #
  nLayers <- length(design$arch)
  if( ncx != design$arch[1] ) stop("input layer does not match X")
  if( ncy != design$arch[nLayers]) stop("output layer does not match Y")
  # ---------------------------------------
  # Decode Control
  #
  # default values
  algorithm  <- 1
  eta        <- 0.1
  etadrop    <- 0
  epochs     <- 1000
  penalty    <- 0
  lambda_w   <- 0.01
  lambda_b   <- 0.01
  warmup     <- 50
  rmsprop    <- 0
  momentum   <- 0
  batch      <- 10
  nIter      <- 10
  beta       <- 0.99
  epsderiv   <- 0
  maxvrise   <- 0
  validgap   <- 1
  trace      <- 100
  
  nm  <- tolower(names(control))
  len <- length(nm)
  for(j in 1:len) {
    if( nm[j] == "eta" ) eta = as.numeric(control[[j]])
    else if( substr(nm[j], 1, 4) == "etad" ) etadrop = floor(as.numeric(control[[j]]))
    else if( substr(nm[j], 1, 4) == "epoc" ) epochs = floor(as.numeric(control[[j]]))
    else if( substr(nm[j], 1, 4) == "nepo" ) epochs = floor(as.numeric(control[[j]]))
    else if( substr(nm[j], 1, 4) == "algo" ) {
      algorithm <- nnEncode(tolower(control[[j]]), c("gd", "gdv", "sgd", "sgdv"),
                               1:4, "Algorithm")
    } else if( substr(nm[j], 1, 4) == "pena" ) {
      penalty <- nnEncode(tolower(control[[j]]), c("l1", "l2"),
                          1:2, "Penalty")
    } else if( substr(nm[j], 1, 7) == "lambdab" ) lambda_b = as.numeric(control[[j]])
    else if( substr(nm[j], 1, 7) == "lambdaw" ) lambda_w = as.numeric(control[[j]])
    else if( substr(nm[j], 1, 4) == "lamb" ) {
      lambda_b = as.numeric(control[[j]])
      lambda_w = as.numeric(control[[j]])
    }
    else if( substr(nm[j], 1, 4) == "batc" ) batch = floor(as.numeric(control[[j]]))
    else if( substr(nm[j], 1, 4) == "iter" ) nIter = floor(as.numeric(control[[j]]))
    else if( substr(nm[j], 1, 4) == "nite" ) nIter = floor(as.numeric(control[[j]]))
    else if( substr(nm[j], 1, 4) == "rmsp" ) rmsprop = as.numeric(control[[j]])
    else if( substr(nm[j], 1, 4) == "mome" ) momentum = as.numeric(control[[j]])
    else if( substr(nm[j], 1, 4) == "warm" ) warmup = floor(as.numeric(control[[j]]))
    else if( substr(nm[j], 1, 4) == "trac" ) trace = floor(as.numeric(control[[j]]))
    else if( substr(nm[j], 1, 4) == "vali" ) validgap = floor(as.numeric(control[[j]]))
    else if( nm[j] == "beta" ) beta = control[[j]]
    else if( substr(nm[j], 1, 4) == "epsd" ) epsderiv = as.numeric(control[[j]])
    else if( substr(nm[j], 1, 4) == "maxv" ) maxvrise = floor(as.numeric(control[[j]]))
    else stop(paste("control option", nm[j], "not recognised" ))
  }
  if( eta <= 0 ) stop("eta must be positive")
  if( etadrop < 0 ) stop("etadrop cannot be negative")
  if( epochs < 0 ) stop("etadrop cannot be negative")
  if( lambda_b < 0 | lambda_w < 0) stop("lambdas cannot be negative")
  if( warmup < 0 ) stop("warmup cannot be negative")
  if( validgap < 1 ) stop("validgap cannot be less than 1")
  if( batch < 0 ) stop("batch cannot be negative")
  if( nIter < 0 ) stop("nIter cannot be negative")
  if( trace < 0 ) stop("trace cannot be negative")
  if( rmsprop < 0 | rmsprop >= 1) stop("rmsprop must be between 0 and 1")
  if( momentum < 0 | momentum >= 1) stop("momentum must be between 0 and 1")
  if( beta < 0 | beta >= 1) stop("beta must be between 0 and 1")
  if( algorithm %in% 1:2 ) {
    for(j in 1:len) {
      if( substr(nm[j], 1, 4) %in% c("batc", "rmsp", "mome", "beta", "iter", "nite", "warm") ) 
        stop(paste("hyperparameter ", nm[j], "not relevant to Gradient Descent"))
    }
    if( algorithm == 1 ) {
      for(j in 1:len) {
        if( substr(nm[j], 1, 3) %in% c("val") |
            substr(nm[j], 1, 5) %in% c("maxvr")  ) 
          stop(paste("hyperparameter ", nm[j], "not relevant without validation"))
      }
      if( nv != 2 ) stop("XV YV not required for algorithm GD")
    }
    if( algorithm == 2 & nv != 0 ) stop("XV YV required for algorithm GDV")
  }
  # ---------------------------------------
  # Call chosen algorithm
  #
  if( algorithm == 1 ) {
    # Gradient Descent without Validation 
    R <- rcpp_GD(X, Y, design, eta, etadrop, penalty, lambda_b,
                 lambda_w, epochs, epsderiv, trace)
  } else if( algorithm == 2 ) {
    # Gradient Descent with Validation
    R <- rcpp_GDV(X, Y, design, XV, YV, eta, etadrop, penalty, lambda_b,
                 lambda_w, epochs, epsderiv, maxvrise, validgap, trace)
  } else if( algorithm == 3 ) {
    # Stochastic Gradient Descent without Validation
    R <- rcpp_SGD(X, Y, design, eta, etadrop, penalty, lambda_b,
                 lambda_w, batch, nIter, momentum, rmsprop, beta, warmup,
                 epochs, epsderiv, trace)
  } else if( algorithm == 4 ) {
    # Stochastic Gradient Descent with Validation
    R <- rcpp_SGDV(X, Y, design, XV, YV, eta, etadrop, penalty, lambda_b,
                  lambda_w, batch, nIter, momentum, rmsprop, beta, warmup,
                  epochs, epsderiv, validgap, trace)
  }
  
  return(R)
}

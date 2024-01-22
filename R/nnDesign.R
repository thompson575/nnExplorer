#' create the design file for a neural network
#'
#' @param architecture vector containing the number of nodes in each layer
#' @param actFun string vector containing the activation functions for layers 2, 3 etc.
#' @param lossFun string containing the name of the loss function
#' @param seed optional positive seed for the initial values
#' 
#' @return The encoded network design.
#' 
#' @examples
#' nnDesign(arch=c(1, 4, 1), act=c("sigmoid", "identity"), loss="L2" )
#' 
nnDesign <- function(architecture, actFun, lossFun, drop=NULL, skip=NULL, seed=-1) {
  nLayers <- length(architecture)
  if( length(actFun) != nLayers - 1 ) stop("Mismatch between architecture and activation")
  # --------------------------------------
  # Encode activation
  #
  actOptions <- c("identity", "sigmoid", "logistic", "csigmoid", "clogistic",
               "relu", "step", "softmax")
  actNumber <- c(1, 2, 2, 3, 3, 4, 5, 6)
  actCode   <- nnEncode(actFun, actOptions, actNumber, "Activation")
  # --------------------------------------
  # Encode loss
  #
  lossOptions <- c("l1", "l2", "huber", "bcross-entropy", "bce",
               "mcross-entropy", "mcr", "multiclass", "mce")
  lossNumber <- c(1, 2, 3, 4, 4, 5, 5, 5, 5)
  lossCode <- nnEncode(lossFun, lossOptions, lossNumber, "Loss")
  # --------------------------------------
  # Encode architecture
  #
  nNodes <- sum(architecture)
  nWt = 0;
  for(j in 2:nLayers) nWt <- nWt + architecture[j-1] * architecture[j];
  from <- to <- rep(0, nWt)
  h <- q1 <- q2 <- 0
  for(j in 2:nLayers) {
    q1 <- q1 + architecture[j-1];
    for(f in 1:architecture[j-1]) {
      for(t in 1:architecture[j]) {
        h       <- h + 1
        from[h] <- f + q2 - 1;
        to[h]   <- t + q1 - 1;
      }
    }
    q2 <- q1
  }
  wPtr <- rep(0, nLayers)
  for(j in 2:nLayers) {
    wPtr[j] <-  wPtr[j-1] + architecture[j-1] * architecture[j];
  }
  wPtr[nLayers] = nWt;
  nPtr <- rep(0, nLayers + 1)
  h    <- 0
  for(j in 1:nLayers) {
    nPtr[j] <- h;
    h       <- h + architecture[j];
  }
  nPtr[nLayers + 1] = nNodes;
  # drop weights
  if( !is.null(drop) ) {
    if( !is.matrix(drop) ) stop("drop must be a matrix")
    if( ncol(drop) != 2 ) stop("matrix drop must have two columns")
    for(i in 1:nrow(drop)) {
      nWts <- length(from)
      j <- 0
      while( j < nWts) {
        j <- j + 1
        cat(from[j], " ", to[j], " ", drop[i, 1], " ", drop[i, 2], "\n")
        if( from[j] == drop[i, 1] & to[j] == drop[i, 2]) {
          for(k in j:(length(from)-1)) {
            from[k] <- from[k+1]
            to[k]   <- to[k+1]
          }
          from <- head(from, -1)
          to   <- head(to, -1)
          for(k in 1:length(wPtr)) {
            if( wPtr[k] >= j ) wPtr[k] <- wPtr[k] - 1
          }
          nWts <- nWts - 1
        }
      }
    }
  }
  # add skip
  if( !is.null(skip) ) {
    if( !is.matrix(skip) ) stop("skip must be a matrix")
    if( ncol(skip) != 2 ) stop("matrix skip must have two columns")
  }
  # check for redundant nodes
  # check for redundant layers
  nX <- architecture[1]
  if( seed > 0 ) set.seed(seed)
  bias       <- runif(nNodes, -1, 1)
  bias[1:nX] <- 0
  weight     <- runif(nWt, -1, 1)
  
  return( list(
    bias     = bias,
    weight   = weight,
    from     = from,
    to       = to,
    nPtr     = nPtr,
    wPtr     = wPtr,
    arch     = architecture,
    actFun   = actCode,
    lossFun  = lossCode
    )
  )
}
#' calculate nodal values for a neural network
#'
#' @param X A matrix of features.
#' @param design encoded version of the network design as created by nnDesign
#' 
#' @return A matrix containing the values corresponding to each row of X.
#' 
#' @examples
#' V <- nnValues(XT, design)
#' 
nnValues <- function(
    X,
    design   # list(arch, actFun, lossFun)
) {
  X <- as.matrix(X)
  if( !is.matrix(X) ) stop("X must be coercable into a matrix")
  nrx <- nrow(X)
  ncx <- ncol(X)
  nLayers <- length(design$arch)
  if( ncx != design$arch[1] ) stop("input layer does not match X")
  if( length(design$actFun) != nLayers - 1 ) stop("architecture and activation not compatible")
  
  return( rcpp_values(X, design) )  
}
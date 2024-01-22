#' number of inputs to each node
#'
#' @param design the coded design
#' 
#' @return A vector containing the number of inputs to each node.
#' 
#' @examples
#' lin(myDesign)
#' 
lin <- function(design) {
  L <- rep(0, length(design$weight))
  nLayers <- length(design$arch)
  k <- 0
  for(i in 2:nLayers) {
    size = design$nPtr[i] - design$nPtr[i-1]
    for(j in (design$wPtr[i-1]+1):design$wPtr[i] ) {
      k    <- k + 1
      L[k] <- size 
    }
  }
  return(L)
}

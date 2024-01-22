#' statistics for scaling datasets
#'
#' @param X A matrix/dataframe of data.
#' @param Y Optionally a second matrix/dataframe of data. Defaults to no second dataset.
#' @param xdrop boolean vector whether to exclude each column of X. Defaults to FALSE
#' @param ydrop boolean vector whether to exclude each column of Y. Defaults to FALSE
#' @param method one of "robust", "zero-one" or "scale". Defaults to "robust" 
#' @param eps proportion of the range by which to expand "zero-one" scaling. Defaults to 0
#' 
#' @return A list of values for scaling the datasets.
#' 
#' @examples
#' scaleStats(X, Y, method="scale")
#' 
scaleStats <- function(X, Y=NULL, xdrop = NULL, ydrop = NULL,
                       method = "robust", eps = 0) {
  # -----------------------------------
  # Process X
  #
  nx <- ncol(X)
  if( is.null(xdrop) ) xdrop <- rep(FALSE, nx) 
  mx <- sx <- rep(NA, nx)
  if( method == "robust" ) {
    for(i in 1:nx) {
      if( !xdrop[i] ) {
        q1 <- quantile(X[, i], probs=0.05)
        q3 <- quantile(X[, i], probs=0.95)
        mx[i] <- (q1 + q3) / 2
        sx[i] <- q3 - q1
      }
    } 
  } else if( method == "zero-one") {
    for(i in 1:nx) {
      if( !xdrop[i] ) {
        q1 <- min(X[, i])
        q3 <- max(X[, i])
        mx[i] <- q1 - eps * (q3 - q1)
        sx[i] <- (q3 - q1) * (1 + 2*eps)
      }
    }
  } else if( method == "scale" ) {
    for(i in 1:nx) {
      if( !xdrop[i] ) {
        mx[i] <- mean(X[, i])
        sx[i] <- sd(X[, i])
      }
    }
  } else stop("method not recognised")
  # ---------------------------------------
  # Return results if no Y
  #
  if( is.null(Y) ) return(list(mx=as.numeric(mx), sx=as.numeric(sx)))
  # ---------------------------------------
  # Process Y
  #
  ny <- ncol(Y)
  if( is.null(ydrop) ) ydrop <- rep(FALSE, ny) 
  my <- sy <- rep(NA, ny)    
  if( method == "robust" ) {
    for(i in 1:ny) {
      if( !ydrop[i] ) {
        q1 <- quantile(Y[, i], probs=0.05)
        q3 <- quantile(Y[, i], probs=0.95)
        my[i] <- (q1 + q3) / 2
        sy[i] <- (q3 - q1) / 10
      }
    }
  } else if( method == "zero-one") {
    for(i in 1:ny) {
      if( !ydrop[i] ) {
        q1 <- min(Y[, i])
        q3 <- max(Y[, i])
        my[i] <- q1 - eps * (q3 - q1)
        sy[i] <- (q3 - q1) * (1 + 2*eps)
      }
    }
  } else if( method == "scale" ) {
    if( !is.null(Y) ) {
      for(i in 1:ny) {
        if( !ydrop[i] ) {
          my[i] <- mean(Y[, i])
          sy[i] <- sd(Y[, i])
        }
      }
    }
  }
  return(list(mx=as.numeric(mx), sx=as.numeric(sx), 
                my=as.numeric(my), sy=as.numeric(sy)))
}

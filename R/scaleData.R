#' scale a dataset
#'
#' @param X A matrix/dataframe of data.
#' @param Y Optionally a second matrix/dataframe of data. Defaults to no second dataset.
#' @param stats list of scaling statistics as returned by function scaleStats
#' 
#' @return A scaled copy of the dataset
#' 
#' @examples
#' sts <- scaleStats(X)
#' XS  <- scaleData(X, stats=sts)
#' 
scaleData <- function(X, Y=NULL, stats) {
  # -----------------------------
  # process X
  #
  nx <- ncol(X)
  XT <- X
  for(i in 1:nx) {
    if( !is.na(stats$mx[i]) & !is.na(stats$sx[i])) {
      if( stats$sx[i] != 0 ) XT[, i] <- (X[, i] - stats$mx[i]) / stats$sx[i]
    }
  }
  # -----------------------------
  # return results if no Y
  #
  if( is.null(Y)) return( XT )
  # -----------------------------
  # process Y
  #
  ny <- ncol(Y)
  YT <- Y
  for( i in 1:ny) {
    if( !is.na(stats$my[i]) & !is.na(stats$sy[i])) {
      if( stats$sy[i] != 0 ) YT[, i] <- (Y[, i] - stats$my[i]) / stats$sy[i]
    }
  }
  return(list(X=XT, Y=YT))
}

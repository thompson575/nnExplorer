#' explore dataset for unusual values
#'
#' @param X A vector/matrix/dataframe of data.
#' @param dp decimal places for returned statistics. Negative value implies no rounding. Defaults to -1.
#' @param alpha proportion for high and low quantiles. Defaults to 0.1
#'
#' @return A list of values for scaling the datasets.
#' 
#' @examples
#' exploreData(X)
#' 
exploreData <- function(X, dp = -1, alpha = 0.1) {
  # -----------------------------------
  # Ensure that X is a data frame
  #
  if( is.matrix(X) | is.vector(X)) X <- as.data.frame(X)
  if( !is.data.frame(X)) stop("X cannot not be coerced into a data frame")
  # -----------------------------------
  # Prepare R to contain the results
  #
  nc <- ncol(X)
  nr <- nrow(X)
  nm <- names(X)
  R  <- data.frame( colname = nm, nRows = rep(nr, nc) ) 
  # -----------------------------------
  # basic stats
  #
  R$nMissing <- 0
  R$nUnique  <- 0
  R$modeFreq <- 0
  R$mean     <- NA
  R$sd       <- NA
  R$min      <- NA
  R$qLow      <- NA
  R$q50      <- NA
  R$qHigh      <- NA
  R$max      <- NA
  for(j in 1:nc ) {
    R$nMissing[j] <- sum( is.na(X[, j]))
    v             <- X[ !is.na(X[, j]), j]
    R$nUnique[j]  <- length(unique(v))
    R$modeFreq[j] <- as.integer(max(table(v)))
    if( is.numeric(X[, j])) {
      R$mean[j] <- mean(v)
      R$sd[j]   <- sd(v)
      R$min[j]  <- min(v)
      q         <- quantile(v, probs=c(alpha, 0.5, 1-alpha))
      R$qLow[j]  <- q[1]      
      R$q50[j]   <- q[2]      
      R$qHigh[j] <- q[3]
      R$max[j]  <- max(X[, j])
      delta     <- (R$max[j] - R$min[j]) / 100
      R$skew[j] <- (q[3] - q[2] + delta) / (q[2] - q[1] + delta)
      if( dp >= 0 ) {
        for(k in 6:13) R[, k] <- round(R[, k], dp)
      }
    }
  }
  return(R)
}
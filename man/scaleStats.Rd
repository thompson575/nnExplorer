% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/scaleStats.R
\name{scaleStats}
\alias{scaleStats}
\title{statistics for scaling datasets}
\usage{
scaleStats(X, Y = NULL, xdrop = NULL, ydrop = NULL, method = "robust", eps = 0)
}
\arguments{
\item{X}{A matrix/dataframe of data.}

\item{Y}{Optionally a second matrix/dataframe of data. Defaults to no second dataset.}

\item{xdrop}{boolean vector whether to exclude each column of X. Defaults to FALSE}

\item{ydrop}{boolean vector whether to exclude each column of Y. Defaults to FALSE}

\item{method}{one of "robust", "zero-one" or "scale". Defaults to "robust"}

\item{eps}{proportion of the range by which to expand "zero-one" scaling. Defaults to 0}
}
\value{
A list of values for scaling the datasets.
}
\description{
statistics for scaling datasets
}
\examples{
scaleStats(X, Y, method="scale")

}

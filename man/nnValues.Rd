% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nnValues.R
\name{nnValues}
\alias{nnValues}
\title{calculate nodal values for a neural network}
\usage{
nnValues(X, design)
}
\arguments{
\item{X}{A matrix of features.}

\item{design}{encoded version of the network design as created by nnDesign}
}
\value{
A matrix containing the values corresponding to each row of X.
}
\description{
calculate nodal values for a neural network
}
\examples{
V <- nnValues(XT, design)

}

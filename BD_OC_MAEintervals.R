###########################################################
############## Interval scale classification ##############
###################### SMAE Measure #######################
###########################################################

### Arguments and notation
# C             = (matrix) Confusion matrix obtained by a validation procedure
# r             = (number) Number of classes (intervals)
# n             = (vector) Number of instances for each interval
# N             = (number) Total number of instances
# L             = (vector) Length of the intervals
# haus.dist     = (function) Hausdorff distance between two intervals given in matrix form
# m             = (matrix) Cost matrix
# index.max     = (vector) Vector of indexes of the interval with maximum Hausdorff distance from interval_j for j=1,...,r
# Amax          = (matrix) Matrix representing the class Smax 


### Complementary functions

# Length of the intervals (given a vector of endpoints)
leng <- function(v) {
    r <- length(v)
    a <- v[1:(r-1)]
    b <- v[2:r]
    L <- b-a
    L
}

# Hausdorff distance between intervals (in matrix form)
haus.dist <- function(L) {
    r <- length(L)
    
    # Lower and upper endpoints of the intervals
    endpoints <- c(0, cumsum(L))
    L.low <- endpoints[1:r]
    L.up <- endpoints[2:(r+1)]
    
    # indexes of the intervals
    d.row <- rep(1:r, times = r)
    d.col <- rep(1:r, each = r)
    
    # Hausdorff distance
    d.low <- abs(L.low[d.row]-L.low[d.col])
    d.up <- abs(L.up[d.row]-L.up[d.col])
    d <- pmax(d.low, d.up)
    d <- matrix(d, nrow = r)
    d
}


### Functions

## MAE.int: Interval scale Mean Absolute Error
# Given a confusion matrix C and a vector of interval lengths L, the function returns the Mean Absolute Error of C
MAE.int <- function(C, L) {
    # Check parameters
    stopifnot("The argument C of the function should be a matrix" = class(C)[1] == 'matrix')
    stopifnot("The argument C of the function should be a square matrix" = nrow(C) == ncol(C))
    stopifnot("The argument C of the function is not a confusion matrix (not all its elements are integer numbers)" = all(C == floor(C)))
    stopifnot("The number of classes and the number of intervals do not match" = nrow(C) == length(L))
    stopifnot("The argument L is not a lengths vector (not all its elements are positive)" = all(L > 0))
    
    # Variables
    r <- nrow(C)
    n <- apply(C, 2, sum)
    N <- sum(n)
    
    # Cost matrix m.ij
    m <- haus.dist(L)/N
    
    # MAE
    MAE <- sum(C*m)
    MAE
}


## MAEmax.int = MAE(Amax) Interval scale Maximum Mean Absolute Error
# Given a confusion matrix C and a vector of interval lengths L, the function returns a list including:
# index     = Vector of indexes of the interval with maximum Hausdorff distance from interval_j for j=1,...,r
# Amax      = Matrix Amax
# MAEmx     = Maximum Mean Absolute Error
MAEmax.int <- function(C, L) {
    # Check parameters
    stopifnot("The argument of the function should be a matrix" = class(C)[1] == 'matrix')
    stopifnot("The argument of the function should be a square matrix" = nrow(C) == ncol(C))
    stopifnot("The argument of the function is not a confusion matrix (not all elements are integer numbers)" = all(C == floor(C)))
    stopifnot("The number of classes and the number of intervals do not match" = nrow(C) == length(L))
    stopifnot("The argument L is not a lengths vector (not all its elements are positive)" = all(L > 0))
    
    # Variables
    r <-  nrow(C)
    n <- apply(C, 2, sum)
    N <- sum(n)
    
    # i.j
    index.max <- apply(haus.dist(L), 2, which.max)
    
    # Amax
    Amax <- matrix(0, r, r)
    for (i in 1:r) {
        Amax[index.max[i],i] <- n[i]
        }
    
    # MAEmax
    MAEmx <- sum(Amax*haus.dist(L))/N
    
    # results
    results <- list(index = index.max, Amax = Amax, MAEmx = MAEmx)
    results
}


# SMAE: Interval scale Standard Mean Absolute Error
# Given a matrix confusion C, the function returns the Standard Mean Absolute Error of C
SMAE.int <- function(C, L) {
    # Check parameters
    stopifnot("The argument of the function should be a matrix" = class(C)[1] == 'matrix')
    stopifnot("The argument of the function should be a square matrix" = nrow(C) == ncol(C))
    stopifnot("The argument of the function is not a confusion matrix (not all elements are integer numbers)" = all(C == floor(C)))
    stopifnot("The number of classes and the number of intervals do not match" = nrow(C) == length(L))
    stopifnot("The argument L is not a lengths vector (not all its elements are positive)" = all(L > 0))
    
    # SMAE
    SMAE <- MAE.int(C, L)/MAEmax.int(C, L)$MAEmx
    SMAE
}


# Examples
M <- matrix(1:9, nrow = 3)
L <- c(2,3,6)
haus.dist(L)

MAE.int(M, L)
MAEmax.int(M, L)
SMAE.int(M, L)


##########################################################
################# Ordinal classification #################
###################### MAE Measure #######################
##########################################################

### Arguments and notation
# C             = (matrix) Confusion matrix obtained by a validation procedure
# r             = (number) Number of classes
# n             = (vector) Number of instances for each class
# N             = (number) Total number of instances
# distance      = (function) Distance between two classes given in matrix form
# m             = (matrix) Cost matrix
# index.max     = (vector) Vector composed by i_j for j=1,...,r
# Amax          = (matrix) Matrix representing the class Smax 


### Complementary functions
# Distance between two classes: |i-j|
distance <- function(C) {
    r <- nrow(C)
    d <- as.data.frame(which(C !='NA', arr.ind = T))
    d <- abs(d$row-d$col)
    d <- matrix(d, nrow = r)
    d
}


### Functions

## MAE: Mean Absolute Error
# Given a confusion matrix C, the function returns the Mean Absolute Error of C
MAE <- function(C) {
    # Check parameters
    stopifnot("The argument of the function should be a matrix" = class(C)[1] == 'matrix')
    stopifnot("The argument of the function should be a square matrix" = nrow(C) == ncol(C))
    stopifnot("The argument of the function is not a confusion matrix (not all its elements are integer numbers)" = all(C == floor(C)))
    
    # Variables
    r <- nrow(C)
    n <- apply(C, 2, sum)
    N <- sum(n)
    
    # Cost matrix m.ij
    m <- distance(C)/N
    
    # MAE
    MAE <- sum(C*m)
    MAE
}


## MAEmax = MAE(Amax) Maximum Mean Absolute Error
# Given a confusion matrix C, the function returns a list including:
# Amax      = Matrix Amax
# MAEmx     = Maximum Mean Absolute Error
MAEmax <- function(C) {
    # Check parameters
    stopifnot("The argument of the function should be a matrix" = class(C)[1] == 'matrix')
    stopifnot("The argument of the function should be a square matrix" = nrow(C) == ncol(C))
    stopifnot("The argument of the function is not a confusion matrix (not all its elements are integer numbers)" = all(C == floor(C)))
    
    # Variables
    r <-  nrow(C)
    n <- apply(C, 2, sum)
    N <- sum(n)
    
    # i.j
    index.max <- apply(distance(C), 2, which.max)
    
    # Amax
    Amax <- matrix(0, r, r)
    for (i in 1:r) {
        Amax[index.max[i],i] <- n[i]
        }
    
    # MAEmax
    MAEmx <- sum(n*abs(index.max-1:r))/N
    
    # results
    results <- list(Amax = Amax, MAEmx = MAEmx)
    results
}


# MAE: Standard Mean Absolute Error
# Given a confusion matrix C, the function returns the Standard Mean Absolute Error of C
SMAE <- function(C) {
    # Check parameters
    stopifnot("The argument of the function should be a matrix" = class(C)[1] == 'matrix')
    stopifnot("The argument of the function should be a square matrix" = nrow(C) == ncol(C))
    stopifnot("The argument of the function is not a confusion matrix (not all its elements are integer numbers)" = all(C == floor(C)))
    
    # STC
    SMAE <- MAE(C)/MAEmax(C)$MAEmx
    SMAE
}


# Examples: Matrices studied in George et al. (2016)
M <- matrix(c(444,144,2,10,37,25,2,2,27,18,8,14,57,0,7,3), nrow = 4)
M <- matrix(c(553,18,15,14,8,1,37,20,18,43,4,2,2,53,11,2), nrow = 4)
M <- matrix(c(539,19,42,0,53,3,2,8,33,24,3,7,52,0,0,15), nrow = 4)
M <- matrix(c(589,0,4,7,4,4,14,44,3,25,6,33,1,17,8,41), nrow = 4)

MAE(M)
MAEmax(M)
SMAE(M)

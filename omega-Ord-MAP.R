
################################################################################
#######.  omega-Ord-MAP criterion
################################################################################


omega.ord.MAP<-function(W=matrix, p=probabilities)  # input: the loss matrix W, 
  # the probability distribution p.
{ 
  r<-length(p)
  
  #
  stopifnot("Argument p must be a probability distribution" = all(p>=0)) 
  stopifnot("Argument p must be a probability distribution" = all.equal(sum(p),1)) 
  
  stopifnot("Argument W should be a matrix" = class(W)[1] %in% c('matrix','table') ) 
  stopifnot("Argument W should be a square matrix" = nrow(W) == ncol(W))
  stopifnot("Dimension of W and p should coincide" = nrow(W) == r)
  stopifnot("Argument W must be a loss matrix (all its elements >=0)" = all(W >= 0))
  stopifnot("Argument W must be a loss matrix (all its diagonal elements =0)" = all(diag(W) == 0))
  # 
  
  sum.left<-vector()
  sum.right<-vector()
  
  for (k in 1:(r-1))
  { sum.left[k]<-0
    sum.right[k]<-0
    for (j in 1:k)
    { sum.left[k]<-sum.left[k]+(W[j,k+1]-W[j,k])*p[j] }
    for (j in (k+1):r)
    { sum.right[k]<-sum.right[k]+(W[j,k]-W[j,k+1])*p[j] }
    
  }
  
  difference<-sum.left-sum.right
  
  if (all(difference<0))
  {h<-r} else {h<-min(which(difference>=0))}
  
  # output: k*=h, the predicted class selected by omega-Ord-MAP criterion
  
  return(h) 
  
}
  
 
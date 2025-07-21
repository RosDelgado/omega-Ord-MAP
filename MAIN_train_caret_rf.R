########################################
############  EXPERIMENTAL PHASE (Section 5)
############. Faces dataset
############
############ For tuning random forests, we use train function, from caret library
############
############ This function sets up a grid of tuning parameters for a number of 
############ classification and regression routines, fits each model and 
############ calculates a resampling based performance measure.
############ Uses "trainControl" argument from caret
###########
########################################



source("mat_square.R")
source("BD_OC_MAE.R")
source("BD_OC_MAEintervals.R")
source("omega-Ord-MAP.R")

library(arules)   # for "discretize".
library(doParallel)
registerDoParallel(cores=6)



load("faces.grey.32.Rda")  # load dataframe "db.faces.grey.32"
df<-db.faces.grey.32
str(df)

cut.points<-c(0,2,10,15,35,60,1000)

age.bin <- arules::discretize(df$age, method = "fixed", breaks=cut.points, infinity=TRUE)
table(age.bin)
df<-as.data.frame(cbind(df,age.bin))

df$age.bin<-as.factor(df$age.bin)

levels.age.int<-names(table(df$age.bin))
levels(df$age.bin)<-c("<2","[2,10)","[10,15)","[15,35)","[35,60)",">=60")
levels.age.ordinal.encod<-unique(as.numeric(df$age.bin))

par(mfrow = c(1, 1))
plot(df$age.bin,df$age)


levels.age.ordinal.encod<-unique(as.numeric(df$age.bin))
df$age.bin.num<-as.numeric(df$age.bin)
table(df$age.bin)

df$age.bin.num.factor<-as.factor(df$age.bin.num)   # we need the classes but factor type 

################################################################################
####### Error functions for summaryFunction argument, trainControl function
#######

standard.MAE.ord<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<- mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
  value<-SMAE(Conf.mat)
  c(MAE.ord=value)
}

#########
v.80<-c(0,2,10,15,35,60,80) # intervals endpoints (assume the last one is 80)

Len.80<-leng(v.80)  # intervals lengths

standard.MAE.int.80<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.80)
c(MAE.int.80 = value)
}

#########
v.90<-c(0,2,10,15,35,60,90) # intervals endpoints (assume the last one is 90)

Len.90<-leng(v.90)  # intervals lengths

standard.MAE.int.90<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.90)
c(MAE.int.90 = value)
}


#########
v.100<-c(0,2,10,15,35,60,100) # intervals endpoints (assume the last one is 100)

Len.100<-leng(v.100)  # intervals lengths

standard.MAE.int.100<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.100)
c(MAE.int.100 = value)
}

#########
v.110<-c(0,2,10,15,35,60,110) # intervals endpoints (assume the last one is 110)

Len.110<-leng(v.110)  # intervals lengths

standard.MAE.int.110<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.110)
c(MAE.int.110 = value)
}

#########
v.120<-c(0,2,10,15,35,60,120) # intervals endpoints (assume the last one is 120)

Len.120<-leng(v.120)  # intervals lengths

standard.MAE.int.120<-function(data,lev=levels.age.ordinal.encod,model=NULL)
  # data=dataframe with columns "obs" and "pred" of character/factor type
  # lev=character string with outcome factor levels
{Conf.mat<-mat.square(table(as.numeric(data$pred),as.numeric(data$obs)),lev)
value<-SMAE.int(Conf.mat,Len.120)
c(MAE.int.120 = value)
}

#
#

################################################################################
####### preparing for k-fold cross-validation with k=5
#######

N=dim(df)[1]
n=round(N/10)

set.seed(12345)
fold<-sample(c(1:10),N,replace=TRUE)
table(fold)

training<-list()
test<-list()
sub.train<-list()
#sub.test<-list()

for (i in 1:10)
{test[[i]]<-df[which(fold==i),]
training[[i]]<-df[-which(fold==i),]}

for (i in 1:10)
{set.seed(12345)
  random.sampl<-sample(which(fold!=i),2000,replace=FALSE)
  sub.train[[i]]<-df[random.sampl,]}

categories<-names(table(df[,1028]))

################################################################################
################################################################################
########## caret::train. Resampling method: cross-validation
################################################################################
library(caret)

mtry<-sqrt(ncol(df)-4)
ntree<-3

fitControl.Accuracy <- trainControl(
  method = "cv",
  number = 3,
  search="random")

fitControl.MAE <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.ord)

fitControl.MAE.int.80 <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.80)

fitControl.MAE.int.90 <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.90)

fitControl.MAE.int.100 <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.100)

fitControl.MAE.int.110 <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.110)

fitControl.MAE.int.120 <- trainControl(
  method = "cv",
  number = 3,
  ## Evaluate performance using 
  ## the following function
  summaryFunction = standard.MAE.int.120)


tuned.rf.caret.Accuracy<-list()
tuned.rf.caret.MAE<-list()
tuned.rf.caret.MAE.int.80<-list()
tuned.rf.caret.MAE.int.90<-list()
tuned.rf.caret.MAE.int.100<-list()
tuned.rf.caret.MAE.int.110<-list()
tuned.rf.caret.MAE.int.120<-list()

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.Accuracy[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                               sub.train[[i]][ ,1028], 
                                               method="rf", 
                                               metric="Accuracy",
                                               tuneLength=10,
                                               trControl=fitControl.Accuracy)
  
  print(i)}



########

################################################################################
### predictions with Ord.MAP criterion:
###

pred.test.rf.caret.Accuracy<-list()
for (i in 1:10)
{pred.test.rf.caret.Accuracy[[i]] <- predict(tuned.rf.caret.Accuracy[[i]],test[[i]],type="prob")
}


pred.Ord.MAP.rf.caret.Accuracy<-list()
for (i in 1:10)
{pred.Ord.MAP.rf.caret.Accuracy[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.Accuracy[[i]])[1])
{h<-min(which(cumsum(unlist(as.vector(pred.test.rf.caret.Accuracy[[i]][j,])))>=0.5))
pred.Ord.MAP.rf.caret.Accuracy[[i]][j]<-categories[h]
}
}

Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy<-list()
for (i in 1:10)
{Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]]<-
  table(pred.Ord.MAP.rf.caret.Accuracy[[i]],
        test[[i]][,1028])
print(i)
}

################################################################################
### predictions with omega.Ord.MAP criterion:
###
# with endopoint 80

v<-v.80
M<-haus.dist(leng(v))

pred.omega.80.MAP.rf.caret.Accuracy<-list()
for (i in 1:10)
{pred.omega.80.MAP.rf.caret.Accuracy[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.Accuracy[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.Accuracy[[i]][j,])))
pred.omega.80.MAP.rf.caret.Accuracy[[i]][j]<-categories[h]
}
}


Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy<-list()
for (i in 1:10)
{Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]]<-
  table(pred.omega.80.MAP.rf.caret.Accuracy[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))


# with endopoint 90

v<-v.90
M<-haus.dist(leng(v))

pred.omega.90.MAP.rf.caret.Accuracy<-list()
for (i in 1:10)
{pred.omega.90.MAP.rf.caret.Accuracy[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.Accuracy[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.Accuracy[[i]][j,])))
pred.omega.90.MAP.rf.caret.Accuracy[[i]][j]<-categories[h]
}
}


Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy<-list()
for (i in 1:10)
{Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]]<-
  table(pred.omega.90.MAP.rf.caret.Accuracy[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

# with endopoint 100

v<-v.100
M<-haus.dist(leng(v))

pred.omega.100.MAP.rf.caret.Accuracy<-list()
for (i in 1:10)
{pred.omega.100.MAP.rf.caret.Accuracy[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.Accuracy[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.Accuracy[[i]][j,])))
pred.omega.100.MAP.rf.caret.Accuracy[[i]][j]<-categories[h]
}
}


Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy<-list()
for (i in 1:10)
{Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]]<-
  table(pred.omega.100.MAP.rf.caret.Accuracy[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

# with endopoint 110

v<-v.110
M<-haus.dist(leng(v))

pred.omega.110.MAP.rf.caret.Accuracy<-list()
for (i in 1:10)
{pred.omega.110.MAP.rf.caret.Accuracy[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.Accuracy[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.Accuracy[[i]][j,])))
pred.omega.110.MAP.rf.caret.Accuracy[[i]][j]<-categories[h]
}
}


Conf.mat.omega.110.MAP.tuned.rf.caret.Accuracy<-list()
for (i in 1:10)
{Conf.mat.omega.110.MAP.tuned.rf.caret.Accuracy[[i]]<-
  table(pred.omega.110.MAP.rf.caret.Accuracy[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

# with endopoint 120

v<-v.120
M<-haus.dist(leng(v))

pred.omega.120.MAP.rf.caret.Accuracy<-list()
for (i in 1:10)
{pred.omega.120.MAP.rf.caret.Accuracy[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.Accuracy[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.Accuracy[[i]][j,])))
pred.omega.120.MAP.rf.caret.Accuracy[[i]][j]<-categories[h]
}
}


Conf.mat.omega.120.MAP.tuned.rf.caret.Accuracy<-list()
for (i in 1:10)
{Conf.mat.omega.120.MAP.tuned.rf.caret.Accuracy[[i]]<-
  table(pred.omega.120.MAP.rf.caret.Accuracy[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.Accuracy[[i]],categories),leng(v))



##############------------------------------------------------------------------

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                          sub.train[[i]][ ,1028], 
                                          method="rf",
                                          metric="MAE.ord",
                                          maximize=FALSE,
                                          tuneLength=10,
                                          trControl=fitControl.MAE)
  
  print(i)}


########

################################################################################
### predictions with Ord.MAP criterion:
###

pred.test.rf.caret.MAE<-list()
for (i in 1:10)
{pred.test.rf.caret.MAE[[i]] <- predict(tuned.rf.caret.MAE[[i]],test[[i]],type="prob")
}


pred.Ord.MAP.rf.caret.MAE<-list()
for (i in 1:10)
{pred.Ord.MAP.rf.caret.MAE[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE[[i]])[1])
{h<-min(which(cumsum(unlist(as.vector(pred.test.rf.caret.MAE[[i]][j,])))>=0.5))
pred.Ord.MAP.rf.caret.MAE[[i]][j]<-categories[h]
}
}

Conf.mat.Ord.MAP.tuned.rf.caret.MAE<-list()
for (i in 1:10)
{Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]]<-
  table(pred.Ord.MAP.rf.caret.MAE[[i]],
        test[[i]][,1028])
print(i)
}

################################################################################
### predictions with omega.Ord.MAP criterion:
###
# with endopoint 80

v<-v.80
M<-haus.dist(leng(v))

pred.omega.80.MAP.rf.caret.MAE<-list()
for (i in 1:10)
{pred.omega.80.MAP.rf.caret.MAE[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE[[i]][j,])))
pred.omega.80.MAP.rf.caret.MAE[[i]][j]<-categories[h]
}
}


Conf.mat.omega.80.MAP.tuned.rf.caret.MAE<-list()
for (i in 1:10)
{Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]]<-
  table(pred.omega.80.MAP.rf.caret.MAE[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

# with endopoint 90

v<-v.90
M<-haus.dist(leng(v))

pred.omega.90.MAP.rf.caret.MAE<-list()
for (i in 1:10)
{pred.omega.90.MAP.rf.caret.MAE[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE[[i]][j,])))
pred.omega.90.MAP.rf.caret.MAE[[i]][j]<-categories[h]
}
}


Conf.mat.omega.90.MAP.tuned.rf.caret.MAE<-list()
for (i in 1:10)
{Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]]<-
  table(pred.omega.90.MAP.rf.caret.MAE[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))


# with endopoint 100

v<-v.100
M<-haus.dist(leng(v))

pred.omega.100.MAP.rf.caret.MAE<-list()
for (i in 1:10)
{pred.omega.100.MAP.rf.caret.MAE[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE[[i]][j,])))
pred.omega.100.MAP.rf.caret.MAE[[i]][j]<-categories[h]
}
}


Conf.mat.omega.100.MAP.tuned.rf.caret.MAE<-list()
for (i in 1:10)
{Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]]<-
  table(pred.omega.100.MAP.rf.caret.MAE[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))


# with endopoint 110

v<-v.110
M<-haus.dist(leng(v))

pred.omega.110.MAP.rf.caret.MAE<-list()
for (i in 1:10)
{pred.omega.110.MAP.rf.caret.MAE[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE[[i]][j,])))
pred.omega.110.MAP.rf.caret.MAE[[i]][j]<-categories[h]
}
}


Conf.mat.omega.110.MAP.tuned.rf.caret.MAE<-list()
for (i in 1:10)
{Conf.mat.omega.110.MAP.tuned.rf.caret.MAE[[i]]<-
  table(pred.omega.110.MAP.rf.caret.MAE[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))


# with endopoint 120

v<-v.120
M<-haus.dist(leng(v))

pred.omega.120.MAP.rf.caret.MAE<-list()
for (i in 1:10)
{pred.omega.120.MAP.rf.caret.MAE[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE[[i]][j,])))
pred.omega.120.MAP.rf.caret.MAE[[i]][j]<-categories[h]
}
}


Conf.mat.omega.120.MAP.tuned.rf.caret.MAE<-list()
for (i in 1:10)
{Conf.mat.omega.120.MAP.tuned.rf.caret.MAE[[i]]<-
  table(pred.omega.120.MAP.rf.caret.MAE[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE[[i]],categories),leng(v))



##############------------------------------------------------------------------

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.80[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                 sub.train[[i]][ ,1028], 
                                                 method="rf", 
                                                 metric="MAE.int.80",
                                                 maximize=FALSE,
                                                 tuneLength=10,
                                                 trControl=fitControl.MAE.int.80)
  
  print(i)}



################################################################################
### predictions with Ord.MAP criterion:
###

pred.test.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{pred.test.rf.caret.MAE.int.80[[i]] <- predict(tuned.rf.caret.MAE.int.80[[i]],test[[i]],type="prob")
}


pred.Ord.MAP.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{pred.Ord.MAP.rf.caret.MAE.int.80[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.80[[i]])[1])
{h<-min(which(cumsum(unlist(as.vector(pred.test.rf.caret.MAE.int.80[[i]][j,])))>=0.5))
pred.Ord.MAP.rf.caret.MAE.int.80[[i]][j]<-categories[h]
}
}

Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]]<-
  table(pred.Ord.MAP.rf.caret.MAE.int.80[[i]],
        test[[i]][,1028])
print(i)
}


################################################################################
### predictions with omega.Ord.MAP criterion:
###
# with endopoint 80

v<-v.80
M<-haus.dist(leng(v))

pred.omega.80.MAP.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{pred.omega.80.MAP.rf.caret.MAE.int.80[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.80[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.80[[i]][j,])))
pred.omega.80.MAP.rf.caret.MAE.int.80[[i]][j]<-categories[h]
}
}


Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]]<-
  table(pred.omega.80.MAP.rf.caret.MAE.int.80[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))


# with endopoint 90

v<-v.90
M<-haus.dist(leng(v))

pred.omega.90.MAP.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{pred.omega.90.MAP.rf.caret.MAE.int.80[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.80[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.80[[i]][j,])))
pred.omega.90.MAP.rf.caret.MAE.int.80[[i]][j]<-categories[h]
}
}


Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.80[[i]]<-
  table(pred.omega.90.MAP.rf.caret.MAE.int.80[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

# with endopoint 100

v<-v.100
M<-haus.dist(leng(v))

pred.omega.100.MAP.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{pred.omega.100.MAP.rf.caret.MAE.int.80[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.80[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.80[[i]][j,])))
pred.omega.100.MAP.rf.caret.MAE.int.80[[i]][j]<-categories[h]
}
}


Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.80[[i]]<-
  table(pred.omega.100.MAP.rf.caret.MAE.int.80[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))


# with endopoint 110

v<-v.110
M<-haus.dist(leng(v))

pred.omega.110.MAP.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{pred.omega.110.MAP.rf.caret.MAE.int.80[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.80[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.80[[i]][j,])))
pred.omega.110.MAP.rf.caret.MAE.int.80[[i]][j]<-categories[h]
}
}


Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.80[[i]]<-
  table(pred.omega.110.MAP.rf.caret.MAE.int.80[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

# with endopoint 120

v<-v.120
M<-haus.dist(leng(v))

pred.omega.120.MAP.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{pred.omega.120.MAP.rf.caret.MAE.int.80[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.80[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.80[[i]][j,])))
pred.omega.120.MAP.rf.caret.MAE.int.80[[i]][j]<-categories[h]
}
}


Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.80<-list()
for (i in 1:10)
{Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.80[[i]]<-
  table(pred.omega.120.MAP.rf.caret.MAE.int.80[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.80[[i]],categories),leng(v))


##############------------------------------------------------------------------

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.90[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                 sub.train[[i]][ ,1028], 
                                                 method="rf",
                                                 metric="MAE.int.90",
                                                 maximize=FALSE,
                                                 tuneLength=10,
                                                 trControl=fitControl.MAE.int.90)
  
  print(i)}


########


################################################################################
### predictions with Ord.MAP criterion:
###

pred.test.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{pred.test.rf.caret.MAE.int.90[[i]] <- predict(tuned.rf.caret.MAE.int.90[[i]],test[[i]],type="prob")
}


pred.Ord.MAP.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{pred.Ord.MAP.rf.caret.MAE.int.90[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.90[[i]])[1])
{h<-min(which(cumsum(unlist(as.vector(pred.test.rf.caret.MAE.int.90[[i]][j,])))>=0.5))
pred.Ord.MAP.rf.caret.MAE.int.90[[i]][j]<-categories[h]
}
}

Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]]<-
  table(pred.Ord.MAP.rf.caret.MAE.int.90[[i]],
        test[[i]][,1028])
print(i)
}


################################################################################
### predictions with omega.Ord.MAP criterion:
###
# with endopoint 80

v<-v.80
M<-haus.dist(leng(v))

pred.omega.80.MAP.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{pred.omega.80.MAP.rf.caret.MAE.int.90[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.90[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.90[[i]][j,])))
pred.omega.80.MAP.rf.caret.MAE.int.90[[i]][j]<-categories[h]
}
}


Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]]<-
  table(pred.omega.80.MAP.rf.caret.MAE.int.90[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

# with endopoint 90

v<-v.90
M<-haus.dist(leng(v))

pred.omega.90.MAP.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{pred.omega.90.MAP.rf.caret.MAE.int.90[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.90[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.90[[i]][j,])))
pred.omega.90.MAP.rf.caret.MAE.int.90[[i]][j]<-categories[h]
}
}


Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]]<-
  table(pred.omega.90.MAP.rf.caret.MAE.int.90[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

# with endopoint 100

v<-v.100
M<-haus.dist(leng(v))

pred.omega.100.MAP.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{pred.omega.100.MAP.rf.caret.MAE.int.90[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.90[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.90[[i]][j,])))
pred.omega.100.MAP.rf.caret.MAE.int.90[[i]][j]<-categories[h]
}
}


Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.90[[i]]<-
  table(pred.omega.100.MAP.rf.caret.MAE.int.90[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))


# with endopoint 110

v<-v.110
M<-haus.dist(leng(v))

pred.omega.110.MAP.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{pred.omega.110.MAP.rf.caret.MAE.int.90[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.90[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.90[[i]][j,])))
pred.omega.110.MAP.rf.caret.MAE.int.90[[i]][j]<-categories[h]
}
}


Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.90[[i]]<-
  table(pred.omega.110.MAP.rf.caret.MAE.int.90[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

# with endopoint 120

v<-v.120
M<-haus.dist(leng(v))

pred.omega.120.MAP.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{pred.omega.120.MAP.rf.caret.MAE.int.90[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.90[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.90[[i]][j,])))
pred.omega.120.MAP.rf.caret.MAE.int.90[[i]][j]<-categories[h]
}
}


Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.90<-list()
for (i in 1:10)
{Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.90[[i]]<-
  table(pred.omega.120.MAP.rf.caret.MAE.int.90[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.90[[i]],categories),leng(v))





##############------------------------------------------------------------------


for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.100[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                  sub.train[[i]][ ,1028], 
                                                  method="rf", 
                                                  metric="MAE.int.100",
                                                  maximize=FALSE,
                                                  tuneLength=10,
                                                  trControl=fitControl.MAE.int.100)
  
  print(i)}

########


################################################################################
### predictions with Ord.MAP criterion:
###

pred.test.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{pred.test.rf.caret.MAE.int.100[[i]] <- predict(tuned.rf.caret.MAE.int.100[[i]],test[[i]],type="prob")
}


pred.Ord.MAP.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{pred.Ord.MAP.rf.caret.MAE.int.100[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.100[[i]])[1])
{h<-min(which(cumsum(unlist(as.vector(pred.test.rf.caret.MAE.int.100[[i]][j,])))>=0.5))
pred.Ord.MAP.rf.caret.MAE.int.100[[i]][j]<-categories[h]
}
}

Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]]<-
  table(pred.Ord.MAP.rf.caret.MAE.int.100[[i]],
        test[[i]][,1028])
print(i)
}


################################################################################
### predictions with omega.Ord.MAP criterion:
###
# with endopoint 80

v<-v.80
M<-haus.dist(leng(v))

pred.omega.80.MAP.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{pred.omega.80.MAP.rf.caret.MAE.int.100[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.100[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.100[[i]][j,])))
pred.omega.80.MAP.rf.caret.MAE.int.100[[i]][j]<-categories[h]
}
}


Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]]<-
  table(pred.omega.80.MAP.rf.caret.MAE.int.100[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

# with endopoint 90

v<-v.90
M<-haus.dist(leng(v))

pred.omega.90.MAP.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{pred.omega.90.MAP.rf.caret.MAE.int.100[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.100[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.100[[i]][j,])))
pred.omega.90.MAP.rf.caret.MAE.int.100[[i]][j]<-categories[h]
}
}


Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.100[[i]]<-
  table(pred.omega.90.MAP.rf.caret.MAE.int.100[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

# with endopoint 100

v<-v.100
M<-haus.dist(leng(v))

pred.omega.100.MAP.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{pred.omega.100.MAP.rf.caret.MAE.int.100[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.100[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.100[[i]][j,])))
pred.omega.100.MAP.rf.caret.MAE.int.100[[i]][j]<-categories[h]
}
}


Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]]<-
  table(pred.omega.100.MAP.rf.caret.MAE.int.100[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))


# with endopoint 110

v<-v.110
M<-haus.dist(leng(v))

pred.omega.110.MAP.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{pred.omega.110.MAP.rf.caret.MAE.int.100[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.100[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.100[[i]][j,])))
pred.omega.110.MAP.rf.caret.MAE.int.100[[i]][j]<-categories[h]
}
}


Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.100[[i]]<-
  table(pred.omega.110.MAP.rf.caret.MAE.int.100[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

# with endopoint 120

v<-v.120
M<-haus.dist(leng(v))

pred.omega.120.MAP.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{pred.omega.120.MAP.rf.caret.MAE.int.100[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.100[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.100[[i]][j,])))
pred.omega.120.MAP.rf.caret.MAE.int.100[[i]][j]<-categories[h]
}
}


Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.100<-list()
for (i in 1:10)
{Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.100[[i]]<-
  table(pred.omega.120.MAP.rf.caret.MAE.int.100[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.100[[i]],categories),leng(v))




##############------------------------------------------------------------------

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.110[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                  sub.train[[i]][ ,1028], 
                                                  method="rf", 
                                                  metric="MAE.int.110",
                                                  maximize=FALSE,
                                                  tuneLength=10,
                                                  trControl=fitControl.MAE.int.110)
  
  print(i)}


########


################################################################################
### predictions with Ord.MAP criterion:
###

pred.test.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{pred.test.rf.caret.MAE.int.110[[i]] <- predict(tuned.rf.caret.MAE.int.110[[i]],test[[i]],type="prob")
}


pred.Ord.MAP.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{pred.Ord.MAP.rf.caret.MAE.int.110[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.110[[i]])[1])
{h<-min(which(cumsum(unlist(as.vector(pred.test.rf.caret.MAE.int.110[[i]][j,])))>=0.5))
pred.Ord.MAP.rf.caret.MAE.int.110[[i]][j]<-categories[h]
}
}

Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]]<-
  table(pred.Ord.MAP.rf.caret.MAE.int.110[[i]],
        test[[i]][,1028])
print(i)
}


################################################################################
### predictions with omega.Ord.MAP criterion:
###
# with endopoint 80

v<-v.80
M<-haus.dist(leng(v))

pred.omega.80.MAP.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{pred.omega.80.MAP.rf.caret.MAE.int.110[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.110[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.110[[i]][j,])))
pred.omega.80.MAP.rf.caret.MAE.int.110[[i]][j]<-categories[h]
}
}


Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]]<-
  table(pred.omega.80.MAP.rf.caret.MAE.int.110[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

# with endopoint 90

v<-v.90
M<-haus.dist(leng(v))

pred.omega.90.MAP.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{pred.omega.90.MAP.rf.caret.MAE.int.110[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.110[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.110[[i]][j,])))
pred.omega.90.MAP.rf.caret.MAE.int.110[[i]][j]<-categories[h]
}
}


Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.110[[i]]<-
  table(pred.omega.90.MAP.rf.caret.MAE.int.110[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

# with endopoint 100

v<-v.100
M<-haus.dist(leng(v))

pred.omega.100.MAP.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{pred.omega.100.MAP.rf.caret.MAE.int.110[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.110[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.110[[i]][j,])))
pred.omega.100.MAP.rf.caret.MAE.int.110[[i]][j]<-categories[h]
}
}


Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.110[[i]]<-
  table(pred.omega.100.MAP.rf.caret.MAE.int.110[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))


# with endopoint 110

v<-v.110
M<-haus.dist(leng(v))

pred.omega.110.MAP.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{pred.omega.110.MAP.rf.caret.MAE.int.110[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.110[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.110[[i]][j,])))
pred.omega.110.MAP.rf.caret.MAE.int.110[[i]][j]<-categories[h]
}
}


Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.110[[i]]<-
  table(pred.omega.110.MAP.rf.caret.MAE.int.110[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

# with endopoint 120

v<-v.120
M<-haus.dist(leng(v))

pred.omega.120.MAP.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{pred.omega.120.MAP.rf.caret.MAE.int.110[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.110[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.110[[i]][j,])))
pred.omega.120.MAP.rf.caret.MAE.int.110[[i]][j]<-categories[h]
}
}


Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.110<-list()
for (i in 1:10)
{Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.110[[i]]<-
  table(pred.omega.120.MAP.rf.caret.MAE.int.110[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.110[[i]],categories),leng(v))





##############------------------------------------------------------------------

for (i in 1:10)
{ set.seed(12345)
  tuned.rf.caret.MAE.int.120[[i]] <- caret::train(sub.train[[i]][ ,-c(1025:1028)], 
                                                  sub.train[[i]][ ,1028], 
                                                  method="rf", 
                                                  metric="MAE.int.120",
                                                  maximize=FALSE,
                                                  tuneLength=10,
                                                  trControl=fitControl.MAE.int.120)
  
  print(i)}


################################################################################
### predictions with Ord.MAP criterion:
###

pred.test.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{pred.test.rf.caret.MAE.int.120[[i]] <- predict(tuned.rf.caret.MAE.int.120[[i]],test[[i]],type="prob")
}


pred.Ord.MAP.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{pred.Ord.MAP.rf.caret.MAE.int.120[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.120[[i]])[1])
{h<-min(which(cumsum(unlist(as.vector(pred.test.rf.caret.MAE.int.120[[i]][j,])))>=0.5))
pred.Ord.MAP.rf.caret.MAE.int.120[[i]][j]<-categories[h]
}
}

Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]]<-
  table(pred.Ord.MAP.rf.caret.MAE.int.120[[i]],
        test[[i]][,1028])
print(i)
}


################################################################################
### predictions with omega.Ord.MAP criterion:
###
# with endopoint 80

v<-v.80
M<-haus.dist(leng(v))

pred.omega.80.MAP.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{pred.omega.80.MAP.rf.caret.MAE.int.120[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.120[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.120[[i]][j,])))
pred.omega.80.MAP.rf.caret.MAE.int.120[[i]][j]<-categories[h]
}
}


Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]]<-
  table(pred.omega.80.MAP.rf.caret.MAE.int.120[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

# with endopoint 90

v<-v.90
M<-haus.dist(leng(v))

pred.omega.90.MAP.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{pred.omega.90.MAP.rf.caret.MAE.int.120[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.120[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.120[[i]][j,])))
pred.omega.90.MAP.rf.caret.MAE.int.120[[i]][j]<-categories[h]
}
}


Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.120[[i]]<-
  table(pred.omega.90.MAP.rf.caret.MAE.int.120[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

# with endopoint 100

v<-v.100
M<-haus.dist(leng(v))

pred.omega.100.MAP.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{pred.omega.100.MAP.rf.caret.MAE.int.120[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.120[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.120[[i]][j,])))
pred.omega.100.MAP.rf.caret.MAE.int.120[[i]][j]<-categories[h]
}
}


Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.120[[i]]<-
  table(pred.omega.100.MAP.rf.caret.MAE.int.120[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))


# with endopoint 110

v<-v.110
M<-haus.dist(leng(v))

pred.omega.110.MAP.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{pred.omega.110.MAP.rf.caret.MAE.int.120[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.120[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.120[[i]][j,])))
pred.omega.110.MAP.rf.caret.MAE.int.120[[i]][j]<-categories[h]
}
}


Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.120[[i]]<-
  table(pred.omega.110.MAP.rf.caret.MAE.int.120[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

# with endopoint 120

v<-v.120
M<-haus.dist(leng(v))

pred.omega.120.MAP.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{pred.omega.120.MAP.rf.caret.MAE.int.120[[i]]<-vector()
for (j in 1:dim(pred.test.rf.caret.MAE.int.120[[i]])[1])
{h<-omega.ord.MAP(M,unlist(as.vector(pred.test.rf.caret.MAE.int.120[[i]][j,])))
pred.omega.120.MAP.rf.caret.MAE.int.120[[i]][j]<-categories[h]
}
}


Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.120<-list()
for (i in 1:10)
{Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.120[[i]]<-
  table(pred.omega.120.MAP.rf.caret.MAE.int.120[[i]],
        test[[i]][,1028])
print(i)
}


MAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))

MAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))
SMAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.120[[i]],categories),leng(v))



####### ------------------------------------------------------------------------

################################################################################
################################################################################
################## Confusion matrices, Accuracy, SMAE and SMAE.int
################################################################################
#

########------------------------------------------------------------------------ 
########-------- For the different error functions for summaryFunction argument, 
########-------- trainControl function

########-------- METRIC: Accuracy, for Ord-MAP and omega-Ord-MAP criteria 

Accuracy.Ord.MAP.Accuracy<-vector()
Accuracy.Ord.MAP.MAE<-vector()
Accuracy.Ord.MAP.MAE.int.80<-vector() 
Accuracy.Ord.MAP.MAE.int.90<-vector() 
Accuracy.Ord.MAP.MAE.int.100<-vector() 
Accuracy.Ord.MAP.MAE.int.110<-vector() 
Accuracy.Ord.MAP.MAE.int.120<-vector() 
#
Accuracy.omega.80.MAP.Accuracy<-vector()
Accuracy.omega.90.MAP.Accuracy<-vector()
Accuracy.omega.100.MAP.Accuracy<-vector()
Accuracy.omega.110.MAP.Accuracy<-vector()
Accuracy.omega.120.MAP.Accuracy<-vector()
#
Accuracy.Ord.MAP.MAE<-vector()
Accuracy.omega.80.MAP.MAE<-vector()
Accuracy.omega.90.MAP.MAE<-vector()
Accuracy.omega.100.MAP.MAE<-vector()
Accuracy.omega.110.MAP.MAE<-vector()
Accuracy.omega.120.MAP.MAE<-vector()
#
Accuracy.Ord.MAP.MAE.int.80<-vector()
Accuracy.Ord.MAP.MAE.int.90<-vector()
Accuracy.Ord.MAP.MAE.int.100<-vector()
Accuracy.Ord.MAP.MAE.int.110<-vector()
Accuracy.Ord.MAP.MAE.int.120<-vector()
#
Accuracy.omega.80.MAP.MAE.int.80<-vector() 
Accuracy.omega.90.MAP.MAE.int.90<-vector() 
Accuracy.omega.100.MAP.MAE.int.100<-vector() 
Accuracy.omega.110.MAP.MAE.int.110<-vector() 
Accuracy.omega.120.MAP.MAE.int.120<-vector() 

for (i in 1:10)
{
  Accuracy.Ord.MAP.Accuracy[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]])
  Accuracy.Ord.MAP.MAE[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]])
  Accuracy.Ord.MAP.MAE.int.80[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]])
  Accuracy.Ord.MAP.MAE.int.90[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]])
  Accuracy.Ord.MAP.MAE.int.100[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]])
  Accuracy.Ord.MAP.MAE.int.110[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]])
  Accuracy.Ord.MAP.MAE.int.120[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]])
  ##
  Accuracy.omega.80.MAP.Accuracy[i]<-sum(diag(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]]))/sum(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]])
  Accuracy.omega.90.MAP.Accuracy[i]<-sum(diag(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]]))/sum(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]])
  Accuracy.omega.100.MAP.Accuracy[i]<-sum(diag(Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]]))/sum(Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]])
  Accuracy.omega.110.MAP.Accuracy[i]<-sum(diag(Conf.mat.omega.110.MAP.tuned.rf.caret.Accuracy[[i]]))/sum(Conf.mat.omega.110.MAP.tuned.rf.caret.Accuracy[[i]])
  Accuracy.omega.120.MAP.Accuracy[i]<-sum(diag(Conf.mat.omega.120.MAP.tuned.rf.caret.Accuracy[[i]]))/sum(Conf.mat.omega.120.MAP.tuned.rf.caret.Accuracy[[i]])
  # 
  Accuracy.Ord.MAP.MAE[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]])
  Accuracy.omega.80.MAP.MAE[i]<-sum(diag(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]]))/sum(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]])
  Accuracy.omega.90.MAP.MAE[i]<-sum(diag(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]]))/sum(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]])
  Accuracy.omega.100.MAP.MAE[i]<-sum(diag(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]]))/sum(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]])
  Accuracy.omega.110.MAP.MAE[i]<-sum(diag(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE[[i]]))/sum(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE[[i]])
  Accuracy.omega.120.MAP.MAE[i]<-sum(diag(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE[[i]]))/sum(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE[[i]])
  #
  Accuracy.Ord.MAP.MAE.int.80[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]])
  Accuracy.Ord.MAP.MAE.int.90[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]])
  Accuracy.Ord.MAP.MAE.int.100[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]])
  Accuracy.Ord.MAP.MAE.int.110[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]])
  Accuracy.Ord.MAP.MAE.int.120[i]<-sum(diag(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]]))/sum(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]])
  #
  Accuracy.omega.80.MAP.MAE.int.80[i]<-sum(diag(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]]))/sum(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]])
  Accuracy.omega.90.MAP.MAE.int.90[i]<-sum(diag(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]]))/sum(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]])
  Accuracy.omega.100.MAP.MAE.int.100[i]<-sum(diag(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]]))/sum(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]])
  Accuracy.omega.110.MAP.MAE.int.110[i]<-sum(diag(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.110[[i]]))/sum(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.110[[i]])
  Accuracy.omega.120.MAP.MAE.int.120[i]<-sum(diag(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.120[[i]]))/sum(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.120[[i]])
}


#
##
###########-------- METRIC: SMAE, for Ord-MAP and omega-Ord-MAP criteria ---
##
#
SMAE.Ord.MAP.Accuracy<-vector()
SMAE.Ord.MAP.MAE<-vector()
SMAE.Ord.MAP.MAE.int.80<-vector() 
SMAE.Ord.MAP.MAE.int.90<-vector() 
SMAE.Ord.MAP.MAE.int.100<-vector() 
SMAE.Ord.MAP.MAE.int.110<-vector() 
SMAE.Ord.MAP.MAE.int.120<-vector() 
#
SMAE.omega.80.MAP.Accuracy<-vector()
SMAE.omega.90.MAP.Accuracy<-vector()
SMAE.omega.100.MAP.Accuracy<-vector()
SMAE.omega.110.MAP.Accuracy<-vector()
SMAE.omega.120.MAP.Accuracy<-vector()
#
SMAE.omega.80.MAP.MAE<-vector()
SMAE.omega.90.MAP.MAE<-vector()
SMAE.omega.100.MAP.MAE<-vector()
SMAE.omega.110.MAP.MAE<-vector()
SMAE.omega.120.MAP.MAE<-vector()
#
SMAE.omega.80.MAP.MAE.int.80<-vector() 
SMAE.omega.90.MAP.MAE.int.90<-vector() 
SMAE.omega.100.MAP.MAE.int.100<-vector() 
SMAE.omega.110.MAP.MAE.int.110<-vector() 
SMAE.omega.120.MAP.MAE.int.120<-vector() 

for (i in 1:10)
{
  
  SMAE.Ord.MAP.Accuracy[i]<-SMAE(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod))
  SMAE.Ord.MAP.MAE.int.80[i]<-SMAE(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],levels.age.ordinal.encod))
  SMAE.Ord.MAP.MAE.int.90[i]<-SMAE(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],levels.age.ordinal.encod))
  SMAE.Ord.MAP.MAE.int.100[i]<-SMAE(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],levels.age.ordinal.encod))
  SMAE.Ord.MAP.MAE.int.110[i]<-SMAE(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],levels.age.ordinal.encod))
  SMAE.Ord.MAP.MAE.int.120[i]<-SMAE(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],levels.age.ordinal.encod))
  #
  SMAE.omega.80.MAP.Accuracy[i]<-SMAE(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod))
  SMAE.omega.90.MAP.Accuracy[i]<-SMAE(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod))
  SMAE.omega.100.MAP.Accuracy[i]<-SMAE(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod))
  SMAE.omega.110.MAP.Accuracy[i]<-SMAE(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod))
  SMAE.omega.120.MAP.Accuracy[i]<-SMAE(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod))
  #
  SMAE.omega.80.MAP.MAE[i]<-SMAE(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod))
  SMAE.omega.90.MAP.MAE[i]<-SMAE(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod))
  SMAE.omega.100.MAP.MAE[i]<-SMAE(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod))
  SMAE.omega.110.MAP.MAE[i]<-SMAE(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod))
  SMAE.omega.120.MAP.MAE[i]<-SMAE(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod))
  #
  SMAE.omega.80.MAP.MAE.int.80[i]<-SMAE(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],levels.age.ordinal.encod))
  SMAE.omega.90.MAP.MAE.int.90[i]<-SMAE(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]],levels.age.ordinal.encod))
  SMAE.omega.100.MAP.MAE.int.100[i]<-SMAE(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]],levels.age.ordinal.encod))
  SMAE.omega.110.MAP.MAE.int.110[i]<-SMAE(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.110[[i]],levels.age.ordinal.encod))
  SMAE.omega.120.MAP.MAE.int.120[i]<-SMAE(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.120[[i]],levels.age.ordinal.encod))
}

#
##
###########-------- METRICS: SMAE.INT, for Ord-MAP and omega-Ord-MAP criteria ---
##
#
SMAE.int.80.Ord.MAP.Accuracy<-vector()
SMAE.int.80.Ord.MAP.MAE<-vector()
SMAE.int.80.Ord.MAP.MAE.int.80<-vector() 
SMAE.int.80.omega.80.MAP.Accuracy<-vector()
SMAE.int.80.omega.80.MAP.MAE<-vector()
SMAE.int.80.omega.80.MAP.MAE.int.80<-vector() 
#
SMAE.int.90.Ord.MAP.Accuracy<-vector()
SMAE.int.90.Ord.MAP.MAE<-vector()
SMAE.int.90.Ord.MAP.MAE.int.90<-vector() 
SMAE.int.90.omega.90.MAP.Accuracy<-vector()
SMAE.int.90.omega.90.MAP.MAE<-vector()
SMAE.int.90.omega.90.MAP.MAE.int.90<-vector() 
#
SMAE.int.100.Ord.MAP.Accuracy<-vector()
SMAE.int.100.Ord.MAP.MAE<-vector()
SMAE.int.100.Ord.MAP.MAE.int.100<-vector() 
SMAE.int.100.omega.100.MAP.Accuracy<-vector()
SMAE.int.100.omega.100.MAP.MAE<-vector()
SMAE.int.100.omega.100.MAP.MAE.int.100<-vector() 
#
SMAE.int.110.Ord.MAP.Accuracy<-vector()
SMAE.int.110.Ord.MAP.MAE<-vector()
SMAE.int.110.Ord.MAP.MAE.int.110<-vector() 
SMAE.int.110.omega.110.MAP.Accuracy<-vector()
SMAE.int.110.omega.110.MAP.MAE<-vector()
SMAE.int.110.omega.110.MAP.MAE.int.110<-vector() 
#
SMAE.int.120.Ord.MAP.Accuracy<-vector()
SMAE.int.120.Ord.MAP.MAE<-vector()
SMAE.int.120.Ord.MAP.MAE.int.120<-vector() 
SMAE.int.120.omega.120.MAP.Accuracy<-vector()
SMAE.int.120.omega.120.MAP.MAE<-vector()
SMAE.int.120.omega.120.MAP.MAE.int.120<-vector() 

for (i in 1:10)
{

  SMAE.int.80.Ord.MAP.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),leng(v.80))
  SMAE.int.80.Ord.MAP.MAE[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),leng(v.80))
  SMAE.int.80.Ord.MAP.MAE.int.80[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.80[[i]],levels.age.ordinal.encod),leng(v.80))
  SMAE.int.80.omega.80.MAP.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),leng(v.80))
  SMAE.int.80.omega.80.MAP.MAE[i]<-SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),leng(v.80))
  SMAE.int.80.omega.80.MAP.MAE.int.80[i]<-SMAE.int(mat.square(Conf.mat.omega.80.MAP.tuned.rf.caret.MAE.int.80[[i]],levels.age.ordinal.encod),leng(v.80))
  #
  SMAE.int.90.Ord.MAP.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),leng(v.90))
  SMAE.int.90.Ord.MAP.MAE[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),leng(v.90))
  SMAE.int.90.Ord.MAP.MAE.int.90[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.90[[i]],levels.age.ordinal.encod),leng(v.90))
  SMAE.int.90.omega.90.MAP.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),leng(v.90))
  SMAE.int.90.omega.90.MAP.MAE[i]<-SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),leng(v.90))
  SMAE.int.90.omega.90.MAP.MAE.int.90[i]<-SMAE.int(mat.square(Conf.mat.omega.90.MAP.tuned.rf.caret.MAE.int.90[[i]],levels.age.ordinal.encod),leng(v.90))
  #
  SMAE.int.100.Ord.MAP.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),leng(v.100))
  SMAE.int.100.Ord.MAP.MAE[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),leng(v.100))
  SMAE.int.100.Ord.MAP.MAE.int.100[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.100[[i]],levels.age.ordinal.encod),leng(v.100))
  SMAE.int.100.omega.100.MAP.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),leng(v.100))
  SMAE.int.100.omega.100.MAP.MAE[i]<-SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),leng(v.100))
  SMAE.int.100.omega.100.MAP.MAE.int.100[i]<-SMAE.int(mat.square(Conf.mat.omega.100.MAP.tuned.rf.caret.MAE.int.100[[i]],levels.age.ordinal.encod),leng(v.100))
  #
  SMAE.int.110.Ord.MAP.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),leng(v.110))
  SMAE.int.110.Ord.MAP.MAE[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),leng(v.110))
  SMAE.int.110.Ord.MAP.MAE.int.110[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.110[[i]],levels.age.ordinal.encod),leng(v.110))
  SMAE.int.110.omega.110.MAP.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),leng(v.110))
  SMAE.int.110.omega.110.MAP.MAE[i]<-SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),leng(v.110))
  SMAE.int.110.omega.110.MAP.MAE.int.110[i]<-SMAE.int(mat.square(Conf.mat.omega.110.MAP.tuned.rf.caret.MAE.int.110[[i]],levels.age.ordinal.encod),leng(v.110))
  #
  SMAE.int.120.Ord.MAP.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),leng(v.120))
  SMAE.int.120.Ord.MAP.MAE[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),leng(v.120))
  SMAE.int.120.Ord.MAP.MAE.int.120[i]<-SMAE.int(mat.square(Conf.mat.Ord.MAP.tuned.rf.caret.MAE.int.120[[i]],levels.age.ordinal.encod),leng(v.120))
  SMAE.int.120.omega.120.MAP.Accuracy[i]<-SMAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.Accuracy[[i]],levels.age.ordinal.encod),leng(v.120))
  SMAE.int.120.omega.120.MAP.MAE[i]<-SMAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE[[i]],levels.age.ordinal.encod),leng(v.120))
  SMAE.int.120.omega.120.MAP.MAE.int.120[i]<-SMAE.int(mat.square(Conf.mat.omega.120.MAP.tuned.rf.caret.MAE.int.120[[i]],levels.age.ordinal.encod),leng(v.120))
}
  


###############################
###############################
### BOXPLOTS AND TESTS

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

### --- 80 ----------

boxplot(SMAE.int.80.Ord.MAP.Accuracy-SMAE.int.80.omega.80.MAP.Accuracy,
        SMAE.int.80.Ord.MAP.MAE-SMAE.int.80.omega.80.MAP.MAE,
        SMAE.int.80.Ord.MAP.MAE.int.80-SMAE.int.80.omega.80.MAP.MAE.int.80,
        main=paste("Right endpoint of the rightmost interval: 80"), 
          xlab="Error functions", ylab="SMAE.int increment Ord-MAP minus omega-Ord-MAP",
          names=c("Error rate", "SMAE", "SMAE.int"),
          col=c2,medcol=c3)
abline(h = 0, col = "red", lwd = 0.5)

wilcox.test(SMAE.int.80.Ord.MAP.Accuracy,SMAE.int.80.omega.80.MAP.Accuracy, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.80.Ord.MAP.MAE,SMAE.int.80.omega.80.MAP.MAE, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.80.Ord.MAP.MAE.int.80,SMAE.int.80.omega.80.MAP.MAE.int.80, paired=TRUE, alternative="greater")$p.value

### --- 90 ----------

boxplot(SMAE.int.90.Ord.MAP.Accuracy-SMAE.int.90.omega.90.MAP.Accuracy,
        SMAE.int.90.Ord.MAP.MAE-SMAE.int.90.omega.90.MAP.MAE,
        SMAE.int.90.Ord.MAP.MAE.int.90-SMAE.int.90.omega.90.MAP.MAE.int.90,
        main=paste("Right endpoint of the rightmost interval: 90"), 
        xlab="Error functions", ylab="SMAE.int increment Ord-MAP minus omega-Ord-MAP",
        names=c("Error rate", "SMAE", "SMAE.int"),
        col=c2,medcol=c3)
abline(h = 0, col = "red", lwd = 0.5)

wilcox.test(SMAE.int.90.Ord.MAP.Accuracy,SMAE.int.90.omega.90.MAP.Accuracy, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.90.Ord.MAP.MAE,SMAE.int.90.omega.90.MAP.MAE, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.90.Ord.MAP.MAE.int.90,SMAE.int.90.omega.90.MAP.MAE.int.90, paired=TRUE, alternative="greater")$p.value


### --- 100 ----------

boxplot(SMAE.int.100.Ord.MAP.Accuracy-SMAE.int.100.omega.100.MAP.Accuracy,
        SMAE.int.100.Ord.MAP.MAE-SMAE.int.100.omega.100.MAP.MAE,
        SMAE.int.100.Ord.MAP.MAE.int.100-SMAE.int.100.omega.100.MAP.MAE.int.100,
        main=paste("Right endpoint of the rightmost interval: 100"), 
        xlab="Error functions", ylab="SMAE.int increment Ord-MAP minus omega-Ord-MAP",
        names=c("Error rate", "SMAE", "SMAE.int"),
        col=c2,medcol=c3)
abline(h = 0, col = "red", lwd = 0.5)

wilcox.test(SMAE.int.100.Ord.MAP.Accuracy,SMAE.int.100.omega.100.MAP.Accuracy, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.100.Ord.MAP.MAE,SMAE.int.100.omega.100.MAP.MAE, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.100.Ord.MAP.MAE.int.100,SMAE.int.100.omega.100.MAP.MAE.int.100, paired=TRUE, alternative="greater")$p.value


### --- 110 ----------

boxplot(SMAE.int.110.Ord.MAP.Accuracy-SMAE.int.110.omega.110.MAP.Accuracy,
        SMAE.int.110.Ord.MAP.MAE-SMAE.int.110.omega.110.MAP.MAE,
        SMAE.int.110.Ord.MAP.MAE.int.110-SMAE.int.110.omega.110.MAP.MAE.int.110,
        main=paste("Right endpoint of the rightmost interval: 110"), 
        xlab="Error functions", ylab="SMAE.int increment Ord-MAP minus omega-Ord-MAP",
        names=c("Error rate", "SMAE", "SMAE.int"),
        col=c2,medcol=c3)
abline(h = 0, col = "red", lwd = 0.5)

wilcox.test(SMAE.int.110.Ord.MAP.Accuracy,SMAE.int.110.omega.110.MAP.Accuracy, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.110.Ord.MAP.MAE,SMAE.int.110.omega.110.MAP.MAE, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.110.Ord.MAP.MAE.int.110,SMAE.int.110.omega.110.MAP.MAE.int.110, paired=TRUE, alternative="greater")$p.value


### --- 120 ----------

boxplot(SMAE.int.120.Ord.MAP.Accuracy-SMAE.int.120.omega.120.MAP.Accuracy,
        SMAE.int.120.Ord.MAP.MAE-SMAE.int.120.omega.120.MAP.MAE,
        SMAE.int.120.Ord.MAP.MAE.int.120-SMAE.int.120.omega.120.MAP.MAE.int.120,
        main=paste("Right endpoint of the rightmost interval: 120"), 
        xlab="Error functions", ylab="SMAE.int increment Ord-MAP minus omega-Ord-MAP",
        names=c("Error rate", "SMAE", "SMAE.int"),
        col=c2,medcol=c3)
abline(h = 0, col = "red", lwd = 0.5)

wilcox.test(SMAE.int.120.Ord.MAP.Accuracy,SMAE.int.120.omega.120.MAP.Accuracy, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.120.Ord.MAP.MAE,SMAE.int.120.omega.120.MAP.MAE, paired=TRUE, alternative="greater")$p.value
wilcox.test(SMAE.int.120.Ord.MAP.MAE.int.120,SMAE.int.120.omega.120.MAP.MAE.int.120, paired=TRUE, alternative="greater")$p.value

####################---------------------------------###########################
####################--------  THE END ---------------###########################
####################---------------------------------###########################


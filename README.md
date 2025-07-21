# omega-Ord-MAP

R Scripts for the experimentation phase (Section 5) of the manuscript:

# Ordinal Classification with label-dependent loss

## By Rosario Delgado

____________________________________________________________________________________________

### EXPERIMENTAL EVALUATION 
We evaluate the impact of using the omega-Ord-MAP criterion as an alternative to the Ord-MAP rule introduced in Delgado, R. (2025) Ord-MAP criterion: extending MAP for Ordinal Classification. Knowledge-Based Systems,
Volume 324, 113837, ISSN 0950-7051,
https://doi.org/10.1016/j.knosys.2025.113837. 

For that, we compare their predictive performance using three metrics for hyper-parameter tuning: the error rate, the standardized Mean Absolute Error (SMAE), and the standardized interval-sensitive metric SMAE.int. We also assess the sensitivity of results to the choice of the length assigned to the rightmost interval. 

## DATASETS: 
We consider five real-world datasets:

## (a) World Values Surveys (WVS) dataset 
This dataset is sourced from the "carData" R package (Dataset to accompany J. Fox and S. Weisberg, An R Companion to Applied Regression, Third Edition, Sage (2019). https://doi.org/10.32614/CRAN.package.carData). Target variable Poverty with three categories: Too Little, About Right, Too Much. 

## (b) Wine dataset 
This dataset is available in the "ordinal" R package (https://doi.org/10.32614/CRAN.package.ordinal). The target variable is Rating, ordinal with five values, from 1 to 5. 

## (c) Hearth dataset
This dataset is included in the "ordinalForest" R package (https://doi.org/10.32614/CRAN.package.ordinalForest). The target variable Class has five ordered categories, from 1 to 5. 

## (d) Parkinson dataset 
This dataset is available from the UC Irvine Machine Learning Repository(https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring). Two target variables are considered, which are continuous and have been discretized: 
• v5:motor UPDRS (clinician’s motor Unified Parkinson’s Disease Rating Scale). This scale is a widely used score to track disease progression. Categorized to 4, 5 and 5 ordinal levels. 
• v6:total UPDRS (clinician’s total Unified Parkinson’s Disease Rating Scale). Also categorized to 4, 5 and 6 ordinal levels.  

## (e) 2011 Canadian National Election Study (CES11) dataset
This dataset is sourced from the "carData" R package (Dataset to accompany J. Fox and S. Weisberg, An R Companion to Applied Regression, Third Edition, Sage (2019). https://doi.org/10.32614/CRAN.package.carData). Target variable Importance with four categories: not, notvey, somewhat, very. 

____________________________

## Procedures
To evaluate the performance of the Ord-MAP criterion vs MAP, different procedures have been applied to any of the datasets:

## Procedure 1: 
Ordinal logistic regression using "polr" function from the "MASS" R package, with five different link functions: logistic, probit, loglog, cloglog, cauchit. 

## Procedure 2: 
Cumulative link (mixed) models (CLMs) also known as ordered regression models, proportional odds models, proportional hazards models for grouped survival times and ordered logit/probit/... models, using the "clm" function from "ordinal" R package. We consider all the combinations of link function (logit, probit, log-log, complementary log-log, and Cauchy) and threshold structure (flexible, symmetric, symmetric2, equidistant). 

## Procedure 3:
Ordinal random forest using the "ordfor" function from the "ordinalForest" R package. We fix the hyperparameters sets=1,000, ntreeperdiv=100, npermtrial=100,
ntreefinal=1,000, and explored 5 values of nbest, ranging from 8 to 12, including the default value of 10.

## Procedure 4: 
Tuning random forest models, using both Accuracy and MAE as optimization metrics. Models were trained using the "train" function from the "caret" R library.  Random forest setup: A small ensemble of three trees. Hyper-parameter tuning: Conducted via 3-fold cross-validation with a random search of 10 iterations. Custom MAE metric: While caret defaults to Accuracy as the performance metric, using the summaryFunction argument in trainControl
we introduced MAE as a custom metric, setting it as the minimization objective.

_____________________________

## Specific R scripts
The four scripts use the script mat_square.R (introduced here), which converts any matrix in a square matrix with desired row/column labels, by adding zeros if needed.

_____________________________________________________________________________________________________

### SIMULATION 
To complement both the theoretical results and experiments, in Section 6 we present a controlled simulation study aimed at analyzing the behavior of the MAP and Ord-MAP decision rules under varying levels of uncertainty.

____________________________

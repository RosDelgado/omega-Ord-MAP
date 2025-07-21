# omega-Ord-MAP

R Scripts for the experimentation phase (Section 5) of the manuscript:

# Ordinal Classification with label-dependent loss

## By Rosario Delgado

____________________________________________________________________________________________

### EXPERIMENTAL EVALUATION 
We evaluate the impact of using the omega-Ord-MAP criterion as an alternative to the Ord-MAP rule introduced in Delgado, R. (2025) Ord-MAP criterion: extending MAP for Ordinal Classification. Knowledge-Based Systems,
Volume 324, 113837, ISSN 0950-7051,
https://doi.org/10.1016/j.knosys.2025.113837. 

For that, we compare their predictive performance using three metrics for hyper-parameter tuning in random forests models: the error rate, the standardized Mean Absolute Error (SMAE), and the standardized interval-sensitive metric SMAE.int. We also assess the sensitivity of results to the choice of the length assigned to the rightmost interval. 

We consider the real-world Facial age dataset (https://www.kaggle.com/datasets/frabbisw/facial-age)


It uses the content in https://github.com/giuliabinotto/ IntervalScaleClassification, which correspond to Section 4 fot the same paper, where scripts facilitate the computation of two ordinal metrics, Mean Absolute Error (MAE) and Total Cost (TC), alongside their interval scale counterparts introduced in the paper, with a specific section designed to address scenarios in which the rightmost interval is unbounded.

## Description
From_png_to_dataframe.R
This script allows to load face files (.png) obtained from https://www.kaggle.com/datasets/frabbisw/facial-age and transform them into a dataframe, with 9,673 rows corresponding to face pictures, and 32x32+1=1025 columns, the last one being "age", while the others are the features V1,..., V1024. The dataframe is save as "faces.grey.32.Rda".

MAIN_train_caret_rf.R
This script loads "faces.grey.32.Rda" and develops the experimental phase explained in Section 5 of the paper, corresponding to the use of use of different interval-scale metrics to tuning hyper-parameter mtry for random forest using the caret library.



Requirements
General R libraries
The following libraries are needed:

magick, stringr, mdatools, png and utils (used by "From_png_to_dataframe.R")

arules (used by "train_caret_rf.R" and "tune_control_e1071.R")

caret (used by "train_caret_rf.R")

e1071 and class (used by "tune_control_e1071.R")

Specific R scripts
mat_square.R (introduced here): converts any matrix in a square matrix with desired row/column labels, by adding zeros if needed.

From https://github.com/giuliabinotto/ IntervalScaleClassification

MAE.R: computes MAE and normalized SMAE metrics.

MAEintervals.R: computes MAE.int and SMAE.int metrics.

Authors
Giulia Binotto & Rosario Delgado (Universitat Autònoma de Barcelona, Spain, 2024).

About
No description, website, or topics provided.
Resources
 Readme
License
 GPL-3.0 license
 Activity
Stars
 0 stars
Watchers
 1 watching
Forks
 0 forks
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Languages
R
100.0%
Footer
© 2025 GitHub, Inc





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

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

___________________________

## Specific R scripts
From_png_to_dataframe.R
This script allows to load face files (.png) obtained from https://www.kaggle.com/datasets/frabbisw/facial-age and transform them into a dataframe, with 9,673 rows corresponding to face pictures, and 32x32+1=1025 columns, the last one being "age", while the others are the features V1,..., V1024. The dataframe is save as "faces.grey.32.Rda".

MAIN_train_caret_rf.R
This script loads "faces.grey.32.Rda" and develops the experimental phase explained in Section 5 of the paper, corresponding to tuning random forest models, using error rate, SMAE and SMAE.int as optimization metrics. Models were trained using the "train" function from the "caret" R library.  Random forest setup: A small ensemble of three trees. Hyper-parameter tuning: Conducted via 3-fold cross-validation with a random search of 10 iterations. Custom metrics: While caret defaults to Accuracy as the performance metric, using the summaryFunction argument in trainControl
we introduced SMAE and SMAE.int as a custom metrics, setting it as the minimization objective.

omega-Ord-MAP.R
Implements the omega-Ord-MAP criterion, as alternative when the misclassification loss is label-dependent to the Ord-MAP criterion for ordinal classification. 

mat_square.R 
Converts any matrix in a square matrix with desired row/column labels, by adding zeros if needed.

From https://github.com/giuliabinotto/ IntervalScaleClassification

BD_OC_MAE.R: computes MAE and normalized SMAE metrics.

BD_OC_MAEintervals.R: computes MAE.int and SMAE.int metrics.


## Requirements
General R libraries
The following libraries are needed:

magick, stringr, mdatools, png and utils (used by "From_png_to_dataframe.R")

arules (used by "train_caret_rf.R")

caret (used by "train_caret_rf.R")


Author
Rosario Delgado (Universitat Autònoma de Barcelona, Spain, 2025).

____________________________

# Supervised Machine Learning Homework - Predicting Credit Risk

Build a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not. 

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

You will be using this data to create machine learning models to classify the risk level of given loans. Specifically, you will be comparing the Logistic Regression model and Random Forest Classifier.

### Data

Training data: Entire year's worth of data (2019)
Test data: Credit risk of loans from the first quarter of the next year (2020).

## Preprocessing: Convert categorical data to numeric

Data was converted from cateogrical to numberical using `pd.get_dummies()`. Categories in the 2019 loans that do not exist in the testing set (2020) was normalized by adding dummy variables in the testing set.

## Educated Guesses
1) Before running the unscaled data on the LogisticRegression and RandomForestClassifier models, looking at the variability of the coeffeicient, Logistic Regression would likely not be a good model as it could get confused the different coefficients; whereas, Random Tree classifier a random forest algorithm will sample the data and build several smaller, simpler decisions trees, each tree is much simpler because it is built from a subset of the data. Each tree is considered a “weak classifier” but when you combine them, they form a “strong classifier.”

2) Before running the featured selected and scaled data on the LogisticRegression and RandomForestClassifier models, RF still seems to be the better model to use based on the variability of the features. 

## Analysis
Based on the training score and testing score comparison, the RandomForestRegression model seem to overfit the data consistently. Scaling, feature selecting and then scaling the data again did not make a difference. The RandomForestRegression is not a great choice for modeling the dataset.

Based on the training and testing score comparison, the LogisticRegression model makes better predictions after the data has been scaled, featured selected and then scaled again. 

Based on the hyperparameter tuned SVC model using grid search estimator, the SVC model with Grid Search Estimated provided a slightly better classification report than with the Randomized Search Estimator. Both methods uses 2351 out of 12180 datapoints as support.

Accuracy = TP+TN/TP+FP+FN+TN
Precision = TP/TP+FP
Recall = TP/TP+FN
F1 Score = 2*(Recall * Precision) / (Recall + Precision)

![]()

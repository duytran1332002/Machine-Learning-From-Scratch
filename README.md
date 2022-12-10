# Machine-Learning-From-Scratch
# Reference:
Bayes: https://github.com/vamc-stash/Naive-Bayes/tree/master/Data

## About the Project 
In this project, I try to recode some popular Machine Learning algorithms such as K - Nearest Neighbor, Logistic Regression, Decision Tree, ... from scratch and testing in table data like Titanic data, Spambase data with main purpose having some experience about how to train a good Machine Learning model and understand clear about how model work. 

## Model and Result
In model.py contain some ML model - KNN, Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Adaboost, XGBoost, Genetic Algorithm, Linear Regression with Pytorch and Numpy.
I compared some classified model with two data set - Titanic and Spambase (data folder).

### Titanic Data

| Model | Test Accuracy | Precision - 0 | Recall - 0 | F1-Score - 0| Precision - 1 | Recall - 1 | F1-Score - 1| 
| ---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| KNN | 72% | 0.79 | 0.75 | 0.77 | 0.62 | 0.68 | 0.65 |
| Logistic Regression | 82% | 0.88 | 0.82 | 0.85 | 0.74 | 0.82 | 0.78 |
| Decision Tree | 66% | 0.78 | 0.62 | 0.69 | 0.53 | 0.71 | 0.61 |
| Random Forest | 73% | 0.75 | 0.86 | 0.8 | 0.69 | 0.53 | 0.6 |
| AdaBoost | 72% | 0.79 | 0.75 | 0.77 | 0.62 | 0.68 | 0.65 |
| XGBoost | 76% | 0.81 | 0.79 | 0.8 | 0.67 | 0.71 | 0.69 |
| Ensemble Learning (KNN, XGBoost, Logistic Regression, Random Forest) | 79% | 0.82 | 0.84 | 0.73 | 0.71 | 0.72 |

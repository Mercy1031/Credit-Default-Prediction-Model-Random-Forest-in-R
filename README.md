# Credit-Default-Prediction-Model-Random-Forest-in-R
# Project Overview

This project demonstrates a Credit Default Prediction Model using Random Forest in R. The goal is to predict whether a customer will default on a loan based on their financial and personal attributes.

This repository is ideal for a portfolio project for data analytics, machine learning, and predictive modeling.

# Table of Contents

Dataset

Problem Statement

Methodology

Data Preprocessing

Modeling

Evaluation

Variable Importance

Predicting New Applicants

Visualizations

Conclusion

# Dataset

The dataset consists of financial and personal attributes of loan applicants, including:

AMOUNT: Loan amount requested

CHK_ACCT: Checking account status

DURATION: Loan duration in months

AGE: Applicant’s age

HISTORY: Credit history

Other features: SAV_ACCT, EMPLOYMENT, INSTALL_RATE, PRESENT_RESIDENT, etc.

Target variable: DEFAULT (1 = default, 0 = no default)

Dataset is synthetic for demonstration purposes.

# Problem Statement

Predict whether a loan applicant will default, enabling financial institutions to:

Reduce credit risk

Make informed lending decisions

Prioritize high-risk applicants for additional scrutiny

# Methodology

1. Data Preprocessing

Removed ID column

Converted categorical variables to factors

Imputed missing numeric values with median

Scaled numeric variables

2. Train-Test Split

80% training, 20% testing

3. Random Forest Modeling

Used caret with 5-fold cross-validation

Tuned hyperparameters using tuneLength = 5

Evaluated using ROC-AUC and confusion matrix

4. Prediction for New Applicants

Probability threshold = 0.6 to flag high-risk applicants

# Data Preprocessing
# Convert categorical variables
refined_data$DEFAULT <- as.factor(refined_data$DEFAULT)
refined_data$EDUCATION <- as.factor(refined_data$EDUCATION)

# Impute numeric variables
refined_data <- refined_data %>%
  mutate(across(where(is.numeric),
                ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Modeling
library(caret)
set.seed(123)
rf_model <- train(
  DEFAULT ~ ., data = train_data,
  method = "rf",
  trControl = trainControl(
    method = "cv", number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  metric = "ROC",
  tuneLength = 5
)

# Evaluation

Confusion Matrix:

confusionMatrix(rf_preds, test_data$DEFAULT)

ROC & AUC:
roc_obj <- roc(test_data$DEFAULT, rf_probs)
plot(roc_obj, col="blue", main="ROC Curve - Random Forest")
auc(roc_obj)

# Variable Importance
Top predictors for credit default:
| Feature  | Importance |
| -------- | ---------- |
| AMOUNT   | 18.67      |
| CHK_ACCT | 18.64      |
| DURATION | 17.90      |
| AGE      | 15.82      |
| HISTORY  | 10.82      |

Interpretation: Loan amount, checking account, and duration are the most critical factors in predicting default.

Low importance features include FOREIGN and MALE_DIV.

# Visualization Example:
library(ggplot2)
ggplot(importance_df[1:10, ], aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat='identity', fill='steelblue') +
  coord_flip() +
  labs(title='Top 10 Important Features', x='Feature', y='Importance')

# Predicting New Applicants
new_applicants$default_prob <- predict(rf_model, newdata=new_applicants, type="prob")[,2]
new_applicants$risk_flag <- ifelse(new_applicants$default_prob > 0.6, "High Risk", "Low Risk")
head(new_applicants)

High-risk applicants can be flagged for further evaluation.

Probabilities allow fine-grained risk assessment.

# Visualizations

Variable Importance Plot

ROC Curve

Optional: Feature distributions, risk probability histograms, correlation heatmaps.

# Conclusion

Random Forest provided robust predictions for credit default.

The most influential factors are loan amount, checking account, loan duration, age, and credit history.

The model can be used to support lending decisions and reduce credit risk

# ============================================================
# Credit Default Prediction Model (Random Forest in R)
# Author: Gbenga Adeyinka
# ============================================================

# -------------------------------
# 1. Install & Load Packages
# -------------------------------
# install.packages("tidyverse")
# install.packages("caret")
# install.packages("pROC")
# install.packages("randomForest")

library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(ggplot2)

# -------------------------------
# 2. Load Dataset
# -------------------------------
credit_data <- read.csv("c:/Users/user/Desktop/CreditRisk_Data1.csv",
                        stringsAsFactors = FALSE)

cat("DATA LOADED SUCCESSFULLY\n")
str(credit_data)
summary(credit_data)

# -------------------------------
# 3. Preprocessing
# -------------------------------

# Remove ID column
refined_data <- credit_data[, -1]

# Convert categorical variables to factors
refined_data$DEFAULT   <- as.factor(refined_data$DEFAULT)
refined_data$EDUCATION <- as.factor(refined_data$EDUCATION)

# Impute missing numeric values using median
refined_data <- refined_data %>%
  mutate(across(where(is.numeric),
                ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Optional: scale numeric variables (RF does not require scaling)
numeric_vars <- sapply(refined_data, is.numeric)
scaled_data <- refined_data
scaled_data[numeric_vars] <- scale(scaled_data[numeric_vars])

# -------------------------------
# 4. Train-Test Split
# -------------------------------
set.seed(123)
train_index <- createDataPartition(refined_data$DEFAULT, p = 0.8, list = FALSE)

train_data <- refined_data[train_index, ]
test_data  <- refined_data[-train_index, ]

# Ensure factor levels match training
test_data$DEFAULT <- factor(test_data$DEFAULT,
                            levels = c(0, 1),
                            labels = c("No", "Yes"))

# -------------------------------
# 5. Random Forest Model
# -------------------------------
set.seed(123)

# Ensure training labels are consistent
train_data$DEFAULT <- factor(train_data$DEFAULT,
                             levels = c(0, 1),
                             labels = c("No", "Yes"))

# Train random forest with 5-fold CV optimizing ROC
rf_model <- train(
  DEFAULT ~ .,
  data = train_data,
  method = "rf",
  trControl = trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    savePredictions = TRUE
  ),
  metric = "ROC",
  tuneLength = 5
)

cat("RANDOM FOREST TRAINING COMPLETE\n")
print(rf_model)

# Variable importance plot
varImpPlot(rf_model$finalModel, main = "Random Forest Variable Importance")

# -------------------------------
# 6. Model Evaluation
# -------------------------------

# Predict probabilities
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[,2]

# Convert to class labels (threshold = 0.5)
rf_preds <- ifelse(rf_probs > 0.5, "Yes", "No")
rf_preds <- factor(rf_preds, levels = c("No", "Yes"))

# Confusion Matrix
cat("CONFUSION MATRIX:\n")
print(confusionMatrix(rf_preds, test_data$DEFAULT))

# ROC + AUC
roc_obj <- roc(test_data$DEFAULT, rf_probs)
plot(roc_obj, col = "blue", main = "ROC Curve - Random Forest")
auc_value <- auc(roc_obj)
cat("AUC =", auc_value, "\n")

# -------------------------------
# 7. Predicting New Applicants
# -------------------------------
new_applicants <- read.csv("c:/Users/user/Desktop/CreditRisk_Verify.csv",
                           stringsAsFactors = FALSE)

# Apply preprocessing
new_applicants <- new_applicants %>%
  mutate(across(where(is.numeric),
                ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

new_applicants$EDUCATION <- factor(new_applicants$EDUCATION,
                                   levels = levels(train_data$EDUCATION))

# Predict probabilities
new_applicants$default_prob <- predict(rf_model,
                                       newdata = new_applicants,
                                       type = "prob")[,2]

# Risk flag (threshold = 0.60)
new_applicants$risk_flag <- ifelse(new_applicants$default_prob > 0.6,
                                   "High Risk", "Low Risk")

cat("NEW APPLICANT SCORING COMPLETE\n")
head(new_applicants)

# -------------------------------
# 8. Variable Importance Table
# -------------------------------
importance_df <- data.frame(
  Feature = rownames(rf_model$finalModel$importance),
  Importance = rf_model$finalModel$importance[, "MeanDecreaseGini"]
) %>% arrange(desc(Importance))

cat("VARIABLE IMPORTANCE:\n")
print(importance_df)

ggplot(importance_df[1:10, ],
       aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat='identity', fill='steelblue') +
  coord_flip() +
  labs(title='Top 10 Important Features', x='Feature', y='Importance')


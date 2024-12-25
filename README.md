
## Telco Customer Churn Analysis and Prediction

This project analyzes customer churn data for a telecommunications company and builds a predictive model using a Random Forest classifier. The goal is to identify factors influencing churn and predict customer churn with high accuracy.

---

## Project Workflow

### 1. Install Required Packages
Install the necessary R packages for data manipulation, visualization, and modeling:
```R
install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("pROC")
```

### 2. Load Libraries
Load the required libraries for the analysis:
```R
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
```

### 3. Load the Dataset
The dataset, `WA_Fn-UseC_-Telco-Customer-Churn.csv`, is loaded from the specified directory:
```R
data <- read.csv("C:/Users/Srushti S/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

---

## Data Preprocessing

### 1. Handling Missing and Incorrect Values
- Convert `TotalCharges` to numeric.
- Remove rows with missing values.
- Ensure `Churn` and `PaymentMethod` are factors.

```R
data$TotalCharges <- as.numeric(as.character(data$TotalCharges))
data <- na.omit(data)
data$Churn <- as.factor(data$Churn)
data$PaymentMethod <- as.factor(data$PaymentMethod)
```

### 2. Exploratory Data Analysis (EDA)
Visualize the dataset to identify patterns and distributions:
- **Tenure by Churn**
- **Monthly Charges by Churn**
- **Churn by Payment Method**

Key plots include histograms and bar charts using `ggplot2`.

---

## Data Splitting and Feature Engineering

### 1. Train-Test Split
Split the data into 80% training and 20% testing sets:
```R
set.seed(123)
trainIndex <- createDataPartition(data$Churn, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
```

### 2. Create New Features
Calculate the average monthly charges as a derived feature:
```R
trainData <- trainData %>%
  mutate(avgMonthlyCharges = TotalCharges / (tenure + 1))
testData <- testData %>%
  mutate(avgMonthlyCharges = TotalCharges / (tenure + 1))
```

---

## Model Building and Evaluation

### 1. Random Forest Model
Build a Random Forest classifier with 100 trees:
```R
set.seed(123)
rfModel <- randomForest(Churn ~ ., data = trainData, ntree = 100, mtry = 3, importance = TRUE)
```

### 2. Model Summary
Print the Random Forest model summary:
```R
print(rfModel)
```

### 3. Predictions and Confusion Matrix
Predict on the test set and evaluate the model's performance:
```R
predictions <- predict(rfModel, newdata = testData)
conf_matrix <- confusionMatrix(predictions, testData$Churn)
print(conf_matrix)
```

### 4. Performance Metrics
Extract key performance metrics:
- **Accuracy**
- **Precision**
- **Recall**

```R
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
```

---

## Feature Importance

Visualize the importance of features in the model:
```R
importance <- importance(rfModel)
varImportance <- data.frame(Variables = row.names(importance), Importance = importance[, 1])

ggplot(varImportance, aes(x = reorder(Variables, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Importance")
```

---

## Results
The project produces insights into factors influencing customer churn and builds a predictive model with:
- **High accuracy**
- **Interpretability through feature importance**

---

## Future Improvements
- Explore additional models for better performance.
- Handle class imbalance if present.
- Enhance feature engineering.

---

## Dataset Source
[Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)


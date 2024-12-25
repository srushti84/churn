# Install required packages
install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("pROC")

# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)

# Load the dataset
data <- read.csv("C:/Users/Srushti S/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Handle missing and incorrect values
data$TotalCharges <- as.numeric(as.character(data$TotalCharges))  # Convert TotalCharges to numeric
data <- na.omit(data)  # Remove rows with missing values
data$Churn <- as.factor(data$Churn)  # Ensure Churn is a factor
data$PaymentMethod <- as.factor(data$PaymentMethod)  # Ensure PaymentMethod is a factor

# Check the structure of the data
str(data)

# Exploratory Data Analysis (EDA)
# Distribution of tenure by churn
ggplot(data, aes(x = tenure, fill = Churn)) +
  geom_histogram(binwidth = 1, position = "dodge") +
  labs(title = "Distribution of Tenure by Churn", x = "Tenure (months)", y = "Count")

# Distribution of monthly charges by churn
ggplot(data, aes(x = MonthlyCharges, fill = Churn)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  labs(title = "Distribution of Monthly Charges by Churn", x = "Monthly Charges", y = "Count")

# Churn distribution by payment method
ggplot(data, aes(x = PaymentMethod, fill = Churn)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Churn Distribution by Payment Method", x = "Payment Method", y = "Percentage", fill = "Churn") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Split the data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(data$Churn, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Create a new feature: average monthly charges
trainData <- trainData %>%
  mutate(avgMonthlyCharges = TotalCharges / (tenure + 1))
testData <- testData %>%
  mutate(avgMonthlyCharges = TotalCharges / (tenure + 1))

# Build the Random Forest model
set.seed(123)
rfModel <- randomForest(Churn ~ ., data = trainData, ntree = 100, mtry = 3, importance = TRUE)

# Print model summary
print(rfModel)

# Predict on the test set
predictions <- predict(rfModel, newdata = testData)

# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, testData$Churn)
print(conf_matrix)

# Extract and print accuracy, precision, and recall
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Pos Pred Value"]  # Precision for class "Yes"
recall <- conf_matrix$byClass["Sensitivity"]        # Recall for class "Yes"

cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")

# Feature Importance Visualization
importance <- importance(rfModel)
varImportance <- data.frame(Variables = row.names(importance), Importance = importance[, 1])

ggplot(varImportance, aes(x = reorder(Variables, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Importance")


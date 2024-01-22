##Question 12.1
library(caret)
library(AppliedPredictiveModeling)
data(hepatic)
injury
table(injury)
barplot(table(injury), col=c('yellow','green','blue'),main='Class Distribution')
#part c: Nearzero & Corr
bio <- bio[, -nearZeroVar(bio)]
highCorrbio <- findCorrelation(cor(bio), cutoff=0.90)
bio<- bio[, -highCorrbio]
##split the data 80/20
install.packages("MLmetrics")
set.seed(100)
trainR <- createDataPartition(injury, p=0.8, list=FALSE)
X.train <- bio[trainR, ]
y.train <- injury[trainR]
X.test <- bio[-trainR, ]
y.test <- injury[-trainR]
ctrl<-trainControl(method='LGOCV',summaryFunction = multiClassSummary,classProbs =
                     TRUE)
##Running models
#install.packages(c("glmnet", "pamr", "rms", "sparseLDA", "subselect"))
##########Logistic Regression############
X.train <- as.data.frame(X.train)
set.seed(100)
bio_lr <- caret::train(X.train,
                       y.train,
                       method ='multinom',
                       metric = "Kappa",
                       trControl = ctrl,
)
bio_lr
plot(bio_lr)
pred_bio<-predict(bio_lr,X.test)
confusionMatrix(data=pred_bio,
                reference=y.test)
reducedRoc <- roc(response = lrReduced$pred$obs,
                  predictor = lrReduced$pred$successful,
                  levels = rev(levels(lrReduced$pred$obs)))
plot(reducedRoc, legacy.axes = TRUE)
auc(reducedRoc)
######Linear Discriminant Analysis##########
library(MASS)
set.seed(100)
bio_lda <- caret::train(x = X.train,
                        y = y.train,
                        method = "lda",
                        metric = "Kappa",
                        trControl = ctrl)
bio_lda
pred_bio2 <- predict(bio_lda,X.test)
confusionMatrix(data =pred_bio2,
                reference = y.test)
######PLSDA####
set.seed(100)
plsBio <- caret::train(x = X.train,
                       y = y.train,
                       method = "pls",
                       tuneGrid = expand.grid(.ncomp = 1:15),
                       preProc = c("center","scale"),
                       metric = "Kappa",
                       trControl = ctrl)
plsBio
plot(plsBio)
predictionPLSBio <-predict(plsBio,X.test)
confusionMatrix(data =predictionPLSBio,
                reference =y.test)
##########Penalized Logistic Regression############
glmnGrid <- expand.grid(.alpha = c( .1, .2, .4, .6, .8, 1),
                        .lambda = seq(0, 2, length = 10))
require(caret)
set.seed(100)
bio_plr <- caret::train(X.train,
                        y.train,
                        method = "glmnet",
                        metric='Kappa',
                        tuneGrid = glmnGrid,
                        preProc = c("center", "scale"),
                        trControl = ctrl)
bio_plr
plot(bio_plr, main='Penalized Logistic Tuning Parameters')
pred_bio4<-predict(bio_plr,X.test)
confusionMatrix(data=pred_bio4,
                reference=y.test)
imp<-varImp(bio_plr, scale=FALSE)
plot(imp,top=5)
### Question 12.3 ####
#install.packages("caret")
#install.packages('e1071')
library(caret)
#install.packages("modeldata")
#install.packages("generics")
library(generics)
#install.packages("tidyselect")
library(tidyselect)
library(modeldata)
data("mlc_churn")
#####NearZeroVariance######
nZv <- nearZeroVar(mlc_churn)
length(nZv)
mlc_churn<-mlc_churn[,-nZv]
dim(mlc_churn)
###Separating predictors from the response####
churn1 <- mlc_churn[,-19]
str(churn1)
####Separating categorical from numerical variables
churn_cat <- churn1[,c(3,4,5)]
churn_num <- churn1[,-c(1,3,4,5)]
str(churn_cat)
str(churn_num)
# response... barplot
counts <- table(mlc_churn$churn)
counts
percentage <- prop.table(counts) * 100
percentage
bp <- barplot(counts,
              names.arg = c("(Yes)", "(No)"),
              col = c("blue", "red"),
              main = "Churn Distribution",
              xlab = "Churn",
              ylab = "Count",
              ylim = c(0, max(counts) + 10)) # Adjust y-axis limits to make space for labels
# cat... barplot
par(mfrow = c(3, 3), pin = c(2, 1))
for (col in 1:ncol(churn_cat)) {
  # Count the frequency of each category
  freq <- table(churn_cat[, col])
  barplot(freq,
          col='lightblue',
          main = colnames(churn_cat)[col],
          xlab = colnames(churn_cat)[col],
  ) # Rotates axis labels for readability
}
# num ... histogram
par(mfrow = c(3, 3), pin = c(2, 1))
for (col in 1:ncol(churn_num)) {
  col_data <- churn_num[[col]] # Changed from churn_num[, col] to churn_num[[col]]
  if (all(is.finite(col_data))) { # Check if all values are finite
    hist(col_data,
         col='lightblue',
         border='blue',
         main = colnames(churn_num)[col],
         xlab = colnames(churn_num)[col])
  } else {
    cat("Skipping column", colnames(churn_num)[col], "due to non-finite values\n")
  }
}
##outliers plot
par(mfrow = c(1, 1))
#Boxplots for all numeric variables
#numeric_data <- mlc_churn[sapply(mlc_churn, is.numeric)]
plots_per_page <- 9
for (i in seq(1, ncol(churn_num), by = plots_per_page)) {
  par(mfrow = c(3, 3))
  for (j in i:min(i+plots_per_page-1, ncol(churn_num))) {
    boxplot(churn_num[[j]],
            main = paste(names(churn_num)[j], "Distribution"),
            ylab = names(churn_num)[j])
  }
}
# corelation [cat + num]
# corr plot
correlation_matrix <- cor(churn_num)
correlation_matrix
library(corrplot)
corrplot(correlation_matrix, order = "hclust")
corrplot
#Highly correlated predictors
highCorr = findCorrelation( cor( churn_num), cutoff=0.9 )
length(highCorr)
churn_num1= churn_num[,-highCorr]
str(churn_num1)
#correlation after
cor_matrix_2 <- cor(churn_num1)
cor_matrix_2
library(corrplot)
par(mfrow = c(1,1), pin=c(2,1))
corrplot(cor_matrix_2, order = "hclust")
churn_num2 <- as.data.frame(churn_num1)
trans <- preProcess(churn_num2, method = c("BoxCox", "center", "scale", "spatialSign"))
trans
# boxplot
trfmd <- predict(trans, churn_num2)
head(trfmd)
par(mfrow = c(3, 3))
for (col in 1:ncol(trfmd)) {
  hist(trfmd[,col],
       main = colnames(trfmd)[col],
       col = 5)
}
dummRes <- dummyVars("~area_code + international_plan + voice_mail_plan",
                     data = churn_cat,
                     fullRank = TRUE)
dummy <- data.frame(predict(dummRes, newdata = churn_cat))
dummy
dim(dummy)
churn_c <- cbind(dummy, trfmd)
dim(churn_c)
str(churn_c)
###splitting####
set.seed(100)
trainR <- createDataPartition(mlc_churn$churn, p=0.8, list=FALSE)
X.train <- churn_c[trainR, ]
y.train <- mlc_churn$churn[trainR]
X.test <- churn_c[-trainR, ]
y.test <- mlc_churn$churn[-trainR]
ctrl<-trainControl(summaryFunction = twoClassSummary,classProbs = TRUE,
                   method='LGOCV',savePredictions = TRUE)
##########Logistic Regression############
require(caret)
set.seed(100)
churn_lr <- caret::train(X.train,
                         y.train,
                         method = "glm",
                         metric = "Kappa",
                         trControl = ctrl)
churn_lr
pred_churn<-predict(churn_lr,X.test)
confusionMatrix(data=pred_churn,
                reference=y.test)
# Predict probabilities on the test set
prob_predictions <- predict(churn_lr, X.test, type = "prob")[, 2]
# Compute the ROC curve
roc_obj <- roc(y.test, prob_predictions)
# Plot the ROC curve
plot(roc_obj, main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True
Positive Rate (Sensitivity)")
auc_value <- auc(roc_obj)
# Print the AUC value
print(auc_value)
######Linear Discriminant Analysis##########
library(MASS)
set.seed(100)
churn_lda <- caret::train(x = X.train,
                          y = y.train,
                          method = "lda",
                          metric = "Kappa",
                          trControl = ctrl)
churn_lda
pred_churn2 <- predict(churn_lda,X.test)
confusionMatrix(data =pred_churn2,
                reference = y.test)
# Predict probabilities on the test set
prob_predictions2 <- predict(churn_lda, X.test, type = "prob")[, 2]
# Compute the ROC curve
roc_obj2 <- roc(y.test, prob_predictions2)
# Plot the ROC curve
plot(roc_obj2, main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True
Positive Rate (Sensitivity)")
auc_value2 <- auc(roc_obj2)
# Print the AUC value
print(auc_value2)
######PLSDA##########
library(MASS)
set.seed(100)
churn_plsda <- caret::train(x = X.train,
                            y = y.train,
                            method = "pls",
                            tuneGrid = expand.grid(
                              .ncomp = 1:14),
                            metric = "Kappa",
                            trControl = ctrl)
churn_plsda
plot(churn_plsda, main='PLSDA tuning parameter')
pred_churn3 <- predict(churn_plsda,X.test)
confusionMatrix(data =pred_churn3,
                reference = y.test)
# Predict probabilities on the test set
prob_predictions3 <- predict(churn_plsda, X.test, type = "prob")[, 2]
# Compute the ROC curve
roc_obj3 <- roc(y.test, prob_predictions3)
# Plot the ROC curve
plot(roc_obj3, main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True
Positive Rate (Sensitivity)")
auc_value3 <- auc(roc_obj3)
# Print the AUC value
print(auc_value3)
##########Penalized Logistic Regression############
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
require(caret)
set.seed(100)
churn_plr <- caret::train(X.train,
                          y.train,
                          method = "glmnet",
                          metric='ROC',
                          tuneGrid = glmnGrid,
                          preProc = c("center", "scale"),
                          trControl = ctrl)
churn_plr
plot(churn_plr, main='Penalized Logistic Tuning Parameters')
pred_churn4<-predict(churn_plr,X.test)
confusionMatrix(data=pred_churn4,
                reference=y.test)
library(pROC)
# Predict probabilities on the test set
prob_predictions4 <- predict(churn_plr, X.test, type = "prob")[, 2]
# Compute the ROC curve
roc_obj4 <- roc(y.test, prob_predictions4)
# Plot the ROC curve
plot(roc_obj4, main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True
Positive Rate (Sensitivity)")
auc_value4 <- auc(roc_obj4)
# Print the AUC value
print(auc_value4)
library(mda)
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
#checking for negative values
negatives_values <- sapply(bio, function(col) any(col < 0))
print(negatives_values)
library(e1071)
skewValues <- apply(bio, 2, skewness)
head(skewValues)
#####Transformation######
bio_trans <- preProcess(bio, method = c("BoxCox", "center", "scale", "spatialSign"))
bio_trans
bio_trsfmd <- predict(bio_trans, bio)
head(bio_trsfmd)
set.seed(100)
trainR <- createDataPartition(injury, p=0.8, list=FALSE)
X.train <- bio_trsfmd[trainR, ]
y.train <- injury[trainR]
X.test <- bio_trsfmd[-trainR, ]
y.test <- injury[-trainR]
ctrl <- trainControl(summaryFunction = multiClassSummary,classProbs = TRUE,
method='LGOCV',savePredictions = TRUE)

#####Nonlinear Classification Models######

####MDA####
library(caret)
set.seed(100)
mdaFit <- caret::train(x = X.train,
                       y = y.train,
                       method = "mda",
                       metric = "Accuracy",
                       tuneGrid = expand.grid(.subclasses = 1:14),
                       trControl = ctrl)
mdaFit
plot(mdaFit)
pred_bio<-predict(mdaFit,X.test)
confusionMatrix(data=pred_bio,
                reference=y.test)



##RDA##
install.packages("rda")
install.packages("rrcov")
library(rda)
library(caret)


set.seed(100)
ctrl <- trainControl(summaryFunction = defaultSummary,
                     classProbs = TRUE)
library(rrcov)
tunegrid <- expand.grid(
  .gamma = seq(0.1, 1, by = 0.1),    # Example values for gamma
  .lambda = seq(0.01, 0.1, by = 0.01) # Example values for lambda
)
rdaFit <- caret::train(X.train, 
                y.train,
                method = "rda",
                metric = "Accuracy",
                tuneGrid = tunegrid,
                trControl = ctrl)
rdaFit
plot(rdaFit)
# prediction
predrda <- predict(rdaFit,newdata=X.test)


# confusion Matrix
confusionMatrix(predrda,y.test)




####NeuralNetworks#####
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (96+ + 1) + (maxSize+1)*3)
nnetFit <- caret::train(x = X.train,
                        y = y.train,
                        method = "nnet",
                        metric = "Accuracy",
                        preProc = c("center", "scale", "spatialSign"),
                        tuneGrid = nnetGrid,
                        trace = FALSE,
                        maxit = 2000,
                        MaxNWts = numWts,
                        trControl = ctrl)
nnetFit
plot(nnetFit)
nnetpred <- predict(nnetFit,X.test)
nnetpred
confusionMatrix(nnetpred,y.test)

####FDA####
library(mda)
library(earth)
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
fdaTuned <- caret::train(x = X.train,
                         y = y.train,
                         method = "fda",
                         metric="Accuracy",
                         tuneGrid = marsGrid,
                         trControl = trainControl(method = "cv"))
fdaTuned
plot(fdaTuned)
fdaPred <- predict(fdaTuned, newdata = X.test)
confusionMatrix(data = fdaPred,reference =y.test)


#####SVM#####
set.seed(100)
library(kernlab)
library(caret)
sigmaRangeReduced <- sigest(as.matrix(X.train[,1:96]))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^seq(-4, 14))
svmRModel <- caret::train(x = X.train,
                          y =y.train,
                          method = "svmRadial",
                          metric = "Accuracy",
                          preProc = c("center", "scale"),
                          tuneGrid = svmRGridReduced,
                          fit = FALSE,
                          trControl = ctrl)
svmRModel
plot(svmRModel)
svmPred <- predict(svmRModel, newdata = X.test)
confusionMatrix(data = svmPred,reference =y.test)


####KNN####
library(caret)
set.seed(100)
knnFit <- caret::train(x = X.train,
                       y = y.train,
                       method = "knn",
                       metric = "Accuracy",
                       preProc = c("center", "scale"),
                       ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                       tuneGrid = data.frame(.k = 1:50),
                       trControl = ctrl)
knnFit
plot(knnFit)
knnpred <- predict(knnFit,X.test)
knnpred
confusionMatrix(knnpred,
                y.test)
######Naive Bayes#####
#install.packages("klaR")
# Create a tuning grid for Naive Bayes
nbGrid <- expand.grid(
  .fL =c(2,3,4),
  .usekernel = TRUE,
  .adjust = TRUE
)
library(klaR)
set.seed(100)
nbFit <- caret::train( x = X.train,
                       y = y.train,
                       method = "nb",
                       metric = "Accuracy",
                       ## preProc = c("center", "scale"),
                       ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                       tuneGrid = nbGrid,
                       trControl = ctrl)
nbFit
plot(nbFit)
nbPred <- predict(nbFit, newdata = X.test)
confusionMatrix(data = nbPred,reference =y.test)

###Important Predictors###


knn_ImpVals=varImp(knnFit)
knn_ImpVals
# top 5. 

plot(knn_ImpVals, 
     top = 5, 
     scales = list(y = list(cex = .95)),
     col = "blue",
     main="KNN: Top 5 Important Predictors "
)


## Question 7.1

set.seed(100)
x <- runif(100, min = 2, max = 10)
y <- sin(x) + rnorm(length(x)) * .25
sinData <- data.frame(x = x, y = y)
plot(x, y)

## Create a grid of x values to use for prediction
dataGrid <- data.frame(x = seq(2, 10, length = 100))

points(x = dataGrid$x, y = modelPrediction[,1],
       type = "l", col = "blue")
#part a
install.packages("kernlab")
library(kernlab)

# Fit the SVM models with different C and epsilon values
C_values <- c(0.01, 1, 10)  
epsilon_values <- c(0.01,0.1,1)


# Set the plotting area to a 3x3 layout
par(mfrow=c(3, 3))

# Loop over the C and epsilon values
for (C_value in C_values) {
  for (epsilon_value in epsilon_values) {
    # Fit the SVM model
    rbfSVM <- ksvm(x = x, y = y, data = sinData,
                   kernel = "rbfdot", kpar = "automatic",
                   C = C_value, epsilon = epsilon_value)
    
    # Generate predictions from the model
    modelPrediction <- predict(rbfSVM, newdata = dataGrid)
    
    # Plot the original data
    plot(sinData$x, sinData$y, main = paste("C =", C_value, "Epsilon =", epsilon_value))
    
    # Add the model predictions to the plot
    lines(dataGrid$x, modelPrediction, col = "blue", lwd = 2)
  }
}


#part b

# Define a range of sigma values to test

sigma_values <- c(0.01, 0.1, 1, 10,100)
par(mfrow=c(3, 2))

for (sigma_value in sigma_values) {
  rbfSVM <- ksvm(x = x, y = y, data = sinData,
                 kernel = "rbfdot", kpar = list(sigma = sigma_value),
                 C = 1, epsilon = 0.1)
  modelPrediction <- predict(rbfSVM, newdata = dataGrid)
  plot(sinData$x, sinData$y, main = paste("C =",1, "Epsilon =", 
                                             0.1, "Sigma =", sigma_value)) 
  lines(dataGrid$x, modelPrediction, col = "blue", lwd = 2)
}
  
  
## Question 7.2
library(caret)
library(mlbench)
set.seed(200)
trainingData <- mlbench.friedman1(200, sd = 1)
 
## We convert the 'x' data from a matrix to a data frame

trainingData$x <- data.frame(trainingData$x)

featurePlot(trainingData$x, trainingData$y)
## or other methods.
## This creates a list with a vector 'y' and a matrix
## of predictors 'x'. Also simulate a large test set to
## estimate the true error rate with good precision:
testData <- mlbench.friedman1(5000, sd = 1)
testData$x <- data.frame(testData$x)


## Model Tuning
# Part a

##KNN
library(caret)
knnModel <- train(x = trainingData$x,
                    y = trainingData$y,
                    method = "knn",
                    preProc = c("center", "scale"),
                    tuneLength = 10)
knnModel

knnPred <- predict(knnModel, newdata = testData$x)
## The function 'postResample' can be used to get the test set performance values
postResample(pred = knnPred, obs = testData$y)
plot(knnModel, metric='RMSE')

##MARS
install.packages( "earth")
library(earth)
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:50)  
## Change 38 to 50

set.seed(100)

# tune
marsTune <- train(trainingData$x, trainingData$y,
                  method = "earth",
                  tuneGrid = marsGrid,
                  trControl = trainControl(method = "cv"))

marsTune
plot(marsTune)
marsPred <- predict(marsTune, testData$x)
postResample(marsPred, testData$y)


#part b
varImp(marsTune)
plot(varImp(marsTune))



##Question 7.5
library(AppliedPredictiveModeling)
data(ChemicalManufacturingProcess)

# imputation
miss <- preProcess(ChemicalManufacturingProcess, method = "knnImpute")
Chemical <- predict(miss, ChemicalManufacturingProcess)

# filtering low frequencies
Chemical <- Chemical[, -nearZeroVar(Chemical)]

set.seed(624)

# index for training
index <- createDataPartition(Chemical$Yield, p = .8, list = FALSE)

# train 
train_x <- Chemical[index, -1]
train_y <- Chemical[index, 1]

# test
test_x <- Chemical[-index, -1]
test_y <- Chemical[-index, 1]

# remove predictors to ensure maximum abs pairwise corr between predictors < 0.75
tooHigh <- findCorrelation(cor(train_x), cutoff = .90)

# removing 21 variables
train_x_nnet <- train_x[, -tooHigh]
test_x_nnet <- test_x[, -tooHigh]

#Part a
##KNN
knnModel <- train(train_x_nnet, train_y,
                  method = "knn",
                  preProc = c("center", "scale"),
                  tuneLength = 10, 
                  tuneGrid = data.frame(.k=1:20))

knnModel
knnPred <- predict(knnModel, test_x_nnet)
## The function 'postResample' can be used to get the test set
## perforamnce values
postResample(pred = knnPred, test_y)
plot(knnModel)

##MARS
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:50)
set.seed(100)
# tune
marsTune <- train(train_x_nnet, train_y,
                  method = "earth",
                  tuneGrid = marsGrid,
                  trControl = trainControl(method = "cv"))

marsTune

marsPred <- predict(marsTune, test_x_nnet)

postResample(marsPred, test_y)
plot(marsTune)

## SVM
set.seed(100)
# tune
svmRTune <- train(train_x_nnet, train_y,
                  method = "svmRadial",
                  preProc = c("center", "scale"),
                  tuneLength = 14,
                  trControl = trainControl(method = "cv"))

svmRTune

svmRPred <- predict(svmRTune, test_x_nnet)

postResample(svmRPred, test_y)
plot(svmRTune)

##NN
nnetGrid <- expand.grid(.decay = c(0.001, 0.01, .1,1),
                        .size = c(1:10),
                        ## The next option is to use bagging (see the
                        ## next chapter) instead of different random
                        ## seeds.
                        .bag = FALSE)

nnetTune2 <- train(train_x_nnet, train_y,
                   method = "avNNet",
                   tuneGrid = nnetGrid,
                   trControl = trainControl(method = "cv"),
                   linout = TRUE,
                   trace = FALSE,
                   MaxNWts = 10 * (ncol(train_x_nnet) + 1) + 10 + 1,
                   maxit = 500
)
nnetTune2
nnPred <- predict(nnetTune2, test_x_nnet)

postResample(nnPred, test_y)
plot (nnetTune2)

rbind(knn = postResample(knnPred, test_y),
      nn = postResample(nnPred, test_y),
      mars = postResample(marsPred, test_y),
      svmR = postResample(svmRPred, test_y))


#part b
varImp(svmRTune)
plot(varImp(svmRTune), top = 20,
     main = "svmRTune Top 20 Important Predictors")
#part c


enetGrid <- expand.grid(.lambda = c(0,0.1, by=.01),
                        .fraction = seq(.05, 1, length = 20))
set.seed(100)
enetTune <- train(x = train_x_nnet,
                  y = train_y,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"))
enetTune
plot(enetTune, metric='RMSE')


prediction4 <- predict(enetTune, newdata = test_x_nnet)

postResample(prediction4,test_y)

plot(enetTune, metric='RMSE')


plot(varImp(svmRTune), top = 10,
     main = "Nonlinear: Top 10 Important Predictors")

plot(varImp(enetTune), top = 10,
     main = "Linear: Top 10 Important Predictors")







#Question 6.1
#Part a
library(caret)
data(tecator)
head(absorp)
head(endpoints)
#part b

#scale and center data prior to PCA
pcaO<- prcomp(absorp, center=TRUE, scale=TRUE)

# cumulative percentage of variance for each component
percentV<-pcaO$sdev^2/sum(pcaO$sdev^2)*100
head(percentV)

screeplot(pcaO, npcs=10, type='lines', main='Scree Plot for PCA analysis')

#part c
c<-table(endpoints[,2])
barplot(c, main='Fat Distribution',xlab='Percentage of Fat')


set.seed(100)
trainR<-createDataPartition(endpoints[,2], p=0.8, list=FALSE)
trainAbs<-absorp[trainR,]
testAbs<-absorp[-trainR,]
trainF<-endpoints[trainR,2]
testF<-endpoints[-trainR,2]
str(trainAbs)
trainAbs<-as.data.frame(trainAbs)

ctrl<-trainControl(method='cv', number=3)

pcrm<-train(x=trainAbs, y=trainF, method='pcr',
            trControl=ctrl, tuneLength=20, preProc=c('center','scale'))
print(pcrm)
plot(pcrm)

#Average RMSE
#rmsev<-pcrm$results$RMSE
# av<-mean(rmsev)
# av

#part d
#prediction<-predict(pcrm, testAbs)
# part d
# Predict the response for the test set
test_pred <- predict(pcrm, newdata =as.data.frame(testAbs))
test_pred
rmse <- sqrt(mean((testF - test_pred)^2))
rmse

postResample(testF, test_pred)

#Question 6.2

#part a
library(AppliedPredictiveModeling)
data(permeability)
head(permeability)
summary(permeability)
str(permeability)

#part b
dim(fingerprints)

fingerprints_F<-fingerprints[,-nearZeroVar(fingerprints)]
dim(fingerprints_F)

#part c
set.seed(100)
#index for training
trainFp<-createDataPartition(permeability, p=0.8, list=FALSE)

#train
trainP_y<-permeability[trainFp,]
trainF_X<-fingerprints_F[trainFp,]

#test
testP_y<-permeability[-trainFp,]
testF_X<-fingerprints_F[-trainFp,]

trainF_X<-as.data.frame(trainF_X)


#cv 
ctrl<-trainControl(method='cv', number=10)
plsT <- train(x=trainF_X,y=trainP_y, method = "pls", metric = "Rsquared",
                 tuneLength = 20, trControl = ctrl,
              preProc = c("center", "scale"))

#plsR<-plsT$results
plsT

plot(plsT)

# set.seed(1)
# ctrl<-trainControl(method='cv',number=10)
# plsT<-train(x=trainF, y=trainP, method='pls', tuneLength =20,
#             trControl =ctrl, preProc=c('center', 'scale') )
# print(plsT)
# plot(plsT, metric='Rsquared', main='PLS tuning parameter for permeability Data')


#part d

#predict 
pls_pred<-predict(plsT, newdata=testF_X)
postResample(pls_pred,testP_y)


#part 6.3

#part a
library(AppliedPredictiveModeling)
data("ChemicalManufacturingProcess")


library(caret)
sum(is.na(ChemicalManufacturingProcess))
missing <- preProcess(ChemicalManufacturingProcess, method =c("knnImpute"))
df.chem<- predict(missing, ChemicalManufacturingProcess)
sum(is.na(df.chem))



colYield <-which(colnames(ChemicalManufacturingProcess) == "Yield")

X<-as.data.frame(df.chem[,-colYield])
Y<-df.chem[,colYield]
dim(X)
# checking neg
neg_cols_c = 0
for (col in 1:ncol(X)) {  
  neg_cols_c = neg_cols_c + length(which(X[,col] < 0))
}
neg_cols_c

pos_c = 0
for (col in 1:ncol(X)) {  
  pos_c = pos_c + length(which(X[,col] >= 0))
}

pos_c
pos_c + neg_cols_c
nrow(X)*ncol(X)

# stores indexes of max value  
max = which(X == max(X), arr.ind = TRUE)   
print(paste("Maximum value: ", X[max])) 
print(max)

# stores indexes of min value  
min = which(X == min(X), arr.ind = TRUE)   
print(paste("Maximum value: ",X[min])) 
print(min)

X[22,30]

head(X[,1])
head(X[,1]+10)

pos_val = 15
for (col in 1:ncol(X)) {  
  X[,col] =X[,col] + pos_val
}
tail(X[,1])

X
dim(X)


ctrl <- trainControl(method = "cv", number = 3)


trainR <- createDataPartition(ChemicalManufacturingProcess$Yield, p=0.8, list=FALSE)

x.train <- X[trainR,]
y.train <- Y[trainR]

x.test <- X[-trainR,]
y.test <- Y[-trainR]
dim(x.train)

#ridge

ridgeGrid <- data.frame(.lambda = seq(0, 0.1, length = 20))

ridgeTune <- train(x = x.train,
                   y = y.train,
                   method = "ridge",
                   tuneGrid = ridgeGrid,
                   tuneLength = 20,
                   trControl = ctrl,
                   preProc = c("center", "scale"))

ridgeTune

predictions <- predict(ridgeTune, newdata = x.test)

postResample(predictions,y.test)

plot(ridgeTune, metric='RMSE')



imp<-varImp(enetTune, scale=FALSE)
plot(imp,top=20)
#enet

enetGrid <- expand.grid(.lambda = c(0,0.1, by=.01),
                        .fraction = seq(.05, 1, length = 20))
set.seed(100)
enetTune <- train(x = x.train,
                  y = y.train,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"))
enetTune
plot(enetTune, metric='RMSE')


prediction4 <- predict(enetTune, newdata = x.test)

postResample(prediction4,y.test)

plot(enetTune, metric='RMSE')


dev.off()

# lasso

library(elasticnet)
set.seed(1)
LassoGrid <- expand.grid(.lambda = 0, .fraction = seq(0.001, 1, length = 20))
set.seed(1)
lassoTune <- train(x.train, y.train, method = "enet",
                   tuneGrid = LassoGrid, trControl = ctrl,
                   preProc = c("center", "scale"))
lassoTune


prediction6 <- predict(lassoTune, newdata = x.test)

postResample(prediction6,y.test)

plot(lassoTune, metric='RMSE')


#lm

nzvpp <- nearZeroVar(X)
nzvpp
df <- X[-nzvpp]
dim(df)

highCorrpp <- findCorrelation(cor(X), cutoff=0.80)
df<- X[, -highCorrpp]

# preP <- preProcess(misX, method = c("BoxCox"))
# df <- predict(preP,misX)


library(caret)

set.seed(100)
trainR <- createDataPartition(Y, p=0.8, list=FALSE)
X.train <- df[trainR, ]
y.train <- Y[trainR]
X.test <- df[-trainR, ]
y.test <- Y[-trainR]



ctrl <- trainControl(method = "cv", number = 3)

#linear model 

set.seed(100)
lm_model <- train(X.train, y.train, method = "lm", 
                  trControl = ctrl)
lm_model


prediction7 <- predict(lm_model, newdata = X.test)

postResample(prediction7,y.test)

xyplot(y.train ~ predict(lm_model),
       ## plot the points (type = 'p') and a background grid ('g')
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Observed")




xyplot(resid(lm_model) ~ predict(lm_model),
       type = c("p", "g"),
       xlab = "Predicted", ylab = "Residuals")




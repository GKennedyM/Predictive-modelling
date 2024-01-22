library(AppliedPredictiveModeling)
data(ChemicalManufacturingProcess)

#Part a
library(caret)
any_missing <- any(is.na(ChemicalManufacturingProcess))
any_missing
yield_missing <- any(is.na(ChemicalManufacturingProcess$Yield))
yield_missing
cleaned <- na.omit(ChemicalManufacturingProcess)

set.seed(1)
pls <- train(Yield ~ .,
              data = cleaned,
              method = "pls",
              preProc = c("center", "scale"),
              tuneLength = 10,
              trControl = trainControl(method = "repeatedcv", 
                                            repeats = 5))
pls


R2values <- pls$results[, c("ncomp", "Rsquared", "RsquaredSD")]
R2values$RsquaredSEM <- R2values$RsquaredSD/sqrt(length(pls$control$index))

R2values


library(ggplot2)
oneSE <- ggplot(R2values,
                  ## Create an aesthetic mapping that plots the
                  ## number of PLS components versus the R^2
                  ## values and their one SE lower bound
                  aes(ncomp, Rsquared,
                        ymin = Rsquared - RsquaredSEM,
                        ## don't add anything here to get
                        ## a one-sided interval plot
                      ymax=Rsquared))
## geom_linerange shoes the interval and geom_pointrange
## plots the resampled estimates.
oneSE + geom_linerange() + geom_pointrange() + theme_bw()




bestR2 <- subset(R2values, ncomp == which.max(R2values$Rsquared))
bestR2$lb <- bestR2$Rsquared - bestR2$RsquaredSEM
candR2 <- subset(R2values, Rsquared >= bestR2$lb & ncomp < bestR2$ncomp)
bestR2$lb
candR2


bestR2 <- subset(R2values, ncomp == which.max(R2values$Rsquared))
R2values$tolerance <- (R2values$Rsquared - bestR2$Rsquared)/bestR2$Rsquared * 100
R2values

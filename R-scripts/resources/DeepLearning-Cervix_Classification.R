#-------------------------------------------------------------------------------
# Sistemas Inteligentes para Gestión de la Empresa
# Curso 2016-2017
# Departamento de Ciencias de la Computación e Inteligencia Artificial
# Universidad de Granada
#
# Juan Gómez-Romero (jgomez@decsai.ugr.es)
# Francisco Herrera Trigueros (herrera@decsai.ugr.es)
#
# Example of Kaggle cervix challenge with Microsoft R using features and caret method="rf"
# https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening
#
# Assuming previous feature extraction (can be done with 'Featurize-Cervix.R':
# ./train-featuremap.xdf <-- feature maps of images used for training
# ./test-featuremap.xdf <-- feature maps of images used for test
# ./submission-featuremap.xdf <-- feature maps of images used for submission
#
# Multi-classification with caret + Random Forests
# Custom LogLoss function
# Saves classification model
# Submission file for Kaggle competition is created
#-------------------------------------------------------------------------------

# Clear workspace
rm(list=ls())

#-------------------------------------------------------------------------------
# Load feature maps
#-------------------------------------------------------------------------------
train_xdf <- rxImport("./train-featuremap.xdf")
train_df  <- train_xdf[, -grep("Pixel", colnames(train_xdf))]
train_df$Label <- as.factor(paste('Class', train_df$Label, sep = ''))
train_df$Path <- NULL

test_xdf  <- rxImport("./test-featuremap.xdf")
test_df   <- test_xdf[, -grep("Pixel", colnames(test_xdf))]
test_df$Label <- as.factor(paste('Class', test_df$Label, sep = ''))
test_df$Path <- NULL

#-------------------------------------------------------------------------------
# Create classificacion models
#-------------------------------------------------------------------------------

# -- Define custom logloss function
# -- from: https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/13064#69102
LogLosSummary <- function (data, lev = NULL, model = NULL) {
  LogLos <- function(actual, pred, eps = 1e-15) {
    stopifnot(all(dim(actual) == dim(pred)))
    pred[pred < eps] <- eps
    pred[pred > 1 - eps] <- 1 - eps
    -sum(actual * log(pred)) / nrow(pred) 
  }
  if (is.character(data$obs)) data$obs <- factor(data$obs, levels = lev)
  pred <- data[, "pred"]
  obs <- data[, "obs"]
  isNA <- is.na(pred)
  pred <- pred[!isNA]
  obs <- obs[!isNA]
  data <- data[!isNA, ]
  cls <- levels(obs)
  
  if (length(obs) + length(pred) == 0) {
    out <- rep(NA, 2)
  } else {
    pred <- factor(pred, levels = levels(obs))
    require("e1071")
    out <- unlist(e1071::classAgreement(table(obs, pred)))[c("diag", "kappa")]
    
    probs <- data[, cls]
    actual <- model.matrix(~ obs - 1)
    out2 <- LogLos(actual = actual, pred = probs)
  }
  out <- c(out, out2)
  names(out) <- c("Accuracy", "Kappa", "LogLoss")
  
  if (any(is.nan(out))) out[is.nan(out)] <- NA 
 
  out
}

# -- Classification (and prediction) with Random Forest
library(caret)
rfCtrl  <- trainControl(method = "repeatedcv", number = 3, repeats = 10, classProbs = TRUE, savePredictions = TRUE, summaryFunction = LogLosSummary)
rfModel <- train(Label ~ ., data = train_df, method = "rf", allowParallel = TRUE, metric = "LogLoss", trControl = rfCtrl)
plot(rfModel)
plot(rfModel$finalModel)
saveRDS(object = model, file = "CervixModel_RandomForest.rds") # save model for further use
rfPrediction<- predict(rfModel, test_df, type = "raw") # type = "prob" for probabilities
confusionMatrix(rfPrediction, test_df$Label)

# -- Boosting with xgboost
# @todo

#-------------------------------------------------------------------------------
# Create submission
#-------------------------------------------------------------------------------
model <- rfModel

# -- Load feature maps
submission_xdf <- rxImport("./submission-featuremap.xdf")
submission_df  <- submission_xdf[, -grep("Pixel", colnames(submission_xdf))]

# -- Apply prediction model
submission_pred <- predict(model, submission_df, type = "prob")

# -- Create submission file
submission_pred$Path <- sapply(submission_df$Path, function(x) strapplyc(x, ".*/(.*.jpg)")[[1]])
submission_pred <- submission_pred[, c(4, 1, 2, 3)]
colnames(submission_pred) <- c('image_name', 'Type_1', 'Type_2', 'Type_3')
write.csv(submission_pred, "./submission_alexnet_features_rf.csv", row.names = FALSE, quote = FALSE)

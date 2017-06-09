#-------------------------------------------------------------------------------
# Sistemas Inteligentes para Gestión de la Empresa
# Curso 2016-2017
# Departamento de Ciencias de la Computación e Inteligencia Artificial
# Universidad de Granada
#
# Juan Gómez-Romero (jgomez@decsai.ugr.es)
# Francisco Herrera Trigueros (herrera@decsai.ugr.es)
#
# Example of Kaggle cervix challenge with Microsoft R using featurizeImage and rxLogisticRegression
# https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening
#
# Generates feature files:
# ./train-featuremap.xdf <-- feature maps of images used for training
# ./test-featuremap.xdf <-- feature maps of images used for test
# ./submission-featuremap.xdf <-- feature maps of images used for submission
#
# Multi-classification with rxLogisticRegression
# Submission file for Kaggle competition is created
#-------------------------------------------------------------------------------

# Clear workspace
rm(list=ls())

#-------------------------------------------------------------------------------
# Prepare image data
#-------------------------------------------------------------------------------

# -- Build train data frame, with "file name" and "label"
train_img_path <- "../r-workspace/train/"
img_file_list_train <- list.files(path = train_img_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)

options(stringsAsFactors = FALSE)
train <- data.frame()
n_img_train <- length(img_file_list_train)
library(gsubfn)
for(i in 1:n_img_train) {
  path <- img_file_list_train[i]
  label <- strapplyc(img_file_list_train[i], ".*/Type_(.*)/")[[1]]
  train <- rbind(train, c(path, label))
}
colnames(train) <- c('Path', 'Label')

# -- Build submission data frame, with "file name" and no "label"
submission_img_path <- "../r-workspace/test/"
img_file_list_submission <- list.files(path = submission_img_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)

submission <- data.frame()
n_img_submission <- length(img_file_list_submission)
for(i in 1:n_img_submission) {
  path <- img_file_list_submission[i]
  submission <- rbind(submission, c(path))
}
colnames(submission) <- c('Path')
options(stringsAsFactors = TRUE)

#-------------------------------------------------------------------------------
# Train neural network based on ResNet
#-------------------------------------------------------------------------------

# -- Split train in train + test
library(caret)
set.seed(1)
train_index <- createDataPartition(train$Label, times = 1, p = 0.9, list = FALSE)
train_for_model <- train[ train_index, ]
test_for_model  <- train[-train_index, ]

# -- Featurize train data set
rxFeaturize(
  data = train_for_model, 
  outData = "train-featuremap.xdf", # save on disk (if set to NULL, a data frame is returned)
  overwrite = TRUE,
  mlTransforms = 
    list(loadImage(vars = list(Image = "Path")),
         resizeImage(vars = "Image", width = 227, height = 227, resizingOption = "IsoPad"),
         extractPixels(vars = list(Pixels = "Image")),
         featurizeImage(var = "Pixels", outVar = "Feature", dnnModel = "alexnet")),
  mlTransformVars = c("Path", "Label"))

# -- Featurize test data set
rxFeaturize(
  data = test_for_model, 
  outData = "test-featuremap.xdf", # save on disk (if set to NULL, a data frame is returned)
  overwrite = TRUE,
  mlTransforms = 
    list(loadImage(vars = list(Image = "Path")),
         resizeImage(vars = "Image", width = 227, height = 227, resizingOption = "IsoPad"),
         extractPixels(vars = list(Pixels = "Image")),
         featurizeImage(var = "Pixels", outVar = "Feature", dnnModel = "alexnet")),
  mlTransformVars = c("Path", "Label"))

# -- Build classification formula with features [0, 4095] 
varInfo  <- rxGetVarInfo("train-featuremap.xdf")

features <- paste("Feature", 0:4095, sep=".", collapse = " + ")
form  <- as.formula(paste("Label", features, sep="~"))

# -- Create model (logistic regression)
model <- rxLogisticRegression(
  formula = form,
  type = "multiClass",
  data = "train-featuremap.xdf")

saveRDS(object = model, file = "CervixModel_LogisticRegression.rds") # save model for further use

#-------------------------------------------------------------------------------
# Test results
#-------------------------------------------------------------------------------
summary(model)

# -- Predict with test subset extracted from train
score1 <- rxPredict(model, data = "test-featuremap.xdf", extraVarsToWrite = "Label")

#-------------------------------------------------------------------------------
# Create submission
#-------------------------------------------------------------------------------

# -- Featurize submission data set
rxFeaturize(
  data = submission, 
  outData = "submission-featuremap.xdf", # save on disk (if set to NULL, a data frame is returned)
  overwrite = TRUE,
  mlTransforms = 
    list(loadImage(vars = list(Image = "Path")),
         resizeImage(vars = "Image", width = 227, height = 227, resizingOption = "IsoPad"),
         extractPixels(vars = list(Pixels = "Image")),
         featurizeImage(var = "Pixels", outVar = "Feature", dnnModel = "alexnet")),
  mlTransformVars = c("Path", "Label"))

# -- Predict with submission dataset (add Label column)
rxDataStep(
  inData = "./submission-featuremap.xdf",
  outFile = "./submission-featuremap-temp.xdf",
  transforms = list(Label = "?"), 
  overwrite = TRUE)
score2 <- rxPredict(model, data = "submission-featuremap-temp.xdf", extraVarsToWrite = c("Path"))

# -- Generate submission file for regression model
submission_df <- score2
submission_df$Path <- sapply(submission_df$Path, function(x) strapplyc(x, ".*/(.*.jpg)")[[1]])
submission_df$PredictedLabel <- NULL
colnames(submission_df) <- c('image_name', 'Type_1', 'Type_2', 'Type_3')


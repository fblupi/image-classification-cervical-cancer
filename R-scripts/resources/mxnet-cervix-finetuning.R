#-------------------------------------------------------------------------------
# Sistemas Inteligentes para Gestión de la Empresa
# Curso 2016-2017
# Departamento de Ciencias de la Computación e Inteligencia Artificial
# Universidad de Granada
#
# Juan Gómez-Romero (jgomez@decsai.ugr.es)
# Francisco Herrera Trigueros (herrera@decsai.ugr.es)
#
# Example of fine tuning with MXNET and Kaggle Cervix dataset
# Inception Network with batch normalization pre-trained with ImageNet data is used
# (from: http://data.dmlc.ml/models/imagenet/)
#
# Implementation based on: 
# https://statist-bhfz.github.io/cats_dogs_finetune (+ section 4)

# Clear workspace
rm(list=ls())

library(mxnet)
library(EBImage)
library(gsubfn)

#-------------------------------------------------------------------------------
# Load and pre-process images (size 224x224)
#-------------------------------------------------------------------------------

# Using pre-processed cervix dataset (already resized to 256x256 with Ebimage and saved to .png)
width  <- 224
height <- 224
train_img_path <- "../mxnet-cervix/data/pre-proc"
file_postfix <- paste("__", format(Sys.time(), "%d-%m-%y_%H-%M-%S"), sep = "")

train_df <- data.frame()
img_file_list <- list.files(path = train_img_path, pattern = "*.png", full.names = TRUE, recursive = TRUE)

n_img <- length(img_file_list)
for(i in 1:n_img) {
  img_file_name <- img_file_list[i]
  img_class <- strapplyc(img_file_list[i], ".*/Type_(.*)/")[[1]]
  img <- readImage(img_file_name)
  # display(img, method="raster")
  img_resized <- resize(img, w=width, h=height)
  img_matrix <- matrix(imageData(img_resized), nrow = height, ncol = width * 3)
  img_vector <- as.vector(t(img_matrix))
  label <- as.numeric(img_class)
  vec <- c(label, img_vector)    
  train_df <- rbind(train_df, vec)
  print(sprintf("[%i of %i] file name: %s, class: %s", i, n_img, img_file_name, img_class))
}

names(train_df) <- c("label", paste("pixel", c(1:width*height)))
saveRDS(object = train_df, file = paste("images-train-", width, "x", height, file_postfix, ".rds", sep=""))

#-------------------------------------------------------------------------------
# Prepare training set (validation is not used)
#-------------------------------------------------------------------------------

# Build training matrix
train <- data.matrix(train_df)
train_x <- t(train[, -1]) 
train_y <- train[, 1]
train_array <- train_x 
dim(train_array) <- c(width, height, 3, ncol(train_x))

#-------------------------------------------------------------------------------
# Load pre-trained model 
#-------------------------------------------------------------------------------

# Load Inception with batch normalization, as it is in the last iteration of the pre-trained model (126)
# File names: Inception-BN-symbol.json, Inception-BN-0126.params
inception_bn <- mx.model.load("Inception-BN/Inception-BN", iteration = 126)
symbol <- inception_bn$symbol
# check symbol$arguments for layer names, specifically for extensions to other networks (e.g. vgg)

# Modify fully-connected layer
internals <- symbol$get.internals()
outputs <- internals$outputs

flatten <- internals$get.output(which(outputs == "flatten_output"))
new_fc <- mx.symbol.FullyConnected(data = flatten, 
                                   num_hidden = 3, 
                                   name = "fc1")      # set name to name not present in symbol$arguments
new_soft <- mx.symbol.SoftmaxOutput(data = new_fc, 
                                    name = "softmax") # set name to name not present in symbol$arguments

arg_params_new <- mxnet:::mx.model.init.params(
  symbol = new_soft, 
  input.shape = c(224, 224, 3, 8), 
  initializer = mxnet:::mx.init.uniform(0.1), 
  ctx = mx.cpu() # Changed from: ctx = mx.gpu(0)
)$arg.params
fc1_weights_new <- arg_params_new[["fc1_weight"]]
fc1_bias_new <- arg_params_new[["fc1_bias"]]

arg_params_new <- inception_bn$arg.params

arg_params_new[["fc1_weight"]] <- fc1_weights_new 
arg_params_new[["fc1_bias"]] <- fc1_bias_new 


#-------------------------------------------------------------------------------
# Create new model object 
#-------------------------------------------------------------------------------
# Create model from previous symbol and train (1 round)
# Model is saved for previous re-use (in ./Inception-BN-xxx.xxx)
model <- mx.model.FeedForward.create(
  symbol             = new_soft,
  X                  = train_array,
  y                  = train_y,
  # eval.data          = val,
  ctx                = mx.cpu(), # mx.gpu(0)
  eval.metric        = mx.metric.accuracy,
  num.round          = 1,
  learning.rate      = 0.05,
  momentum           = 0.9,
  wd                 = 0.00001,
  kvstore            = "local",
  array.batch.size   = 128,
  epoch.end.callback = mx.callback.save.checkpoint("Inception-BN"),
  batch.end.callback = mx.callback.log.train.metric(150),
  initializer        = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
  optimizer          = "sgd",
  arg.params         = arg_params_new,
  aux.params         = inception_bn$aux.params
)

#-------------------------------------------------------------------------------
# Train model
#-------------------------------------------------------------------------------
# Reload model from disk
model <- mx.model.load("Inception-BN", 1)

# Full model train 
# Model is saved (in ./Inception-BN-xxx.xxx)
model <- mx.model.FeedForward.create(
  symbol             = model$symbol,
  X                  = train_array,
  y                  = train_y,
  # eval.data          = val,
  ctx                = mx.cpu(), # mx.gpu(0)
  eval.metric        = mx.metric.accuracy,
  num.round          = 20,
  learning.rate      = 0.03,
  momentum           = 0.9,
  wd                 = 0.00001,
  kvstore            = "local",
  array.batch.size   = 100,
  epoch.end.callback = mx.callback.save.checkpoint("Inception-BN"),
  batch.end.callback = mx.callback.log.train.metric(150),
  initializer        = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
  optimizer          = "sgd",
  arg.params         = model$arg.params, 
  aux.params         = model$aux.params
)

#-------------------------------------------------------------------------------
# Prediction
#-------------------------------------------------------------------------------
# Reload last iteration of model (not necessary)
model <- mx.model.load("Inception-BN", 20)

# Predict labels (train accuracy is calculated, extension for test accuracy is trivial)
predicted <- predict(model, train_array)
predicted_labels <- max.col(t(predicted))
sum(diag(table(train[, 1], predicted_labels)))/length(img_file_list)
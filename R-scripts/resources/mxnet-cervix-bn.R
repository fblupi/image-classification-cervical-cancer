#-------------------------------------------------------------------------------
# Sistemas Inteligentes para Gestión de la Empresa
# Curso 2016-2017
# Departamento de Ciencias de la Computación e Inteligencia Artificial
# Universidad de Granada
#
# Juan Gómez-Romero (jgomez@decsai.ugr.es)
# Francisco Herrera Trigueros (herrera@decsai.ugr.es)
#
# Example of Kaggle cervix challenge with mxnetR using a simple network topology
# https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening

# Image loading and basic pre-processing with EBImage
# BN images
# Pre-processed images and output model are saved in RDS format
# If subsample_train option is set, a subsample of the input images is used
# If reload_train_df option is set, a previous training set is used
#-------------------------------------------------------------------------------

# Clear workspace
rm(list=ls())

#-------------------------------------------------------------------------------
# Set run parameters
#-------------------------------------------------------------------------------
# Processed image and models will be saved for later inspection and use
# data should be in ./data/train/Type_X/* and ./data/test/*

train_img_path <- "./data/train"
width  <- 80
height <- 80
file_postfix <- paste("__", format(Sys.time(), "%d-%m-%y_%H-%M-%S"), sep = "")

subsample_train <- TRUE   # if TRUE, only use a data subset (for testing the script)
subsample_train_size <- 10
if(subsample_train) {
  file_postfix <- paste(file_postfix, "__sample", subsample_train_size, sep = "")
}

reload_train_df <- TRUE # if TRUE, a previous training set is loaded (overrides subsample_train, unless line 42 is uncommented)
train_df_path <- "./images-train-80x80__11-04-17_13-23-16.rds"

training_rounds <- 3

test_img_path <- "./data/test"

#-------------------------------------------------------------------------------
# Load and pre-process train images
#-------------------------------------------------------------------------------

# Load EBImage library
library(EBImage)

# Load images into a dataframe
library(gsubfn)

if(reload_train_df) {
  train_df <- readRDS(train_df_path)
  
  if(subsample_train) {
    # train_df <- sample(train_df, subsample_train_size, replace = FALSE)
    train_df <- train_df[train_df$label == 1,]
    n_img <- nrow(train_df)
  }
  
} else {
  train_df <- data.frame()
  
  img_file_list <- list.files(path = train_img_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)
  
  if(subsample_train) {
    img_file_list <- sample(img_file_list, subsample_train_size, replace=FALSE)
  }
  
  n_img <- length(img_file_list)
  for(i in 1:n_img) {
    img_file_name <- img_file_list[i]
    img_class <- strapplyc(img_file_list[i], ".*/Type_(.*)/")[[1]]
    img <- readImage(img_file_name)
    # display(img, method="raster")
    img_resized <- resize(img, w=width, h=height)
    img_gray <- channel(img_resized, "gray")
    img_matrix <- imageData(img_gray)
    img_vector <- as.vector(t(img_matrix))
    label <- as.numeric(img_class)
    vec <- c(label, img_vector)    
    train_df <- rbind(train_df, vec)
    
    print(sprintf("[%i of %i] file name: %s, class: %s", i, n_img, img_file_name, img_class))
  }
  
  names(train_df) <- c("label", paste("pixel", c(1:width*height)))
  
  saveRDS(object = train_df, file = paste("images-train-", width, "x", height, file_postfix, ".rds", sep=""))
}

#-------------------------------------------------------------------------------
# Setup mxnetR
#-------------------------------------------------------------------------------

# Install mxnetR
# https://github.com/dmlc/mxnet/tree/master/R-package
# install.packages("drat", repos="https://cran.rstudio.com")
# drat:::addRepo("dmlc")
# install.packages("mxnet")

# Load MXNet
require(mxnet)
  
#-------------------------------------------------------------------------------
# Prepare training and validation sets
#-------------------------------------------------------------------------------

# Build training matrix
train <- data.matrix(train_df)
train_x <- t(train[, -1]) 
train_y <- train[, 1]
train_array <- train_x 
dim(train_array) <- c(width, height, 1, ncol(train_x))

# Build validation matrix
# @todo, same as training (see mxnet with MNIST example)

#-------------------------------------------------------------------------------
# Set up the symbolic model
#-------------------------------------------------------------------------------
data <- mx.symbol.Variable('data')

# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")

# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = tanh_1, kernel = c(5, 5), num_filter = 20)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")

# 3rd convolutional layer
conv_3 <- mx.symbol.Convolution(data = tanh_2, kernel = c(5, 5), num_filter = 20)
tanh_3 <- mx.symbol.Activation(data = conv_3, act_type = "tanh")
pool_3 <- mx.symbol.Pooling(data=tanh_3, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

# 4th convolutional layer
conv_4 <- mx.symbol.Convolution(data = pool_3, kernel = c(5, 5), num_filter = 50)
tanh_4 <- mx.symbol.Activation(data = conv_4, act_type = "tanh")

# 5th convolutional layer
conv_5 <- mx.symbol.Convolution(data = tanh_4, kernel = c(5, 5), num_filter = 50)
tanh_5 <- mx.symbol.Activation(data = conv_5, act_type = "tanh")

# 6th convolutional layer
conv_6 <- mx.symbol.Convolution(data = tanh_5, kernel = c(5, 5), num_filter = 50)
tanh_6 <- mx.symbol.Activation(data = conv_6, act_type = "tanh")
pool_6 <- mx.symbol.Pooling(data=tanh_6, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_6)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_1st_fully <- mx.symbol.Activation(data = fc_1, act_type = "tanh")

# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_1st_fully, num_hidden = 3)

# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

#-------------------------------------------------------------------------------
# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(100)

# Device used. CPU in my case
devices <- mx.cpu()

#-------------------------------------------------------------------------------
# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = training_rounds,
                                     array.batch.size = n_img,
                                     learning.rate = 0.001,
                                     momentum = 0.8,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

saveRDS(object = model, file = paste("FFmodel", file_postfix, ".rds", sep=""))
graph.viz(model$symbol)

#-------------------------------------------------------------------------------
# Testing (using the training set, not the test set)
#-------------------------------------------------------------------------------

# Predict labels
predicted <- predict(model, train_array)
# Assign labels
predicted_labels <- max.col(t(predicted)) - 1
# Get accuracy
sum(diag(table(train[, 1], predicted_labels)))/n_img

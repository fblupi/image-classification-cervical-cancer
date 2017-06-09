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
# Color images
# Pre-processed images and output model are saved in RDS format
#-------------------------------------------------------------------------------

# Clear workspace
rm(list=ls())

#-------------------------------------------------------------------------------
# Load and pre-process images
#-------------------------------------------------------------------------------

# Set run parameters parameters
train_img_path <- "./data/train"
width  <- 100
height <- 100
file_postfix <- paste("__", format(Sys.time(), "%d-%m-%y_%H-%M-%S"), sep = "")
subsample_train <- TRUE
subsample_train_size <- 300
if(subsample_train) {
  file_postfix <- paste(file_postfix, "__sample", subsample_train_size, sep = "")
}
training_rounds <- 100

test_img_path <- "./data/test"

#-------------------------------------------------------------------------------
# Load and pre-process train images
#-------------------------------------------------------------------------------

# Load EBImage library
library(EBImage)

# Load images into a dataframe
library(gsubfn)
img_file_list <- list.files(path = train_img_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)

if(subsample_train) {
  img_file_list <- sample(img_file_list, subsample_train_size, replace=FALSE)
}

train_df <- data.frame()

for(i in 1:length(img_file_list)) {
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
  
  print(sprintf("[%i of %i] file name: %s, class: %s", i, length(img_file_list), img_file_name, img_class))
}

names(train_df) <- c("label", paste("pixel", c(1:width*height*3)))

saveRDS(object = train_df, file = paste("images-train-", width, "x", height, file_postfix, ".rds", sep=""))

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
dim(train_array) <- c(width, height, 3, ncol(train_x))

# Build validation matrix
# @todo

#-------------------------------------------------------------------------------
# Set up the symbolic model
#-------------------------------------------------------------------------------
data <- mx.symbol.Variable('data')

# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 3)  # 40 in original example, since there are 40 subjects
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

#-------------------------------------------------------------------------------
# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(100)

# Device used. CPU in my case (using the R version )
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
                                     array.batch.size = length(img_file_list), # 40 in the original example, number of subjects of the experiment
                                     learning.rate = 0.01,
                                     momentum = 0.5,
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
predicted_labels <- max.col(t(predicted))
# Get accuracy
sum(diag(table(train[, 1], predicted_labels)))/length(img_file_list)

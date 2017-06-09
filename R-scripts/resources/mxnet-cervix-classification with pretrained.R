#-------------------------------------------------------------------------------
# Sistemas Inteligentes para Gestión de la Empresa
# Curso 2016-2017
# Departamento de Ciencias de la Computación e Inteligencia Artificial
# Universidad de Granada
#
# Juan Gómez-Romero (jgomez@decsai.ugr.es)
# Francisco Herrera Trigueros (herrera@decsai.ugr.es)
#
# Example of classification with pre-trained network using MXNET and Kaggle Cervix dataset
# Inception Network with batch normalization pre-trained with ImageNet data is used
# (from: http://data.dmlc.ml/models/imagenet/)
#
# Adapted from:
# https://github.com/dmlc/mxnet/blob/master/R-package/vignettes/classifyRealImageWithPretrainedModel.Rmd
library(mxnet)
library(imager)

# Clear workspace
rm(list=ls())

#-------------------------------------------------------------------------------
# Load and pre-process images
#-------------------------------------------------------------------------------
# Load model
model <- mx.model.load("Inception/Inception_BN", iteration = 39)

# Load mean image, used for pre-processing (already stored in n-dimensional array mean_224.nd)
# http://mxnet.io/api/python/ndarray.html
mean.img <- as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])  

# Load image to classify (parrots.png in imager package)
im <- load.image(system.file("extdata/parrots.png", package = "imager"))
plot(im)

# Define pre-processing function (crop an image to 224x224)
# Usage: normed <- preproc.image(im, mean.img)
preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)  
  yy <- floor((shape[2] - short.edge) / 2)
  croped <- crop.borders(im, xx, yy)
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255  # original image values in [0, 1]
  dim(arr) <- c(224, 224, 3)
  # subtract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}

#-------------------------------------------------------------------------------
# Prediction
#-------------------------------------------------------------------------------
# Crop parrots image and apply prediction model
normed <- preproc.image(im, mean.img)
prob <- predict(model, X = normed)

# Retrieve name of the class from output array (one of the 1000 classes used in training)
dim(prob)
max.idx <- max.col(t(prob))
max.idx
synsets <- readLines("Inception/synset.txt")
print(paste0("Predicted Top-class: ", synsets[[max.idx]]))

# Repeat with one of the images of the cervix dataset
im <- load.image("image-6-class-2.png")
plot(im)
normed <- preproc.image(im, mean.img)
prob <- predict(model, X = normed)
max.idx <- max.col(t(prob))
print(paste0("Predicted Top-class: ", synsets[[max.idx]]))

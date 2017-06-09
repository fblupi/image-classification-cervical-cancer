# Clear workspace
rm(list=ls())

# Load libraries
library(EBImage)
library(gsubfn)
require(mxnet)
require(nnet)
library(dplyr)

#-------------------------------------------------------------------------------
# Load and pre-process images
#-------------------------------------------------------------------------------

# Set run parameters parameters
train_img_path <- "./data/train_resized"
test_img_path <- "./data/test_resized"
train_rds <- "./rds/images-train_extra_resized-64x64.rds"
test_rds <- "./rds/images-test_resized-64x64.rds"
width  <- 64
height <- 64
num_train <- 1394
file_postfix <- paste("__", format(Sys.time(), "%d-%m-%y_%H-%M-%S"), sep = "")

create_train <- FALSE
create_test <- FALSE

#-------------------------------------------------------------------------------
# Load and pre-process train and test images
#-------------------------------------------------------------------------------

# Train
if(create_train) {
  img_file_list <- list.files(path = train_img_path, pattern = "*.png", full.names = TRUE, recursive = TRUE)
  
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
} else {
  train_df <- readRDS(train_rds)
}

# Test
if (create_test) {
  img_file_list <- list.files(path = test_img_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)
  
  test_df <- data.frame()
  
  for(i in 1:length(img_file_list)) {
    img_file_name <- img_file_list[i]
    img_class <- "NA"
    img <- readImage(img_file_name)
    img_resized <- resize(img, w=width, h=height)
    img_matrix <- matrix(imageData(img_resized), nrow = height, ncol = width * 3)
    img_vector <- as.vector(t(img_matrix))
    label <- as.numeric(img_class)
    vec <- c(label, img_vector)    
    test_df <- rbind(test_df, vec)
    
    print(sprintf("[%i of %i] file name: %s, class: %s", i, length(img_file_list), img_file_name, img_class))
  }
  
  names(test_df) <- c("label", paste("pixel", c(1:width*height*3)))
  
  saveRDS(object = test_df, file = paste("images-test-", width, "x", height, file_postfix, ".rds", sep=""))
} else {
  test_df <- readRDS(test_rds)
}

#-------------------------------------------------------------------------------
# Prepare training and validation sets
#-------------------------------------------------------------------------------

# Undersample for balanced classes
train_undersample <- train_df %>%
  group_by(label) %>%
  sample_n(num_train) %>%
  ungroup() %>%
  sample_frac(1) %>%
  sample_frac(1)

# Build training matrix
train <- data.matrix(train_undersample)
train_x <- t(train[, -1]) 
train_y <- train[, 1]
train_array <- train_x 
dim(train_array) <- c(width, height, 3, ncol(train_x))

# Build validation matrix
test <- data.matrix(test_df)
test_x <- t(test[, -1])
test_array <- test_x
dim(test_array) <- c(width, height, 3, ncol(test_x))

#-------------------------------------------------------------------------------
# Set up the symbolic model
#-------------------------------------------------------------------------------

# LogLoss func
mLogLoss.normalize = function(p, min_eta=1e-15, max_eta = 1.0){
  for(ix in 1:dim(p)[2]) {
    p[,ix] = ifelse(p[,ix]<=min_eta,min_eta,p[,ix]);
    p[,ix] = ifelse(p[,ix]>=max_eta,max_eta,p[,ix]);
  }
  for(ix in 1:dim(p)[1]) {
    p[ix,] = p[ix,] / sum(p[ix,]);
  }
  return(p);
}

mlogloss = function(y, p, min_eta=1e-15,max_eta = 1.0){
  class_loss = c(dim(p)[2]);
  loss = 0;
  p = mLogLoss.normalize(p,min_eta, max_eta);
  for(ix in 1:dim(y)[2]) {
    p[,ix] = ifelse(p[,ix]>1,1,p[,ix]);
    class_loss[ix] = sum(y[,ix]*log(p[,ix]));
    loss = loss + class_loss[ix];
  }
  return (list("loss"=-1*loss/dim(p)[1],"class_loss"=class_loss));
}

mx.metric.mlogloss <- mx.metric.custom("mlogloss", function(label, pred){
  p = t(pred);
  m = mlogloss(class.ind(label),p);
  gc();
  return(m$loss);
})

data <- mx.symbol.Variable('data')

conv_1 <- mx.symbol.Convolution(data = data, kernel = c(11, 11), num_filter = 16)
relu_1 <- mx.symbol.Activation(data = conv_1, act_type = "relu")

conv_2 <- mx.symbol.Convolution(data = relu_1, kernel = c(7, 7), num_filter = 32)
relu_2 <- mx.symbol.Activation(data = conv_2, act_type = "relu")
pool_2 <- mx.symbol.Pooling(data = relu_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

conv_3 <- mx.symbol.Convolution(data = pool_2, kernel = c(5, 5), num_filter = 64)
relu_3 <- mx.symbol.Activation(data = conv_3, act_type = "relu")
pool_3 <- mx.symbol.Pooling(data = relu_3, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

conv_4 <- mx.symbol.Convolution(data = pool_3, kernel = c(3, 3), num_filter = 32)
relu_4 <- mx.symbol.Activation(data = conv_4, act_type = "relu")

conv_5 <- mx.symbol.Convolution(data = relu_4, kernel = c(3, 3), num_filter = 16)
relu_5 <- mx.symbol.Activation(data = conv_5, act_type = "relu")

conv_6 <- mx.symbol.Convolution(data = relu_5, kernel = c(3, 3), num_filter = 8)
relu_6 <- mx.symbol.Activation(data = conv_6, act_type = "relu")
pool_6 <- mx.symbol.Pooling(data = relu_6, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

flatten <- mx.symbol.Flatten(data = pool_6)
fc_7 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 1024)
relu_7 <- mx.symbol.Dropout(data = fc_7, p = 0.5)

fc_8 <- mx.symbol.FullyConnected(data = relu_7, num_hidden = 256)
relu_8 <- mx.symbol.Dropout(data = fc_8, p = 0.5)

fc_9 <- mx.symbol.FullyConnected(data = relu_8, num_hidden = 64)
relu_9 <- mx.symbol.Dropout(data = fc_9, p = 0.5)

fc_10 <- mx.symbol.FullyConnected(data = relu_9, num_hidden = 48)
relu_10 <- mx.symbol.Dropout(data = fc_10, p = 0.5)

fc_11 <- mx.symbol.FullyConnected(data = relu_10, num_hidden = 32)
relu_11 <- mx.symbol.Dropout(data = fc_11, p = 0.5)

fc_12 <- mx.symbol.FullyConnected(data = relu_11, num_hidden = 32)
relu_12 <- mx.symbol.Dropout(data = fc_12, p = 0.5)

fc_13 <- mx.symbol.FullyConnected(data = relu_12, num_hidden = 16)
relu_13 <- mx.symbol.Dropout(data = fc_13, p = 0.5)

fc_14 <- mx.symbol.FullyConnected(data = relu_13, num_hidden = 3)
NN_model <- mx.symbol.SoftmaxOutput(data = fc_14)

#-------------------------------------------------------------------------------
# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(100)

#-------------------------------------------------------------------------------
# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = mx.cpu(),
                                     num.round = 20,
                                     array.batch.size = length(train_y),
                                     learning.rate = 0.1,
                                     momentum = 0.005,
                                     wd = 0.004,
                                     eval.metric = mx.metric.mlogloss,
                                     batch.end.callback = mx.callback.log.train.metric(10))

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
sum(diag(table(train[, 1], predicted_labels)))/length(train_y)

#-------------------------------------------------------------------------------
# Testing (using the test set)
#-------------------------------------------------------------------------------

# Predict labels
predicted <- predict(model, test_array)
# Export CSV
predicted_transpose <- t(predicted)
write.csv(predicted_transpose, paste("result", file_postfix, ".csv", sep=""))

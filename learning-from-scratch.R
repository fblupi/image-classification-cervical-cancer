# Clear workspace
rm(list=ls())

# Load libraries
library(EBImage)
library(gsubfn)
require(mxnet)
require(nnet)

#-------------------------------------------------------------------------------
# Load and pre-process images
#-------------------------------------------------------------------------------

# Set run parameters parameters
train_img_path <- "./data/train_resized"
test_img_path <- "./data/test_resized"
width  <- 64
height <- 64
file_postfix <- paste("__", format(Sys.time(), "%d-%m-%y_%H-%M-%S"), sep = "")
training_rounds <- 10

create_train <- FALSE
create_test <- FALSE

#-------------------------------------------------------------------------------
# Load and pre-process train images
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
  train_df <- readRDS("./rds/images-train_extra_resized-64x64.rds")
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
  test_df <- readRDS("./rds/images-test-64x64.rds")
}

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
test <- data.matrix(test_df)
test_x <- t(test[, -1])
test_array <- test_x
dim(test_array) <- c(width, height, 3, ncol(test_x))

#-------------------------------------------------------------------------------
# Set up the symbolic model
#-------------------------------------------------------------------------------
data <- mx.symbol.Variable('data')

conv_1 <- mx.symbol.Convolution(data = data, kernel = c(3, 3), num_filter = 3)
relu_1 <- mx.symbol.Activation(data = conv_1, act_type = "relu")
pool_1 <- mx.symbol.Pooling(data = relu_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(3, 3), num_filter = 14)
relu_2 <- mx.symbol.Activation(data = conv_2, act_type = "relu")
pool_2 <- mx.symbol.Pooling(data = relu_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 84)
relu_3 <- mx.symbol.Activation(data = fc_1, act_type = "relu")

fc_2 <- mx.symbol.FullyConnected(data = relu_3, num_hidden = 3)
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

#-------------------------------------------------------------------------------
# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(100)

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

#-------------------------------------------------------------------------------
# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = mx.cpu(),
                                     num.round = training_rounds,
                                     array.batch.size = length(train_y),
                                     learning.rate = 0.1,
                                     momentum = 0.005,
                                     wd = 0.004,
                                     initializer=mx.init.uniform(0.07),
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

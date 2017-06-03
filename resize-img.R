# Clear workspace
rm(list=ls())

#-------------------------------------------------------------------------------
# Set run parameters
#-------------------------------------------------------------------------------
# Processed image and models will be saved for later inspection and use
# data should be in ./data/train/Type_X/* and ./data/test/*

img_path <- "./data/test"
read_dir_name <- "test"
write_dir_name <- "test_resized"
width  <- 256
height <- 256

#-------------------------------------------------------------------------------
# Load and pre-process train images
#-------------------------------------------------------------------------------

# Load EBImage library
#source("https://bioconductor.org/biocLite.R")
#biocLite("EBImage")
library(EBImage)

# Load images into a dataframe
#install.packages("gsubfn")
library(gsubfn)

img_file_list <- list.files(path = img_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)
n_img <- length(img_file_list)
for(i in 1:n_img) {
  message(sprintf("Writing image %d out of %d\n", i, n_img))
  img_file_name <- img_file_list[i]
  img_class <- strapplyc(img_file_list[i], ".*/Type_(.*)/")[[1]]
  img <- readImage(img_file_name)
  img_resized <- resize(img, w=width, h=height)
  new_file_name <- gsub(read_dir_name, write_dir_name, img_file_name)
  writeImage(img_resized, new_file_name, quality = 85)
}

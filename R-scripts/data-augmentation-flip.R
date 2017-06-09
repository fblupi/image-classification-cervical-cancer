rm(list=ls())

img_path <- "./data/train_extra_resized"

#source("https://bioconductor.org/biocLite.R")
#biocLite("EBImage")
library(EBImage)

#install.packages("gsubfn")
library(gsubfn)

img_file_list <- list.files(path = img_path, pattern = "*.jpg", full.names = TRUE, recursive = TRUE)
n_img <- length(img_file_list)
for(i in 1:n_img) {
  message(sprintf("Writing image %d out of %d\n", i, n_img))
  img_file_name <- img_file_list[i]
  img <- readImage(img_file_name)
  # flip vertical
  new_img <- flip(img)
  new_file_name <- gsub(".jpg", "fv.jpg", img_file_name)
  writeImage(new_img, new_file_name)
  # flip horizontal
  new_img <- flop(img)
  new_file_name <- gsub(".jpg", "fh.jpg", img_file_name)
  writeImage(new_img, new_file_name)
  # flip flop
  new_img <- flip(new_img)
  new_file_name <- gsub(".jpg", "fhv.jpg", img_file_name)
  writeImage(new_img, new_file_name)
}

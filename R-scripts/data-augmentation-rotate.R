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
  # rotate right
  angle <- runif(1, 5.0, 10.0)
  new_img <- rotate(img, angle)
  new_file_name <- gsub(".jpg", "rr.jpg", img_file_name)
  writeImage(new_img, new_file_name)
  # rotate left
  angle <- runif(1, 5.0, 10.0)
  new_img <- rotate(img, -angle)
  new_file_name <- gsub(".jpg", "rl.jpg", img_file_name)
  writeImage(new_img, new_file_name)
}

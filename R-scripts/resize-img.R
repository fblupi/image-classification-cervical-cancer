rm(list=ls())

img_path <- "./data/train_extra"
read_dir_name <- "train_extra"
write_dir_name <- "train_extra_resized"
width  <- 256
height <- 256

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
  img_resized <- resize(img, w=width, h=height)
  new_file_name <- gsub(read_dir_name, write_dir_name, img_file_name)
  writeImage(img_resized, new_file_name, quality = 85)
}

library(keras)
library(reticulate)
#library(imager)
library(magick)
library(dplyr)

IM_HEIGHT = 708
IM_WIDTH = 990 

IMG_NEW_SIZE = 128
IMG_CHANNELS = 3

magick.format <- paste0(as.character(IMG_NEW_SIZE),"x",as.character(IMG_NEW_SIZE),"!")

resize_position <- function(xy) {
  # xy <- c(100,300,400,500)
  x1 <- xy[1]
  y1 <- xy[2]
  x2 <- xy[3]
  y2 <-xy[4]
  x1_r <- x1 / (IM_WIDTH / IMG_NEW_SIZE)
  y1_r <- y1 / (IM_HEIGHT / IMG_NEW_SIZE)
  x2_r <- x2 / (IM_WIDTH / IMG_NEW_SIZE)
  y2_r <- y2 / (IM_HEIGHT / IMG_NEW_SIZE)
  
  xy_r <- round(c(x1_r, y1_r, x2_r, y2_r))
  
  xy_r
}


read_rectangles <- function(fname)
{
  positions <- read.csv2(bb.csv,header = FALSE,sep = ' ')
  positions <- positions %>% select(V5:V8)
  
  positions <- t(apply(positions, MARGIN = 1,FUN = resize_position))
  positions
}


create_bbox_mask <- function(positions, IMG_NEW_SIZE)
{
  
  mask = array(0, dim = c(IMG_NEW_SIZE, IMG_NEW_SIZE, 1))
  for (i in 1:nrow(positions)) {
    row <- positions[i,]
    mask[row[2]:row[4], row[1]:row[3],1] = 255
  } 
  mask
}


INPUT_PATH <-  'C:\\HEK\\Dewpoint\\BB'

IMAGE_DIR <-  paste(INPUT_PATH,'images\\',sep = '\\')
BB_DIR <-  paste(INPUT_PATH,'positions\\',sep = '\\') 

images <- list.files(IMAGE_DIR)
bbs <- unlist(lapply(images,FUN = function(x) paste0(unlist(strsplit(x,split = '.',fixed = TRUE))[1],'.txt')))

length.images <-  length(images)
length.images <- 10

X = array(as.integer(0),dim = c(length.images, IMG_NEW_SIZE, IMG_NEW_SIZE, IMG_CHANNELS))
Y = array(0,dim = c(length.images, IMG_NEW_SIZE, IMG_NEW_SIZE, 1))

for(i in seq(length.images))
{
  image_name <- images[i]
  im <-  image_read(paste0(IMAGE_DIR,image_name))
  im <- image_resize(im, magick.format)
  X[i,,,] <- as.integer(image_data(im))/255
  
  bb.csv <- paste0(BB_DIR,bbs[i])
  positions <- read_rectangles(bb.csv)
  
  Y[i,,,] <- create_bbox_mask(positions, IMG_NEW_SIZE)/255
}

set.seed(12345)

test_inds <-  sample(1:length.images, ceiling(length.images * 0.2))

X_test = X[test_inds,,, ]
Y_test = Y[test_inds ,,,] 

X_train = X[-test_inds,,, ]
Y_train = Y[-test_inds ,,,] 

showImageMask <- function(ig, msk) {
  ig.data <- as.integer(image_data(ig)) / 256
  ig2 <- ig.data
  ig2[, , 3] <- ig.data[, , 3] + 0.22 * msk
  ig2[, , 2] <- ig.data[, , 2] + 0 * msk
  
  out.im <- image_read(ig2)  %T>% plot()
}

r <- sample(length.images, 1)
ig <- image_read(X[r, , , ])
msk <-  Y[r, , , 1]

showImageMask(ig, msk)

dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) / 
    (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
  return(result)
}

bce_dice_loss <- function(y_true, y_pred) {
  result <- loss_binary_crossentropy(y_true, y_pred) +
    (1 - dice_coef(y_true, y_pred))
  return(result)
}


# 
# IM2 <- c(im,image_read(mask/255))
# 
# s <- image_combine(IM2, colorspace = "rgb")

get_unet_128 <- function(input_shape = c(IMG_NEW_SIZE, IMG_NEW_SIZE, IMG_CHANNELS),
                         num_classes = 1) {
  
  inputs <- layer_input(shape = input_shape)
  # 128
  
  down1 <- inputs %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down1_pool <- down1 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 64
  
  down2 <- down1_pool %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down2_pool <- down2 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 32
  
  down3 <- down2_pool %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down3_pool <- down3 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 16
  
  down4 <- down3_pool %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down4_pool <- down4 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 8
  
  center <- down4_pool %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  # center
  
  up4 <- center %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down4, .), axis = 3)} %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 16
  
  up3 <- up4 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 32
  
  up2 <- up3 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 64
  
  up1 <- up2 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  # 128
  
  classify <- layer_conv_2d(up1,
                            filters = num_classes, 
                            kernel_size = c(1, 1),
                            activation = "sigmoid")
  
  model <- keras_model(
    inputs = inputs,
    outputs = classify
  )
  
  model %>% compile(
    optimizer = optimizer_rmsprop(lr = 0.0001),
    loss = bce_dice_loss,
    metrics = c(dice_coef)
  )
  
  return(model)
}

model <- get_unet_128()

history <- model %>% fit(x = X_train,y = Y_train, epochs=5 ,validation_split=0.1, batch_size=16)
plot(history)

evaluate(model, X_test, Y_test, verbose = 0)

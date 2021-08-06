# Print_Hook <- function(grad){
#   print(grad)
# }
# $register_hook(print(grad))


#################################################################################################################################
# Demo
create_data <- function(num){
  x = (rnorm(num) - 0.5) * 2 * X_MAX # np.random.rand(num)
  y = (rnorm(num) - 0.5) * 2 * Y_MAX
  w = (rnorm(num) - 0.5) * 2 * SCALE + 1
  h = (rnorm(num) - 0.5) * 2 * SCALE + 1
  alpha = rnorm(num) * pi
  corners =  array(0.0, dim=c(num,4,2)) # np.zeros((num, 4, 2)).astype(np.float)
  
  for (i in 1:dim(corners)[1]){
    corners[i,,] = box2corners(x[i], y[i], w[i], h[i], alpha[i])
  }
  label = cbind(x, y , w, h, alpha)
  
  return(list(corners, label))
}

#################################################################################################################################
# Demo
save_dataset <- function(DATA_DIR, NUM_TRAIN, NUM_TEST){
  Train_datasets <- create_data(NUM_TRAIN)
  train_data <- Train_datasets[[1]]
  train_label <- Train_datasets[[2]] 
  # browser()
  write.csv(train_data, paste(DATA_DIR, "/train_data.csv", sep = ""), row.names =F)   
  write.csv(train_label, paste(DATA_DIR, "/train_label.csv", sep = ""), row.names =F)
  
  Test_datasets <- create_data(NUM_TEST)
  test_data <- Test_datasets[[1]]
  test_label <- Test_datasets[[2]] 
  write.csv(test_data, paste(DATA_DIR, "/test_data.csv", sep = ""), row.names =F)
  write.csv(test_label, paste(DATA_DIR, "/test_label.csv", sep = ""), row.names =F)
  
  print(paste("data saved in: ", DATA_DIR))
}

Data <- save_dataset(DATA_DIR, NUM_TRAIN, NUM_TEST)

#################################################################################################################################
# Demo
create_network <- function(){
  nn_sequential(nn_conv1d(8, 128, 1, bias=FALSE), 
                nn_batch_norm1d(128), 
                nn_relu (TRUE), # nn.ReLU
                nn_conv1d(128, 512, 1, bias=FALSE), 
                nn_batch_norm1d(512),
                nn_relu(TRUE),
                nn_conv1d(512, 128, 1, bias=FALSE),
                nn_batch_norm1d(128),
                nn_relu(TRUE),
                nn_conv1d(128, 5, 1),
                nn_sigmoid()) 
}


#################################################################################################################################
# Demo
BoxDataSet <- dataset(
  name = "TREES_DS_FUN",
  initialize = function(split = "train") {
    self$split =split
    #browser()
    self$data = torch_tensor(as.matrix(read.csv(paste(DATA_DIR, "/", split, "_data.csv", sep=""))))
    self$data <- self$data$view(c(dim(self$data)[1],2,4)) %>%  torch_transpose(2, 3)
    
    self$label = torch_tensor(as.matrix(read.csv(paste(DATA_DIR, "/", split, "_label.csv", sep=""))))
  },
  .length = function() {
    self$data$size(1)
  },
  .getitem = function(index) {
    # browser()
    d = self$data[index,,]
    l = self$label[index,]
    
    return(list(d, l))
  }
)

#################################################################################################################################
# Demo
parse_pred <- function(pred){
  p1 = (pred[,, 1] - 0.5) * 2 * X_MAX
  p2 = (pred[,, 2] - 0.5) * 2 * Y_MAX
  p3 = (pred[,, 3] - 0.5) * 2 * SCALE + 1
  p4 = (pred[,, 4] - 0.5) * 2 * SCALE + 1
  p5 = pred[,, 5] * pi
  return (torch_stack(c(p1,p2,p3,p4,p5), dim=-1))
}
# Print_Hook <- function(grad){
#   print(grad)
# }
# $register_hook(print(grad))


#################################################################################################################################
# Demo
create_data <- function(num){
  x = (runif(num) - 0.5) * 2 * X_MAX # np.random.rand(num)    (x,y,z,w,h,l,alpha)
  y = (runif(num) - 0.5) * 2 * Y_MAX
  z = (runif(num) - 0.5) * 2 * Z_MAX
  w = (runif(num) - 0.5) * 2 * SCALE + 1
  h = (runif(num) - 0.5) * 2 * SCALE + 1
  l = (runif(num) - 0.5) * 2 * SCALE + 1
  alpha = rnorm(num) * pi
  corners =  array(0.0, dim=c(num,8,3)) #         # np.zeros((num, 4, 2)).astype(np.float)

  for (i in 1:dim(corners)[1]){
    # browser()
    corners[i,,] = box2corners_3d(x[i], y[i], z[i], w[i], h[i], l[i], alpha[i])
    
  }
  label = cbind(x, y , z,  w, h, l, alpha)
  # browser()
  return(list(corners, label))
}

#################################################################################################################################
# Demo
save_dataset <- function(DATA_DIR, NUM_TRAIN, NUM_TEST){
  Train_datasets <- create_data(NUM_TRAIN)
  train_data <- Train_datasets[[1]]
  train_label <- Train_datasets[[2]] 

  # write.csv(train_data, paste(DATA_DIR, "/train_data.csv", sep = ""), row.names =F)   
  # write.csv(train_label, paste(DATA_DIR, "/train_label.csv", sep = ""), row.names =F)
  np$save(paste(DATA_DIR, "/train_data.npy", sep = ""), train_data)
  np$save(paste(DATA_DIR, "/train_label.npy", sep = ""), train_label)
  

  Test_datasets <- create_data(NUM_TEST)
  test_data <- Test_datasets[[1]]
  test_label <- Test_datasets[[2]] 
  # write.csv(test_data, paste(DATA_DIR, "/test_data.csv", sep = ""), row.names =F)
  # write.csv(test_label, paste(DATA_DIR, "/test_label.csv", sep = ""), row.names =F)
  np$save(paste(DATA_DIR, "/test_data.npy", sep = ""), test_data)
  np$save(paste(DATA_DIR, "/test_label.npy", sep = ""), test_label)
  
  print(paste("data saved in: ", DATA_DIR))
}

Data <- save_dataset(DATA_DIR, NUM_TRAIN, NUM_TEST)
#browser()
#################################################################################################################################
# Demo
create_network <- function(){
  
  nn_sequential(nn_conv1d(24, 128, 1, bias=FALSE), 
                nn_batch_norm1d(128), 
                nn_relu (TRUE), # nn.ReLU
                nn_conv1d(128, 512, 1, bias=FALSE), 
                nn_batch_norm1d(512),
                nn_relu(TRUE),
                nn_conv1d(512, 128, 1, bias=FALSE),
                nn_batch_norm1d(128),
                nn_relu(TRUE),
                nn_conv1d(128, 7, 1), # CHANGED THIS FOR 3D COMPUTATION
                nn_sigmoid()) 
}


#################################################################################################################################
# Demo
BoxDataSet <- dataset(
  name = "TREES_DS_FUN",
  initialize = function(split = "train") {
    self$split =split

    # self$data <- self$data$view(c(dim(self$data)[1],2,4)) %>%  torch_transpose(2, 3)
    # self$label = torch_tensor(as.matrix(read.csv(paste(DATA_DIR, "/", split, "_label.csv", sep=""))))
    #browser()
    self$data =  torch_tensor(np$load(paste(DATA_DIR, "/", split, "_data.npy", sep="")), dtype = torch_float())$to(device =  device) # torch_tensor(as.matrix(read.csv(paste(DATA_DIR, "/", split, "_data.npy", sep=""))))
    self$label =  torch_tensor(np$load(paste(DATA_DIR, "/", split, "_label.npy", sep="")), dtype = torch_float())$to(device =  device)
  
    # self$data = torch_tensor(as.matrix(read.csv(paste(DATA_DIR, "/", split, "_data.csv", sep=""))))
    # self$data <- self$data$view(c(dim(self$data)[1],2,4)) %>%  torch_transpose(2, 3)
    # self$label = torch_tensor(as.matrix(read.csv(paste(DATA_DIR, "/", split, "_label.csv", sep=""))))
    
  },
  .length = function() {
    # browser()
    self$data$size(1)
  },
  .getitem = function(index) {
    # 
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

#################################################################################################################################
# Demo
parse_pred_3d <- function(pred){
  p1 = (pred[,, 1] - 0.5) * 2 * X_MAX
  p2 = (pred[,, 2] - 0.5) * 2 * Y_MAX
  p3 = (pred[,, 3] - 0.5) * 2 * Z_MAX
  p4 = (pred[,, 4] - 0.5) * 2 * SCALE + 1
  p5 = (pred[,, 5] - 0.5) * 2 * SCALE + 1
  p6 = (pred[,, 6] - 0.5) * 2 * SCALE + 1
  p7 = pred[,, 7] * pi
  return (torch_stack(c(p1,p2,p3,p4,p5,p6,p7), dim=-1))
}
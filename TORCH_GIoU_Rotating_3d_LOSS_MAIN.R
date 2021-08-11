
# This demo is used to validate the back-propagation of the torch implementation of oriented
# 2d/3D box intersection. 
# 
# This demo trains a network which takes N set of box corners and predicts the x, y, w, h and angle
# of each rotated boxes. In order to do the back-prop, the prediected box parameters and GT are 
# converted to coordinates of box corners. The area of intersection area is calculated using 
# the pytorch function with CUDA extension. Then, the GIoU loss or DIoU loss can be calculated.
# 
# This demo first generates data and then do the training.
# 
# The network is simply a shared MLP (implemented as Conv-layers with 1x1 kernel).
# 
# Demo translates the python code from Lanxiao Li # 2020.08

#################################################################################################################################


# Error in (function (tensors, dim)  : 
#             CUDA error: device-side assert triggered
#           CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
#           For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
          
rm(list=ls())
options(digits = 12)
options(warn=1, error = NULL) # options(warn=2, error=recover)

Sys.setenv(CUDA_LAUNCH_BLOCKING=1)

library(torch)
library(reticulate)
np <- import("numpy")

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu" #  

DATA_DIR = "D:/PYTHON_CODE/Rotated_IoU-master/data"
SOURCE_DIR = "D:/Y_Drive/CNN/R_Code/TORCH_ITCD_EXTRAP_V22/PYTORCH_TRANSLATIONS/Torch_R_GIoU_3dRotated_BBox_Loss"


# Sys.setenv(CUDA_LAUNCH_BLOCKING=1)

# MODEL PARAMETRES
X_MAX = 3
Y_MAX = 3
Z_MAX = 3
SCALE = 0.5
BATCH_SIZE = 8 # 32
N_DATA = 32 # 128
NUM_TRAIN = 20 * BATCH_SIZE * N_DATA # 200 * BATCH_SIZE * N_DATA
NUM_TEST = 2 * BATCH_SIZE * N_DATA # 20 * BATCH_SIZE * N_DATA
NUM_EPOCH = 1000
EPSILON = 1e-8

#################################################################################################################################
# # HOOK FUNCTIONS
# e_hook <- function(grad){
#   print("e_hook")
#   # browser()
#   #print(paste("iou_loss", as.character(as.array(grad$to(device = "cpu")))))
#   return(grad)
# }
# 
# max_hook1<- function(grad){
#   print("maxHook1")
#   #browser()
#   #print(paste("ar", as.character(as.array(grad$to(device = "cpu")))))
#   return(grad)
# }
# 
# max_hook2<- function(grad){
#   print("maxHook2")
#   #browser()
#   #print(paste("ar", as.character(as.array(grad$to(device = "cpu")))))
#   return(grad)
# }
#################################################################################################################################
# SOURCE CODE
source(paste(SOURCE_DIR, "/TORCH_GIoU_BOX_INTERSECT.R", sep=""))
source(paste(SOURCE_DIR, "/TORCH_GIoU_MIN_ENCLOSING.R", sep=""))
source(paste(SOURCE_DIR, "/TORCH_GIoU_ORIENT_IoU.R", sep=""))
source(paste(SOURCE_DIR, "/TORCH_GIoU_UTILES.R", sep=""))

source(paste(SOURCE_DIR, "/TORCH_GIoU_DEMO.R", sep=""))
#################################################################################################################################
Main <- function(Loss_Type = "Complete_iou_3d", enclosing_type){ # 
  # browser()
  ds_train = BoxDataSet("train")
  ds_test = BoxDataSet("test")
  
  ld_train = ds_train %>% dataloader( BATCH_SIZE * N_DATA, shuffle=TRUE) #TRUE , num_workers=4
  ld_test = ds_test %>%  dataloader( BATCH_SIZE * N_DATA, shuffle=FALSE) #, num_workers=4
  # browser()
  net = create_network()$to(device=device)
  
  optimizer <- optim_sgd(net$parameters, lr= 0.01,   momentum=0.9) # model_Instance_Para
  lr_scheduler <- optimizer %>% lr_one_cycle( max_lr = 0.01, total_steps =NUM_EPOCH)  # max_lr = 0.01, epochs = NUM_EPOCH, steps_per_epoch = ld_train$.length()
  num_batch = length(ds_train)/(BATCH_SIZE*N_DATA)
 
  for (epoch in 1:(NUM_EPOCH)){
   
    net$train()
    ld_train$.length()

    Count_Corro <- 0
    coro::loop(for (b in ld_train) {   # for i, data in enumerate(ld_train, 1):
      
      start_time <- proc.time()
      Count_Corro <- Count_Corro + 1
      #data <- train_batch(i) # ??
      # print("enumerate loop")
      # browser()
      
      # DO YOU HAVE TO ADD THE Z VALUE TO EACH (OR NO???)
      box <- b[[1]]$to(device=device)
      
     # browser()
      box <- box$view(c(BATCH_SIZE, -1, 8*3)) %>% torch_transpose(2, 3)  # 4  8 16  (B, 8, N)
      
      label <- b[[2]]$to(device=device)
      label <- label$view(c(BATCH_SIZE, -1, 7)) #%>% torch_transpose(2, 3)   # 3d CHANGES CHANGED 5 to 7
      
      # browser()                                                           # 4 16  5
      optimizer$zero_grad()
      # browser()
      pred = net(box)  %>% torch_transpose(2, 3)                             # (B, N, 7)
      
      # FOR DEBUGGING OPEN THE PYTHON PREDICTIONS
      # pred = torch_tensor(np$load(paste(DATA_DIR, "/",  "Pred_data.npy", sep="")), dtype = torch_float())$to(device =  device)
      # browser()
      pred = parse_pred_3d(pred)
      iou_loss <- NA
      iou <- NA
      
      # if(Loss_Type == "giou_3d"){
      #   # browser()  
      #   Output_IoU = cal_giou_3d(pred, label, enclosing_type)
      #   iou_loss <- Output_IoU[[1]] 
      #   iou <- Output_IoU[[2]]
      # }

      if(Loss_Type == "Complete_iou_3d"){
        #browser()
        Output_IoU = cal_complete_iou_3d(pred, label, enclosing_type)
        iou_loss <- Output_IoU[[1]] 
        iou <- Output_IoU[[2]]
      }
      
      iou_loss <- torch_mean(iou_loss)  #iou_loss = torch.mean(iou_loss)

      end_time <- proc.time()
      Forward_Time_Taken <- round((end_time - start_time)[[3]],2)
      
      start_time <- proc.time()
      #with_detect_anomaly({
      iou_loss$backward()
      #})
      end_time <- proc.time()
      Backward_Time_Taken <- round((end_time - start_time)[[3]],2)

      optimizer$step()
      
      
      if (epoch%%10 == 0){
        
        # pred  box
        iou_mask = (iou > 0)$to(dtype = torch_float())
        mean_iou = torch_sum(iou) / (torch_sum(iou_mask) + 1e-8)
        cat(sprintf("\nEpoch %d, Cnt:%d, Batch:%d, loss:%3f,  mean_iou: %4f, Fwd_Time: %4f, Back_Time: %4f",
                    epoch, Count_Corro, num_batch, 
                    round(as.array(iou_loss$to(device = "cpu")),3), 
                    round(as.array(mean_iou$to(device = "cpu")),3),
                    Forward_Time_Taken,
                    Backward_Time_Taken))
        
        # browser()
        # pred_V <- pred$view(c(-1, 7))
        # Plot Results to See
        
        }
      })
    
    lr_scheduler$step()

    # validate
    net$eval()
    Valid_loss = c()
    Valid_mean_iou = c()
    with_no_grad({
      coro::loop(for (b in ld_test) {   # for i, data in enumerate(ld_train, 1):
       
        #data <- train_batch(i) # ??
        #browser()
        box <- b[[1]]$to(device=device)
        box <- box$view(c(BATCH_SIZE, -1, 8*3)) %>% torch_transpose(2, 3) 
        
        label <- b[[2]]$to(device=device)
        label <- label$view(c(BATCH_SIZE, -1, 7)) # %>% torch_transpose(2, 3) 
        
        optimizer$zero_grad()
        
        pred = net(box)  %>% torch_transpose(2, 3)                             # (B, 5, N)
        
        pred = parse_pred_3d(pred)
        iou_loss <- NA
        iou <- NA
        
        # if(Loss_Type == "giou_3d"){
        #   #browser()
        #   Output_IoU = cal_giou_3d(pred, label, enclosing_type)
        #   iou_loss <- Output_IoU[[1]] 
        #   iou <- Output_IoU[[2]]
        # }
        
        if(Loss_Type == "Complete_iou_3d"){
          #browser()
          Output_IoU = cal_complete_iou_3d(pred, label, enclosing_type)
          iou_loss <- Output_IoU[[1]] 
          iou <- Output_IoU[[2]]
        }
        
        iou_loss <- torch_mean(iou_loss)
        Valid_loss <- c(Valid_loss, as.array(iou_loss$cpu()))
        
        iou_mask = (iou > 0)$to(dtype = torch_float())
        mean_iou = torch_sum(iou) / (torch_sum(iou_mask) + 1e-8)
        Valid_mean_iou = c(Valid_mean_iou, as.array(mean_iou$cpu())) # += mean_iou.cpu().item()
      })
    })
    aver_loss = sum(Valid_loss)
    aver_mean_iou <- sum(Valid_mean_iou)
    cat(sprintf("... validate epoch %d ", epoch))
    n_iter = (length(ds_test)/BATCH_SIZE)/N_DATA
    cat(sprintf("  average loss: %.4f" , (aver_loss/n_iter)))
    cat(sprintf("  average iou: %.4f" , (aver_mean_iou/n_iter)))
    
    Output_Valid <- c(Output_Valid, (aver_loss/n_iter))
    plot(1:epoch, Output_Valid)
    print("..............................")
  }
}
# browser()
# THIS WILL BE USED FOR MINIMUM ENCLOSED BOX... NEED TO CHECK ITS CORRECT
generate_table_out = generate_table()
LINES <- generate_table_out[[1]]
POINTS <- generate_table_out[[2]]
# browser()
# RUN THE IoU ML
Output_Valid <- c()
Main(Loss_Type ="Complete_iou_3d", enclosing_type="smallest") # aligned









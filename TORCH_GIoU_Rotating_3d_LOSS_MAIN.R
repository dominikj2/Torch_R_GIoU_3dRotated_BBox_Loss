
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

rm(list=ls())
options(digits = 12)
options(warn=1, error = NULL) # options(warn=2, error=recover)

library(torch)

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

DATA_DIR = "D:/PYTHON_CODE/Rotated_IoU-master/data"
SOURCE_DIR = "D:/Y_Drive/CNN/R_Code/TORCH_ITCD_EXTRAP_V22/PYTORCH_TRANSLATIONS/TORCH_GIoU_Rotating_3d_LOSS"


# MODEL PARAMETRES
X_MAX = 3
Y_MAX = 3
SCALE = 0.5
BATCH_SIZE = 4
N_DATA = 16
NUM_TRAIN = 50 * BATCH_SIZE * N_DATA
NUM_TEST = 5 * BATCH_SIZE * N_DATA
NUM_EPOCH = 20
EPSILON = 1e-8

#################################################################################################################################
# SOURCE CODE
source(paste(SOURCE_DIR, "/TORCH_GIoU_BOX_INTERSECT.R", sep=""))
source(paste(SOURCE_DIR, "/TORCH_GIoU_MIN_ENCLOSING.R", sep=""))
source(paste(SOURCE_DIR, "/TORCH_GIoU_ORIENT_IoU.R", sep=""))
source(paste(SOURCE_DIR, "/TORCH_GIoU_UTILES.R", sep=""))

source(paste(SOURCE_DIR, "/TORCH_GIoU_DEMO.R", sep=""))
#################################################################################################################################
Main <- function(Loss_Type = "giou", Enclosing_Type="aligned"){

  ds_train = BoxDataSet("train")
  ds_test = BoxDataSet("test")
  
  ld_train = ds_train %>% dataloader( BATCH_SIZE * N_DATA, shuffle=TRUE)
  ld_test = ds_test %>%  dataloader( BATCH_SIZE * N_DATA, shuffle=FALSE)
  
  net = create_network()$to(device=device)

  optimizer <- optim_sgd(net$parameters, lr= 0.01,   momentum=0.9) # model_Instance_Para
  lr_scheduler <- optimizer %>% lr_one_cycle( max_lr = 0.01, total_steps =NUM_EPOCH)  # max_lr = 0.01, epochs = NUM_EPOCH, steps_per_epoch = ld_train$.length()
  num_batch = length(ds_train)/(BATCH_SIZE*N_DATA)
 
  for (epoch in 1:(NUM_EPOCH+1)){
    net$train()
    ld_train$.length()

    coro::loop(for (b in ld_train) {   # for i, data in enumerate(ld_train, 1):
      # browser()
      #data <- train_batch(i) # ??
      
      box <- b[[1]]$to(device=device)
      box <- box$view(c(BATCH_SIZE, -1, 4*2)) %>% torch_transpose(2, 3)  # 4  8 16  (B, 8, N)
      
      label <- b[[2]]$to(device=device)
      label <- label$view(c(BATCH_SIZE, -1, 5)) #%>% torch_transpose(2, 3) 
      #browser()                                                           # 4 16  5
      optimizer$zero_grad()
      #browser() 
      pred = net(box)  %>% torch_transpose(2, 3)                             # (B, N, 5)
      #
      
      pred = parse_pred(pred)
      iou_loss <- NA
      iou <- NA
      # browser()
      if(Loss_Type == "giou"){
        #browser()
        Output_IoU = cal_giou(pred, label, enclosing_type)
        iou_loss <- Output_IoU[[1]] 
        iou <- Output_IoU[[2]]
      }

      if(Loss_Type == "diou"){
         Output_IoU = cal_diou(pred, label, enclosing_type)
         iou_loss <- Output_IoU[[1]] 
         iou <- Output_IoU[[2]]
      }
      browser()
      iou_loss <- torch_mean(iou_loss) # $unsqueeze(1)
      with_detect_anomaly({
        iou_loss$backward()
      })
      browser()
      
      #iou_loss$backward()
      optimizer$step()

      if (i%%10 == 0){
        browser()
        #iou_mask = (iou > 0).float()
        mean_iou = torch_sum(iou) / (torch_sum(iou_mask) + 1e-8)
        cat(sprintf("\nEpoch %d, TRAIN: loss:%3f, loss_VOX_IoU: %4f, mean_iou: %4f\n",
                    epoch, i, num_batch, as.array(iou_los), as.array(mean_iou)))
        }
      })
    lr_scheduler$step()
    

    # validate
    net$eval()
    aver_loss = 0
    aver_mean_iou = 0
    with_no_grad({
      coro::loop(for (b in ld_test) {   # for i, data in enumerate(ld_train, 1):
        # browser()
        #data <- train_batch(i) # ??
        
        box <- b[[1]]$to(device=device)
        box <- box$view(c(BATCH_SIZE, -1, 4*2)) %>% torch_transpose(2, 3) 
        
        label <- b[[2]]$to(device=device)
        label <- label$view(c(BATCH_SIZE, -1, 5)) %>% torch_transpose(2, 3) 
        
        optimizer$zero_grad()
        
        pred = net(box)  %>% torch_transpose(2, 3)                             # (B, 5, N)
        browser()
        
        pred = parse_pred(pred)
        iou_loss <- NA
        iou <- NA
        
        if(Loss_Type == "giou"){
          Output_IoU = cal_giou(pred, label, enclosing_type)
          iou_loss <- Output_IoU[[1]] 
          iou <- Output_IoU[[2]]
        }
        
        if(Loss_Type == "diou"){
          Output_IoU = cal_diou(pred, label, enclosing_type)
          iou_loss <- Output_IoU[[1]] 
          iou <- Output_IoU[[2]]
        }
        
        iou_loss <- torch_mean(iou_loss)
        iou_loss$backward()
        optimizer$step()
        
        if (i%%10 == 0){
          browser()
          #iou_mask = (iou > 0).float()
          mean_iou = torch_sum(iou) / (torch_sum(iou_mask) + 1e-8)
          cat(sprintf("\nEpoch %d, TRAIN: loss:%3f, loss_VOX_IoU: %4f, mean_iou: %4f\n",
                      epoch, i, num_batch, as.array(iou_los), as.array(mean_iou)))
          aver_mean_iou <- aver_mean_iou +  as.array(mean_iou)
          cat(sprintf("... validate epoch %d ...", epoch))
          n_iter = (length(ds_test)/BATCH_SIZE)/N_DATA
          cat(sprintf("average loss: %.4f" , (aver_loss/n_iter)))
          cat(sprintf("average iou: %.4f" , (aver_mean_iou/n_iter)))
          print("..............................")
        }
      })
    })
  }
}

# THIS WILL BE USED FOR MINIMUM ENCLOSED BOX... NEED TO CHECK ITS CORRECT
generate_table_out = generate_table()
LINES <- generate_table_out[[1]]
POINTS <- generate_table_out[[2]]
# browser()
# RUN THE IoU ML
Main("giou", "aligned")



#     
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--loss", type=str, default="diou", help="type of loss function. support: diou or giou. [default: diou]")
# #     parser.add_argument("--enclosing", type=str, default="smallest", 
# #         help="type of enclosing box. support: aligned (axis-aligned) or pca (rotated) or smallest (rotated). [default: smallest]")
# #     flags = parser.parse_args()
# #     main(flags.loss, flags.enclosing)







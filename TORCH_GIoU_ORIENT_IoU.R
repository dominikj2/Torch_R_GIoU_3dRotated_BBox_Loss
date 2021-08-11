
#################################################################################################################################
# oriented_iou_loss.py  
enclosing_box <- function(corners1, corners2, enclosing_type ="smallest"){ 
  # browser()
  if(enclosing_type == "aligned"){
    return(enclosing_box_aligned(corners1, corners2))
  }
  
  if(enclosing_type == "pca"){
    return (enclosing_box_pca(corners1, corners2))
  }
  if(enclosing_type == "smallest")
    #browser()
    Enclosed_out <- smallest_bounding_box(torch_cat(list(corners1, corners2), dim=-2))
    return (Enclosed_out)
}


#################################################################################################################################
# box_intersection_2d.py
box1_in_box2 <- function(corners1, corners2){
  # """check if corners of box1 lie in box2
  #   Convention: if a corner is exactly on the edge of the other box, it's also a valid point
  # 
  #   Args:
  #       corners1 (torch.Tensor): (B, N, 4, 2)
  #       corners2 (torch.Tensor): (B, N, 4, 2)
  # 
  #   Returns:
  #       c1_in_2: (B, N, 4) Bool
  #   """
  
  a = corners2[, , 1, ]$unsqueeze(3)  # (B, N, 1, 2)
  b = corners2[, , 2, ]$unsqueeze(3)  # (B, N, 1, 2)
  d = corners2[, , 4, ]$unsqueeze(3)  # (B, N, 1, 2)
  ab = b - a                  # (B, N, 1, 2)
  am = corners1 - a           # (B, N, 4, 2)
  ad = d - a                  # (B, N, 1, 2)
  p_ab = torch_sum(ab * am, dim=-1)       # (B, N, 4)
  norm_ab = torch_sum(ab * ab, dim=-1)    # (B, N, 1)
  p_ad = torch_sum(ad * am, dim=-1)       # (B, N, 4)
  norm_ad = torch_sum(ad * ad, dim=-1)    # (B, N, 1)
  # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
  # also stable with different scale of bboxes
  # browser()
  cond1 = ((p_ab / norm_ab) > - EPSILON) * ((p_ab / norm_ab) < (1 + EPSILON))   # (B, N, 4)
  cond2 = ((p_ad / norm_ad )> - EPSILON) * ((p_ad / norm_ad) < (1 + EPSILON))   # (B, N, 4)
  
  return (cond1*cond2)
}

#################################################################################################################################
# oriented_iou_loss.py
box2corners_th <- function(box){
  # """convert box coordinate to corners
  # 
  #   Args:
  #       box (torch.Tensor): (B, N, 5) with x, y, w, l, alpha
  # 
  #   Returns:
  #       torch.Tensor: (B, N, 4, 2) corners
  #   """
  # browser()
  B = box$size(1)
  x = box[, , 1]$unsqueeze(3)
  y = box[, , 2]$unsqueeze(3)
  w = box[, , 3]$unsqueeze(3)
  l = box[, , 4]$unsqueeze(3)
  alpha = box[, , 5]$unsqueeze(3) # (B, N, 1)
  x4 = torch_tensor(c(0.5, -0.5, -0.5, 0.5))$unsqueeze(1)$unsqueeze(1)$to(device = device) # (1,1,4)
  x4 = x4 * w     # (B, N, 4)
  y4 = torch_tensor(c(0.5, 0.5, -0.5, -0.5))$unsqueeze(1)$unsqueeze(1)$to(device = device)# to(box.device) # (1,1,4) 
  y4 = y4 * l     # (B, N, 4)
  corners = torch_stack(list(x4, y4), dim=-1)     # (B, N, 4, 2)
  sin = torch_sin(alpha)
  cos = torch_cos(alpha)
  row1 = torch_cat(list(cos, sin), dim=-1)
  row2 = torch_cat(list(-sin, cos), dim=-1)       # (B, N, 2)
  rot_T = torch_stack(list(row1, row2), dim=-2)   # (B, N, 2, 2)
  
  rotated = torch_bmm(corners$view(c(-1,4,2)), rot_T$view(c(-1,2,2)))
  rotated = rotated$view(c(B,-1,4,2))          # (B*N, 4, 2) -> (B, N, 4, 2)
  
  # THERE MAY BE A POTENTIAL PROBLEM HERE
  
  rotated[.., 1]$add_(x)
  rotated[.., 2]$add_(y)
  
  # rotated[.., 1] <- rotated[.., 1]$add_(x)
  # rotated[,,, 2] <- rotated[,,, 2]$add_(y) 
  
  # rotated_Cl = torch_clone(rotated)
  # rotated_Cl[.., 1] <- rotated_Cl[.., 1]$add(x) #$add_(x)
  # rotated_Cll = torch_clone(rotated_Cl)
  # rotated_Cll[,,, 2] <- rotated_Cll[,,, 2]$add(y) #$add_(x)
  
  # rotated[.., 1] <- rotated[.., 1] + x # $add_(x)
  # rotated[,,, 2] <- rotated[,,, 2] + y #  $add_(y) 
  return (rotated)
}


#################################################################################################################################
# oriented_iou_loss.py  
cal_iou <- function(box1, box2){
  
  # """calculate iou
  # 
  #   Args:
  #       box1 (torch.Tensor): (B, N, 5)
  #       box2 (torch.Tensor): (B, N, 5)
  #   
  #   Returns:
  #       iou (torch.Tensor): (B, N)
  #       corners1 (torch.Tensor): (B, N, 4, 2)
  #       corners1 (torch.Tensor): (B, N, 4, 2)
  #       union (torch.Tensor): (B, N) area1 + area2 - inter_area
  #   """
  
  corners1 = box2corners_th(box1)
  corners2 = box2corners_th(box2)

  inter_area = oriented_box_intersection_2d(corners1, corners2)        #(B, N)
  # browser()
  area1 = box1[, , 3] * box1[, , 4]
  area2 = box2[, , 3] * box2[, , 4]
  union = area1 + area2 - inter_area[[1]]
  iou = inter_area[[1]] / union
  return(list(iou, corners1, corners2, union))
}

#################################################################################################################################
# oriented_iou_loss.py  

cal_diou <- function(box1, box2, enclosing_type="smallest"){
  # """calculate diou loss
  # 
  #   Args:
  #       box1 (torch.Tensor): [description]
  #       box2 (torch.Tensor): [description]
  #   """
  
  Calc_IoU <- cal_iou(box1, box2)
  iou <- Calc_IoU[[1]]
  corners1 <- Calc_IoU[[2]]
  corners2 <- Calc_IoU[[3]]
  union <- Calc_IoU[[4]]
  enclosing_box_out = enclosing_box(corners1, corners2, enclosing_type)
  
  w = enclosing_box_out[[1]]
  l = enclosing_box_out[[2]]

  c2 = w*w + l*l      # (B, N)
  x_offset = box1[,,0] - box2[,, 0]
  y_offset = box1[,,1] - box2[,, 1]
  d2 = x_offset*x_offset + y_offset*y_offset
  diou_loss = 1. - iou + d2/c2
  return(list(diou_loss, iou))
}

#################################################################################################################################
# oriented_iou_loss.py  
cal_giou <- function(box1, box2, enclosing_type){
  # browser()
  cal_iou_out  = cal_iou(box1, box2)
  iou <- cal_iou_out[[1]]
  corners1 <- cal_iou_out[[2]] 
  corners2 <- cal_iou_out[[3]] 
  union <- cal_iou_out[[4]] 
  enclosing_box_out = enclosing_box(corners1, corners2, enclosing_type) # = "smallest"
  w <-  enclosing_box_out[[1]]
  l <-  enclosing_box_out[[2]]
  
  area_c =  w*l
  # area_c$register_hook(e_hook)
  giou_loss = 1. - iou + ( area_c - union )/area_c
  # giou_loss$register_hook(e_hook)
  # browser()
  return (list(giou_loss, iou) )
}

# #################################################################################################################################
# # oriented_iou_loss.py
# cal_diou_3d <- function(box3d1, box3d2, enclosing_type){
#   # """calculated 3d DIoU loss. assume the 3d bounding boxes are only rotated around z axis
#   #
#   #   Args:
#   #       box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,l,h,alpha)
#   #       box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,l,h,alpha)
#   #       enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".
#   #
#   #   Returns:
#   #       (torch.Tensor): (B, N) 3d DIoU loss
#   #       (torch.Tensor): (B, N) 3d IoU
#   #   """
#   cal_iou_3d_out <- cal_iou_3d(box3d1, box3d2, verbose=TRUE)
#   iou3d <- cal_iou_3d_out[[1]]
#   corners1 <- cal_iou_3d_out[[2]]
#   corners2 <- cal_iou_3d_out[[3]]
#   z_range <- cal_iou_3d_out[[4]]
#   union3d <- cal_iou_3d_out[[5]]
#   browser()
#   enclosing_box_out <- enclosing_box(corners1, corners2, enclosing_type)
#   w <- enclosing_box_out[[1]]
#   l <- enclosing_box_out[[1]]
#   x_offset = box3d1[..,1] - box3d2[.., 1]
#   y_offset = box3d1[..,2] - box3d2[.., 2]
#   z_offset = box3d1[..,3] - box3d2[.., 3]
#   d2 = x_offset*x_offset + y_offset*y_offset + z_offset*z_offset
#   c2 = w*w + l*l + z_range*z_range
#   diou = 1. - iou3d + d2/c2
#   return(list(diou, iou3d))
# }


#################################################################################################################################
# oriented_iou_loss.py
cal_iou_3d <- function(box3d1, box3d2, verbose=TRUE){
  # """calculated 3d iou. assume the 3d bounding boxes are only rotated around z axis
  #
  #   Args:
  #       box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,l,h,alpha)
  #       box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,l,h,alpha)
  #   """
  
  # GET 2D Box
  box1 = box3d1[.., c(1,2,4,5,7)]$to(device=device)    # 2d box  x,y,w,l, alpha
  box2 = box3d2[.., c(1,2,4,5,7)]$to(device=device)    # 2d box  x,y,w,l, alpha
  
  # OVERLAP IN THE Z DIRECTION
  zmin1 = box3d1[.., 3]  - box3d1[.., 6] * 0.5
  zmax1 = box3d1[.., 3] + box3d1[.., 6]  * 0.5
  
  zmin2 = box3d2[.., 3] - box3d2[.., 6] * 0.5
  zmax2 = box3d2[.., 3] + box3d2[.., 6] * 0.5
 
  z_overlap = (torch_min(zmax1, other=zmax2) - torch_max(zmin1, other=zmin2))$clamp_min(0.)
  
  # CALCULATES IoU FOR 2D Box, also outputs corners 
  cal_iou_Out <- cal_iou(box1, box2)        # (B, N)
  iou_2d <- cal_iou_Out[[1]]
  corners1 <- cal_iou_Out[[2]]
  corners2 <- cal_iou_Out[[3]]
  union <- cal_iou_Out[[4]]
  intersection_3d = iou_2d * union * z_overlap # NOTE THAT iou_2d = intersection2D/union SO Intersection3D is intersection2D*z_overlap
  v1 = box3d1[.., 4] * box3d1[.., 5] * box3d1[.., 6] # w,l,h
  v2 = box3d2[.., 4] * box3d2[.., 5] * box3d2[.., 6] #  w,l,h
  union3d = v1 + v2 - intersection_3d
  IoU3D <- intersection_3d/union3d
  if(verbose == TRUE){
    z_range = (torch_max(zmax1, other=zmax2) - torch_min(zmin1, other=zmin2))$clamp_min(0.)
    return (list(IoU3D, corners1, corners2, z_range, union3d))
  }else{
    return (IoU3D)
  }

}

#################################################################################################################################
# oriented_iou_loss.py
cal_giou_3d <- function(box3d1, box3d2, enclosing_type)
  {
  #  """calculated 3d GIoU loss. assume the 3d bounding boxes are only rotated around z axis
  #
  #   Args:
  #       box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,l,h,alpha)
  #       box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,l,h,alpha)
  #       enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".
  #
  #   Returns:
  #       (torch.Tensor): (B, N) 3d GIoU loss
  #       (torch.Tensor): (B, N) 3d IoU
  # """
  # 
  cal_iou_3d_Out <- cal_iou_3d(box3d1, box3d2, verbose=TRUE) # cal_iou for 2d
  iou3d <- cal_iou_3d_Out[[1]]
  corners1 <- cal_iou_3d_Out[[2]]
  corners2 <- cal_iou_3d_Out[[3]]
  z_range <- cal_iou_3d_Out[[4]]
  union3d <- cal_iou_3d_Out[[5]]

  enclosing_box_out <- enclosing_box(corners1, corners2, enclosing_type)
  
  w <- enclosing_box_out[[1]] # return width of the smallest bounding box which encloses two 2D boxes.
  l <- enclosing_box_out[[2]] # return length of the smallest bounding box which encloses two 2D boxes.
  v_c = z_range * w * l

  giou_loss = 1. - iou3d + (v_c - union3d)/v_c
  return (list(giou_loss, iou3d))
}

# box3d1 = array(c(0,0,0,3,3,3,0))
# box3d2 = array(c(1,1,1,2,2,2,pi/3))
# tensor1 = torch_tensor(box3d1, device=device, dtype= torch_float())$unsqueeze(1)$unsqueeze(1)
# tensor2 = torch_tensor(box3d2, device=device, dtype= torch_float())$unsqueeze(1)$unsqueeze(1)
# cal_giou_3d_Output <- cal_giou_3d(tensor1, tensor1)
# giou_loss = cal_giou_3d_Output[[1]]
# iou = cal_giou_3d_Output[[2]]

#################################################################################################################################
# oriented_iou_loss.py
cal_complete_iou_3d <- function(box3d1, box3d2, enclosing_type)
{
  #  """calculated 3d GIoU loss. assume the 3d bounding boxes are only rotated around z axis
  #
  #   Args:
  #       box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,l,h,alpha)
  #       box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,l,h,alpha)
  #       enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".
  #
  #   Returns:
  #       (torch.Tensor): (B, N) 3d GIoU loss
  #       (torch.Tensor): (B, N) 3d IoU
  # """
  # browser()
  
  # NORMALISING x,y,z,w,l,h
  box3d1_Norm <- torch_clone(box3d1)
  box3d2_Norm <- torch_clone(box3d2)
  
  box3d1_Norm[,,1:6] = torch_sigmoid(box3d1[,,1:6])
  box3d2_Norm[,,1:6] = torch_sigmoid(box3d2[,,1:6])
  
  cal_iou_3d_Out <- cal_iou_3d(box3d1_Norm, box3d2_Norm, verbose=TRUE) # cal_iou for 2d
  iou3d <- cal_iou_3d_Out[[1]]
  S = 1. -iou3d
  corners1 <- cal_iou_3d_Out[[2]]
  corners2 <- cal_iou_3d_Out[[3]]
  z_range <- cal_iou_3d_Out[[4]]  # DOM DOM DOM !!! MAKE SURE THAT THE Z RANGE IS CORRECT ...
  union3d <- cal_iou_3d_Out[[5]]

  enclosing_box_out <- enclosing_box(corners1, corners2, enclosing_type)
  w <- enclosing_box_out[[1]] # return width of the smallest bounding box which encloses two 2D boxes.
  l <- enclosing_box_out[[2]] # return length of the smallest bounding box which encloses two 2D boxes.
  
  # # THIS IS ADOPTED FROM CIoU code
  # w <- torch_exp(w)
  # l <- torch_exp(w)
  # z_range <- torch_exp(z_range)
  
  x_offset = box3d1_Norm[..,1] - box3d2_Norm[.., 1]
  y_offset = box3d1_Norm[..,2] - box3d2_Norm[.., 2]
  z_offset = box3d1_Norm[..,3] - box3d2_Norm[.., 3]
  d2 = x_offset*x_offset + y_offset*y_offset + z_offset*z_offset
  c2 = w*w + l*l + z_range*z_range
  D = torch_pow(d2,2)/torch_pow(c2,2) # SQUARING USING ZHENG 2021 formula 
  
  # D_a <- as.array(D_Norm$to(device = "cpu"))
  # browser()
  # DOM DOM DOM !!! D NEEDS TO BE NORMALISED
  
  # RATIO MEASURE
  V = (4 / (pi ** 2)) * torch_pow((torch_atan(w / l) - torch_atan(w / l)), 2)
  
  with_no_grad(
    S_TrueFalse <- (iou3d >= 0.5)$to(dtype= torch_float()) # 
  )
  
  alpha = S_TrueFalse*V/(1-iou3d- V)
  
    # if(iou3d >= 0.5){
    #   S =S$to(dtype= torch_float())
    #   alpha = S*V/(1-iou- V)
    # }else{
    #   alpha = 0
    # }
  
  cIoU = S + D + alpha*V # (v_c - union3d)/v_c
  return (list(cIoU, iou3d))
}



# BELOW ASSUMES THE BOXES ARE AXIS ALIGNED... WON'T WORK FOR ROTATING BOXES
# cal_complete_iou_3d_axis_aligned  <- function(bboxes1, bboxes2){ 
#   browser()
#   bboxes1 = torch_sigmoid(bboxes1)                        # (x,y,z,w,l,h,alpha)
#   bboxes2 = torch_sigmoid(bboxes2)
#   # rows = dim(bboxes1)[[0]]
#   # cols = dim(bboxes2)[[0]]
#   # cious = torch_zeros(c(crows, cols))
#   # if(rows * cols == 0){
#   #   return(cious)
#   # }
#   # exchange <-  FALSE
#   # if (dim(bboxes1)[[0]] > dim(bboxes2)[[0]]){
#   #   bboxes1Temp <- bboxes1
#   #   bboxes1 <- bboxes2 
#   #   bboxes2 <- bboxes1Temp 
#   #   cious = torch_zeros(c(cols, rows))
#   #   exchange = TRUE
#   # }
# 
#   # CALCULATE THE VOLUME OF EACH BOUNDING BOX
#   w1 = torch_exp(bboxes1[,, 4]) 
#   h1 = torch_exp(bboxes1[,, 5])
#   l1 = torch_exp(bboxes1[,, 6])
#   w2 = torch_exp(bboxes2[,, 4])
#   h2 = torch_exp(bboxes2[,, 5])
#   l2 = torch_exp(bboxes1[,, 6])
#   vol1 = w1 * h1 * l1
#   vol2 = w2 * h2 * l2
#   
#   center_x1 = bboxes1[,, 1]
#   center_y1 = bboxes1[,, 2]
#   center_z1 = bboxes1[,, 3]
#   center_x2 = bboxes2[,, 1]
#   center_y2 = bboxes2[,, 2]
#   center_z2 = bboxes2[,, 3]
#   
#   inter_xl = torch_max(center_x1 - w1 / 2,center_x2 - w2 / 2)
#   inter_xr = torch_min(center_x1 + w1 / 2,center_x2 + w2 / 2)
#   inter_yl = torch_max(center_y1 - h1 / 2,center_y2 - h2 / 2)
#   inter_yr = torch_min(center_y1 + h1 / 2,center_y2 + h2 / 2)
#   inter_zl = torch_max(center_y1 - h1 / 2,center_y2 - h2 / 2)
#   inter_zr = torch_min(center_y1 + h1 / 2,center_y2 + h2 / 2)
#   
#   inter_area = torch_clamp((inter_xr - inter_xl),min=0) * torch_clamp((inter_yr - inter_yl),min=0) * torch_clamp((inter_zr - inter_zl),min=0)
#   
#   c_l = torch_min(center_x1 - w1 / 2,center_x2 - w2 / 2)
#   c_r = torch_max(center_x1 + w1 / 2,center_x2 + w2 / 2)
#   c_t = torch_min(center_y1 - h1 / 2,center_y2 - h2 / 2)
#   c_b = torch_max(center_y1 + h1 / 2,center_y2 + h2 / 2)
#   
#   inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
#   c_diag = torch_clamp((c_r - c_l),min=0)**2 + torch_clamp((c_b - c_t),min=0)**2
#   
#   union = vol1+vol2-inter_area
#   u = (inter_diag) / c_diag
#   iou = inter_area / union
#   V = (4 / (pi ** 2)) * torch_pow((torch_atan(w2 / h2) - torch_atan(w1 / h1)), 2)
#   with_no_grad():
#     S = (iou>0.5)$to(dtype = tensor_float())
#   alpha= S*V/(1-iou+V)
#   cious = iou - u - alpha * V
#   cious = torch_clamp(cious,min=-1.0,max = 1.0)
#   if(exchange){
#     cious = cious.T
#     }
#   return (torch_sum(1-cious))
# }


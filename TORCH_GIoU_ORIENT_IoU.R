
#################################################################################################################################
# oriented_iou_loss.py  
enclosing_box <- function(corners1, corners2, enclosing_type ="smallest"){
  if(enclosing_type == "aligned"){
    return(enclosing_box_aligned(corners1, corners2))
  }
  
  if(enclosing_type == "pca"){
    return (enclosing_box_pca(corners1, corners2))
  }
  if(enclosing_type == "smallest")
    return (smallest_bounding_box(torch_cat(list(corners1, corners2), dim=-2)))
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
  #       box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha
  # 
  #   Returns:
  #       torch.Tensor: (B, N, 4, 2) corners
  #   """
  
  B = box$size(1)
  x = box[, , 1]$unsqueeze(3)
  y = box[, , 2]$unsqueeze(3)
  w = box[, , 3]$unsqueeze(3)
  h = box[, , 4]$unsqueeze(3)
  alpha = box[, , 5]$unsqueeze(3) # (B, N, 1)
  x4 = torch_tensor(c(0.5, -0.5, -0.5, 0.5))$unsqueeze(1)$unsqueeze(1)$to(device = device) # (1,1,4)
  x4 = x4 * w     # (B, N, 4)
  y4 = torch_tensor(c(0.5, 0.5, -0.5, -0.5))$unsqueeze(1)$unsqueeze(1)$to(device = device)# to(box.device) # (1,1,4) 
  y4 = y4 * h     # (B, N, 4)
  corners = torch_stack(list(x4, y4), dim=-1)     # (B, N, 4, 2)
  sin = torch_sin(alpha)
  cos = torch_cos(alpha)
  row1 = torch_cat(list(cos, sin), dim=-1)
  row2 = torch_cat(list(-sin, cos), dim=-1)       # (B, N, 2)
  rot_T = torch_stack(list(row1, row2), dim=-2)   # (B, N, 2, 2)
  
  rotated = torch_bmm(corners$view(c(-1,4,2)), rot_T$view(c(-1,2,2)))
  rotated = rotated$view(c(B,-1,4,2))          # (B*N, 4, 2) -> (B, N, 4, 2)
  
  # THERE MAY BE A POTENTIAL PROBLEM HERE
  # rotated[.., 1] <- rotated[.., 1]$add_(x)
  # rotated[,,, 2] <- rotated[,,, 2]$add_(y) 
  #browser()
  rotated_Cl = torch_clone(rotated)
  rotated_Cl[.., 1] <- rotated_Cl[.., 1]$add(x) #$add_(x)
  rotated_Cll = torch_clone(rotated_Cl)
  rotated_Cll[,,, 2] <- rotated_Cll[,,, 2]$add(y) #$add_(x)
  
  # rotated[.., 1] <- rotated[.., 1] + x # $add_(x)
  # rotated[,,, 2] <- rotated[,,, 2] + y #  $add_(y) 
  return (rotated_Cll)
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
  #       U (torch.Tensor): (B, N) area1 + area2 - inter_area
  #   """
  # browser()
  corners1 = box2corners_th(box1)
  corners2 = box2corners_th(box2)
  
  inter_area = oriented_box_intersection_2d(corners1, corners2)        #(B, N)
  # browser()
  area1 = box1[, , 3] * box1[, , 4]
  area2 = box2[, , 3] * box2[, , 4]
  u = area1 + area2 - inter_area[[1]]
  iou = inter_area[[1]] / u
  return(list(iou, corners1, corners2, u))
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
  u <- Calc_IoU[[4]]
  enclosing_box_out = enclosing_box(corners1, corners2, enclosing_type)
  
  w = enclosing_box_out[[1]]
  h = enclosing_box_out[[2]]
  browser()
  c2 = w*w + h*h      # (B, N)
  x_offset = box1[,,0] - box2[,, 0]
  y_offset = box1[,,1] - box2[,, 1]
  d2 = x_offset*x_offset + y_offset*y_offset
  diou_loss = 1. - iou + d2/c2
  return(list(diou_loss, iou))
}

#################################################################################################################################
# oriented_iou_loss.py  
cal_giou <- function(box1, box2, enclosing_type){
  #browser()
  cal_iou_out  = cal_iou(box1, box2)
  iou <- cal_iou_out[[1]]
  corners1 <- cal_iou_out[[2]] 
  corners2 <- cal_iou_out[[3]] 
  u <- cal_iou_out[[4]]
  enclosing_box_out = enclosing_box(corners1, corners2, enclosing_type= "smallest")
  w <-  enclosing_box_out[[1]]
  h <-  enclosing_box_out[[2]]
  area_c =  w*h
  giou_loss = 1. - iou + ( area_c - u )/area_c
  # browser()
  return (list(giou_loss, iou) )
}

# #################################################################################################################################
# # oriented_iou_loss.py 
# cal_diou_3d <- function(box3d1, box3d2, enclosing_type="smallest"){
#   # """calculated 3d DIoU loss. assume the 3d bounding boxes are only rotated around z axis
#   # 
#   #   Args:
#   #       box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
#   #       box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
#   #       enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".
#   # 
#   #   Returns:
#   #       (torch.Tensor): (B, N) 3d DIoU loss
#   #       (torch.Tensor): (B, N) 3d IoU
#   #   """
#   cal_iou_3d_out <- cal_iou_3d(box3d1, box3d2, verbose=True)
#   iou3d <- cal_iou_3d_out[[1]]
#   corners1 <- cal_iou_3d_out[[2]]
#   corners2 <- cal_iou_3d_out[[3]]
#   z_range <- cal_iou_3d_out[[4]]
#   u3d <- cal_iou_3d_out[[5]]
#   
#   enclosing_box_out <- enclosing_box(corners1, corners2, enclosing_type)
#   w <- enclosing_box_out[[1]]
#   h <- enclosing_box_out[[1]]
#   x_offset = box3d1[..,1] - box3d2[.., 1]
#   y_offset = box3d1[..,2] - box3d2[.., 2]
#   z_offset = box3d1[..,3] - box3d2[.., 3]
#   d2 = x_offset*x_offset + y_offset*y_offset + z_offset*z_offset
#   c2 = w*w + h*h + z_range*z_range
#   diou = 1. - iou3d + d2/c2
#   return(list(diou, iou3d))
# }
#  
# 
# #################################################################################################################################
# # oriented_iou_loss.py 
# cal_iou_3d <- function(box3d1, box3d2, verbose=TRUE){
#   # """calculated 3d iou. assume the 3d bounding boxes are only rotated around z axis
#   # 
#   #   Args:
#   #       box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
#   #       box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
#   #   """
#   box1 = box3d1[.., c(1,2,4,5,7)]$to(device=device)    # 2d box
#   box2 = box3d2[.., c(1,2,4,5,7)]$to(device=device)  
#   zmax1 = box3d1[.., 3] + box3d1[.., 6] * 0.5
#   zmin1 = box3d1[.., 3] - box3d1[.., 6] * 0.5
#   zmax2 = box3d2[.., 3] + box3d2[.., 6] * 0.5
#   zmin2 = box3d2[.., 3] - box3d2[.., 6] * 0.5
#   z_overlap = (torch_min(zmax1, zmax2) - torch_max(zmin1, zmin2))$clamp_min(0.)
#   cal_iou_Out <- cal_iou(box1, box2)        # (B, N)
#   iou_2d <- cal_iou_Out[[1]] 
#   corners1 <- cal_iou_Out[[2]] 
#   corners2 <- cal_iou_Out[[3]] 
#   browser()
#   u <- cal_iou_Out[[4]] 
#   intersection_3d = iou_2d * u * z_overlap
#   v1 = box3d1[.., 4] * box3d1[.., 5] * box3d1[.., 6]
#   v2 = box3d2[.., 4] * box3d2[.., 5] * box3d2[.., 6]
#   u3d = v1 + v2 - intersection_3d
#   if(verbose == TRUE){
#     z_range = (torch_max(zmax1, zmax2) - torch_min(zmin1, zmin2))$clamp_min(0.)
#     return (list(intersection_3d / u3d, corners1, corners2, z_range, u3d))
#   }else{
#     return (intersection_3d / u3d)
#   }
#     
# }
# 
# #################################################################################################################################
# # oriented_iou_loss.py 
# cal_giou_3d <- function(box3d1, box3d2, enclosing_type="smallest")
#   {
#   #  """calculated 3d GIoU loss. assume the 3d bounding boxes are only rotated around z axis
#   # 
#   #   Args:
#   #       box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
#   #       box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
#   #       enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".
#   # 
#   #   Returns:
#   #       (torch.Tensor): (B, N) 3d GIoU loss
#   #       (torch.Tensor): (B, N) 3d IoU
#   # """
# 
#   cal_iou_3d_Out <- cal_iou_3d(box3d1, box3d2, verbose=True)
#   iou3d <- cal_iou_3d_Out[[1]]
#   corners1 <- cal_iou_3d_Out[[2]] 
#   corners2 <- cal_iou_3d_Out[[3]]
#   z_range <- cal_iou_3d_Out[[4]]
#   u3d <- cal_iou_3d_Out[[5]]
#     
# 
#   enclosing_box_out <- enclosing_box(corners1, corners2, enclosing_type)
#   w <- enclosing_box_out[[1]]
#   h <- enclosing_box_out[[2]]
#   v_c = z_range * w * h
#   giou_loss = 1. - iou3d + (v_c - u3d)/v_c
#   return (list(giou_loss, iou3d))
# }


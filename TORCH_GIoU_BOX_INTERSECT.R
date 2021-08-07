# box_intersection_2d.py
#################################################################################################################################

sort_indices <- function(vertices, mask){
  # """[summary]
  # 
  #   Args:
  #       vertices (torch.Tensor): float (B, N, 24, 2)
  #       mask (torch.Tensor): bool (B, N, 24)
  # 
  #   Returns:
  #       sorted_index: bool (B, N, 9)
  #   
  #   Note:
  #       why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
  #       the index should have following structure:
  #           (A, B, C, ... , A, X, X, X) 
  #       and X indicates the index of arbitary elements in the last 16 (intersections not corners) with 
  #       value 0 and mask False. (cause they have zero value and zero gradient)
  #   """

  num_valid = torch_sum(mask$to(dtype = torch_int()), dim=3)$to(dtype = torch_int())    # (B, N)
  mean = (torch_sum(vertices * mask$to(dtype = torch_float())$unsqueeze(-1), dim=3, keepdim=TRUE) / num_valid$unsqueeze(-1)$unsqueeze(-1))
  vertices_normalized = (vertices - mean)      # normalization makes sorting easier
  # browser()
  sorted_vertices <-contrib_sort_vertices(vertices_normalized, mask, num_valid)$to(dtype = torch_long())

  # sorted_vertices <- contrib_sort_vertices(vertices_normalized1, mask1, num_valid1)$to(dtype = torch_long())
  return (sorted_vertices)
}

#################################################################################################################################

build_vertices <- function(corners1, corners2, 
                           c1_in_2, c2_in_1, 
                           inters, mask_inter){
  
  # """find vertices of intersection area
  # 
  #   Args:
  #       corners1 (torch.Tensor): (B, N, 4, 2)
  #       corners2 (torch.Tensor): (B, N, 4, 2)
  #       c1_in_2 (torch.Tensor): Bool, (B, N, 4)
  #       c2_in_1 (torch.Tensor): Bool, (B, N, 4)
  #       inters (torch.Tensor): (B, N, 4, 4, 2)
  #       mask_inter (torch.Tensor): (B, N, 4, 4)
  #   
  #   Returns:
  #       vertices (torch.Tensor): (B, N, 24, 2) vertices of intersection area. only some elements are valid
  #       mask (torch.Tensor): (B, N, 24) indicates valid elements in vertices
  #   """
  # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0). 
  # can be used as trick
  
  B = corners1$size(1)
  N = corners1$size(2)
  #
  vertices = torch_cat(c(corners1, corners2, inters$view(c(B, N, -1, 2))), dim=3) # (B, N, 4+4+16, 2)
  mask = torch_cat(c(c1_in_2, c2_in_1, mask_inter$view(c(B, N,-1))), dim=3) # Bool (B, N, 4+4+16)
  # browser()
  return (list(vertices, mask))
}

#################################################################################################################################

box_in_box_th <- function(corners1, corners2){
  # """check if corners of two boxes lie in each other
  # 
  #   Args:
  #       corners1 (torch.Tensor): (B, N, 4, 2)
  #       corners2 (torch.Tensor): (B, N, 4, 2)
  # 
  #   Returns:
  #       c1_in_2: (B, N, 4) Bool. i-th corner of box1 in box2
  #       c2_in_1: (B, N, 4) Bool. i-th corner of box2 in box1
  #   """
  c1_in_2 = box1_in_box2(corners1, corners2)
  c2_in_1 = box1_in_box2(corners2, corners1)
  return(list(c1_in_2, c2_in_1))
}

#################################################################################################################################
# box_intersection_2d.py
box_intersection_th <- function(corners1, corners2){
  # """find intersection points of rectangles
  #   Convention: if two edges are collinear, there is no intersection point
  # 
  #   Args:
  #       corners1 (torch.Tensor): B, N, 4, 2
  #       corners2 (torch.Tensor): B, N, 4, 2
  # 
  #   Returns:
  #       intersectons (torch.Tensor): B, N, 4, 4, 2
  #       mask (torch.Tensor) : B, N, 4, 4; bool
  #   """
  # build edges from corners
  #browser()
  line1 = torch_cat(c(corners1, corners1[, , c(2, 3, 4, 1),]), dim=4) # B, N, 4, 4: Batch, Box, edge, point
  line2 = torch_cat(c(corners2, corners2[, , c(2, 3, 4, 1),]), dim=4)
  # duplicate data to pair each edges from the boxes
  # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
  #browser()
  line1_ext = line1$unsqueeze(4)$'repeat'(c(1,1,1,4,1))
  line2_ext = line2$unsqueeze(3)$'repeat'(c(1,1,4,1,1))
  
  x1 = line1_ext[.., 1]
  y1 = line1_ext[.., 2]
  x2 = line1_ext[.., 3]
  y2 = line1_ext[.., 4]
  x3 = line2_ext[.., 1]
  y3 = line2_ext[.., 2]
  x4 = line2_ext[.., 3]
  y4 = line2_ext[.., 4]
  # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
  num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)     
  den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
  
  # THERE MAY BE A POTENTIAL PROBLEM HERE
  
  t = den_t / num
  t[num == .0] = -1.
  # tt<- torch_where(num == 0.0, t, torch_tensor(-1.0, device=device)$to(dtype = torch_float()))
  #browser()
  mask_t = (t > 0) * (t < 1)                # intersection on line segment 1
  den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
  u = -den_u / num
  #browser()
  u[num == .0] = -1.
  #uu <- torch_where(num == 0.0, u, torch_tensor(-1.0, device=device)$to(dtype = torch_float()))
  mask_u = (u > 0) * (u < 1)                # intersection on line segment 2
  mask = mask_t * mask_u 
  t = den_t / (num + EPSILON)                 # overwrite with EPSILON. otherwise numerically unstable
  intersections = torch_stack(c(x1 + t*(x2-x1), y1 + t*(y2-y1)), dim=-1)
  intersections = intersections * mask$to(dtype = torch_float())$unsqueeze(-1)
  return(list(intersections, mask))
}


# FF = torch_rand(2,2, requires_grad=FALSE)
# FF[1,1] = 0
# FF[FF == .0] = -1.
#################################################################################################################################
# box_intersection_2d.py
calculate_area <- function(idx_sorted, vertices){
  # """calculate area of intersection
  # 
  #   Args:
  #       idx_sorted (torch.Tensor): (B, N, 9)
  #       vertices (torch.Tensor): (B, N, 24, 2)
  #   
  #   return:
  #       area: (B, N), area of intersection
  #       selected: (B, N, 9, 2), vertices of polygon with zero padding 
  #   """
  

 
  idx_ext = idx_sorted$unsqueeze(-1)$'repeat'(c(1,1,1,2))
 
  # ERROR CATCHING
  # print(idx_ext$min())
  # print(idx_ext$max())
  # print(dim(vertices))
  # print(dim(idx_ext))
  # print(min(as.vector(as.array(idx_ext$to(device= "cpu")))))
  # print(max(as.vector(as.array(idx_ext$to(device= "cpu")))))
  #browser()
  if(min(as.vector(as.array(idx_ext$to(device= "cpu")))) == 0){browser()}
  # browser()
  selected = torch_gather(vertices, 3, idx_ext)
  
  
  # t = torch_tensor(matrix(c(1,2,3,4,5,6,7,8), ncol = 2, byrow = TRUE))
  # TT <- torch_tensor(matrix(c(1,1,2,1,2,2,1,1), ncol = 2, byrow=TRUE), dtype = torch_int64())
  # TTT <- torch_gather(t, 2, TT)
  
  
  
  D_s <- dim(selected)
  total = selected[, , 1:(D_s[3]-1), 1]*selected[, , 2:D_s[3], 2] - selected[, , 1:(D_s[3]-1), 2]*selected[, , 2:D_s[3], 1] # head( 1:4, -1)

  total = torch_sum(total, dim=3)
  area = torch_abs(total) / 2
  return(list(area, selected))
}


#################################################################################################################################
# box_intersection_2d.py
oriented_box_intersection_2d <- function(corners1, corners2){
  # """calculate intersection area of 2d rectangles 
  # 
  #   Args:
  #       corners1 (torch.Tensor): (B, N, 4, 2)
  #       corners2 (torch.Tensor): (B, N, 4, 2)
  # 
  #   Returns:
  #       area: (B, N), area of intersection
  #       selected: (B, N, 9, 2), vertices of polygon with zero padding 
  #   """

  box_intersection_th_out <- box_intersection_th(corners1, corners2)

  inters <- box_intersection_th_out[[1]]
  mask_inter <- box_intersection_th_out[[2]]
  box_in_box_th_out = box_in_box_th(corners1, corners2)
  c12 <- box_in_box_th_out[[1]]
  c21 <- box_in_box_th_out[[2]]

  # vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
  build_vertices_out <- build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
  vertices <- build_vertices_out[[1]]
  mask <- build_vertices_out[[2]]

  sorted_indices = sort_indices(vertices, mask)
  #browser()
  # if(as.array(sorted_indices$min()$to(device="cpu")) == 0){browser()}
  # browser()
  sorted_indices = sorted_indices + 1L # WORK AROUND
  Output <- calculate_area(sorted_indices, vertices)

  return(Output)
}


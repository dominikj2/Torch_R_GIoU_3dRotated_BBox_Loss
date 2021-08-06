# #################################################################################################################################

# min_enclosing_box.py
generate_table <- function(){
  # """generate candidates of hull polygon edges and the the other 6 points
  #
  #   Returns:
  #       lines: (24, 2)
  #       points: (24, 6)
  #   """
  skip = list(c(1,3), c(2,4), c(6,8), c(5,7))     # impossible hull edge
  line = list()
  points = list()
  
  all_except_two <- function(o1, o2){
    a = list()
    for (i in 1:8){
      if (i != o1 & i != o2){
        a[[i]] <- i # NOT SURE IF THIS IS CORRECT
      }
    }
    return (a)
  }
  
  for (i in 1:7){
    for (j in (i+1):8){
      
      if (!(list(c(i, j)) %in% skip)){ # DOM DOM DOM !!! CHECK THIS IS CORRECT
        line[[length(line) + 1]] <- c(i, j)
        points[[length(points) + 1]] <- all_except_two(i, j)
        
      }
    }
    # print(line)
    # browser() 
  }
  return(list(line, points))
}

# LINES = np.array(LINES).astype(np.int)
# POINTS = np.array(POINTS).astype(np.int)

#################################################################################################################################
# min_enclosing_box.py
gather_lines_points <- function(corners){
  # """get hull edge candidates and the rest points using the index
  # 
  #   Args:
  #       corners (torch.Tensor): (..., 8, 2)
  #   
  #   Return: 
  #       lines (torch.Tensor): (..., 24, 2, 2)
  #       points (torch.Tensor): (..., 24, 6, 2)
  #       idx_lines (torch.Tensor): Long (..., 24, 2, 2)
  #       idx_points (torch.Tensor): Long (..., 24, 6, 2)
  #   """
  dim_lngth = length(dim(corners))
  # browser()
  idx_lines = torch_tensor(unlist(LINES))$view(c(length(LINES),2))$to(dtype = torch_long())$unsqueeze(-1)$to(device=device)       # (24, 2, 1)
  idx_points = torch_tensor(unlist(POINTS))$view(c(length(POINTS),6))$to(dtype = torch_long())$unsqueeze(-1)$to(device=device)     # (24, 6, 1)
  idx_lines = idx_lines$'repeat'(c(1,1,2))                                  # (24, 2, 2)
  idx_points = idx_points$'repeat'(c(1,1,2))                                   # (24, 6, 2)
  if (dim_lngth > 2){
    # browser()
    for (i in 1:(dim_lngth-2)){
      idx_lines = torch_unsqueeze(idx_lines, 1)
      idx_points = torch_unsqueeze(idx_points, 1)
    } # FOR LOOP
    #browser()
    idx_points = idx_points$'repeat'(c(corners$size()[1:2], 1, 1, 1))  # DOM DOM DOM * URL -->         # (..., 24, 2, 2) for * see https://stackoverflow.com/questions/5239856/asterisk-in-function-call
    idx_lines = idx_lines$'repeat'(c(corners$size()[1:2], 1, 1, 1))    # DOM DOM DOM *  URL -->          # (..., 24, 6, 2)  
  } # IF LOOP
  
  corners_ext = corners$unsqueeze(-3)$'repeat'( c(rep(1,(dim_lngth-2)), 24, 1, 1)) # DOM DOM DOM * URL -->       # (..., 24, 8, 2)
  # browser()
  lines = torch_gather(corners_ext, dim=-2, index=idx_lines)                  # (..., 24, 2, 2)
  points = torch_gather(corners_ext, dim=-2, index=idx_points)                # (..., 24, 6, 2)
  return (list(lines, points, idx_lines, idx_points))
}


#################################################################################################################################
# min_enclosing_box.py
point_line_distance_range <- function(lines, points){
  # """calculate the maximal distance between the points in the direction perpendicular to the line
  #   methode: point-line-distance
  # 
  #   Args:
  #       lines (torch.Tensor): (..., 24, 2, 2)
  #       points (torch.Tensor): (..., 24, 6, 2)
  #   
  #   Return:
  #       torch.Tensor: (..., 24)
  #   """
  x1 = lines[.., 1, 1]$unsqueeze(-1)  # DOM DOM DOM MAYBE unsqueeze       # (..., 24, 1)
  y1 = lines[.., 1, 2]$unsqueeze(-1)       # (..., 24, 1)
  x2 = lines[.., 2, 1]$unsqueeze(-1)       # (..., 24, 1)
  y2 = lines[.., 2, 2] $unsqueeze(-1)      # (..., 24, 1)
  x = points[.., 1]            # (..., 24, 6)
  y = points[.., 2]            # (..., 24, 6)
  
  den = (y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1
  # den$register_hook(max_hook1)
  
  # NOTE: the backward pass of torch.sqrt(x) generates NaN if x==0
  
  num = torch_sqrt( (y2-y1)$square() + (x2-x1)$square() + 1e-14 )
  # num$register_hook(max_hook1)
  
  d = den/num         # (..., 24, 6)
  #d$register_hook(max_hook1)

  d_max = d$max(dim=-1)[[1]] # torch_max(d, dim =-1)[[1]]       # (..., 24)
  #d_max$register_hook(max_hook1)

  d_min = d$min(dim=-1)[[1]]  # torch_min(d, dim=-1)[[1]]      # (..., 24)
  #d_min$register_hook(max_hook1)
  
  d1 = d_max - d_min  
  #d1$register_hook(max_hook1)
  
  # suppose points on different side
  d_abs = d$abs()
  #d_abs$register_hook(max_hook1)
  
  # d2 = torch_max(d_abs, dim=-1)[[1]]      # or, all points are on the same side
  d2 = d_abs$max(dim=-1)[[1]]
  #d2$register_hook(max_hook1)

  output = torch_max(d1, other = d2)
  #output$register_hook(max_hook1)

  # NOTE: if x1 = x2 and y1 = y2, this will return 0
  return (output)
}


#################################################################################################################################
# min_enclosing_box.py 
point_line_projection_range <- function(lines, points){
  # """calculate the maximal distance between the points in the direction parallel to the line
  #   methode: point-line projection
  # 
  #   Args:
  #       lines (torch.Tensor): (..., 24, 2, 2)
  #       points (torch.Tensor): (..., 24, 6, 2)
  #   
  #   Return:
  #       torch.Tensor: (..., 24)
  #  """
  # browser()
  x1 = lines[.., 1, 1]$unsqueeze(-1)  # DOM DOM DOM MAYBE unsqueeze     # (..., 24, 1)
  y1 = lines[.., 1, 2]$unsqueeze(-1)       # (..., 24, 1)
  x2 = lines[.., 2, 1]$unsqueeze(-1)       # (..., 24, 1)
  y2 = lines[.., 2, 2]$unsqueeze(-1)       # (..., 24, 1)
  k = (y2 - y1)/(x2 - x1 + 1e-8)      # (..., 24, 1)
  vec = torch_cat(c(torch_ones_like(k, dtype=k$dtype), k), dim=-1)  # (..., 24, 2) torch_float()
  vec = vec$unsqueeze(-2)             # (..., 24, 1, 2)  
  points_ext = torch_cat(c(lines, points), dim=-2)         # (..., 24, 8), consider all 8 points
  den = torch_sum(points_ext * vec, dim=-1)               # (..., 24, 8) 
  proj = den / torch_norm(vec, dim=-1, keepdim=FALSE)     # (..., 24, 8)

  proj_max = proj$max(dim=-1)[[1]]  #torch_max(proj, dim =-1)[[1]]       # (..., 24)
  #proj_max$register_hook(max_hook2)

  proj_min = proj$min(dim=-1)[[1]]  #torch_min(proj, dim =-1)[[1]]      # (..., 24)
  #proj_min$register_hook(max_hook2)
  
  Output <- proj_max - proj_min
  #Output$register_hook(max_hook2)

  return (Output) #
}


#################################################################################################################################
# min_enclosing_box.py
smallest_bounding_box <- function(corners, verbose=FALSE){
  # """return width and length of the smallest bouding box which encloses two boxes.
  # 
  #   Args:
  #       lines (torch.Tensor): (..., 24, 2, 2)
  #       verbose (bool, optional): If True, return area and index. Defaults to False.
  # 
  #   Returns:
  #       (torch.Tensor): width (..., 24)
  #       (torch.Tensor): height (..., 24)
  #       (torch.Tensor): area (..., )
  #       (torch.Tensor): index of candiatae (..., )
  #   """
  gather_lines_points_out <- gather_lines_points(corners)
  lines <- gather_lines_points_out[[1]] 
  points <- gather_lines_points_out[[2]]  
  idx_lines <- gather_lines_points_out[[3]]  
  idx_points <- gather_lines_points_out[[4]] 
  # browser()
  proj = point_line_projection_range(lines, points)   # (..., 24)
  dist = point_line_distance_range(lines, points)     # (..., 24)

  area = proj * dist
  
  # remove area with 0 when the two points of the line have the same coordinates
  zero_mask = (area == 0)$to(dtype = corners$dtype) #type(corners.dtype)
  
  fake = torch_ones_like(zero_mask, dtype=corners$dtype)* 1e8 * zero_mask
  area_cl <- torch_clone(area)    # IN-PLACE OPERATION FIX
  area_cl <- area_cl$add(fake)       # $add_(fake) add large value to zero_mask
  area_min_out <- torch_min(area_cl, dim=-1, keepdim=TRUE)     # (..., 1)
  area_min <- area_min_out[[1]]$squeeze(-1)$to(dtype = torch_float())
  idx  <- area_min_out[[2]]

  w = torch_gather(proj, dim=-1, index=idx)$squeeze(-1)$to(dtype = torch_float())
  h = torch_gather(dist, dim=-1, index=idx)$squeeze(-1)$to(dtype = torch_float())          # (..., 1)
  # w = w$squeeze(-1)$to(dtype = torch_float())
  # h = h$squeeze(-1)$to(dtype = torch_float())
  # area_min = area_min$squeeze(-1)$to(dtype = torch_float())
  if(verbose == TRUE){
    return (list(w, h, area_min, idx$squeeze(-1)))
  }
  else{
    return (list(w, h)) 
  }
}

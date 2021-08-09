
#################################################################################################################################
# utiles.py
box2corners <- function(x, y, w, h, alpha){  # WORKING 
  "box parameters to four box corners"
  x4 = c(0.5, -0.5, -0.5, 0.5) * w
  y4 = c(0.5, 0.5, -0.5, -0.5) * h
  corners = cbind(x4, y4)
  sin = sin(alpha)
  cos = cos(alpha)
  R = matrix(c(cos, -sin, sin, cos), nrow=2, ncol=2)
  rotated = corners %*% R
  # browser()
  rotated[, 1] <- rotated[, 1] + x 
  rotated[, 2] <- rotated[, 2] + y
  return(rotated)
}
# Test <- box2corners(box1[1], box1[2], box1[3], box1[4], box1[5])
# 
box2corners_3d <- function(x, y, z, w, h, l, alpha){  # WORKING
  "box parameters to four box corners"
  x4 = c(0.5, -0.5, -0.5, 0.5) * w
  y4 = c(0.5, 0.5, -0.5, -0.5) * h
  #z4 = c(0.5, 0.5, -0.5, -0.5) * l
  corners = cbind(x4, y4)
  sin = sin(alpha)
  cos = cos(alpha)
  R = matrix(c(cos, -sin, sin, cos), nrow=2, ncol=2)
  rotated = corners %*% R
  #browser()
  # browser()
  rotated[, 1] <- rotated[, 1] + x
  rotated[, 2] <- rotated[, 2] + y
  rotated_botz <- cbind(rotated, (z - 0.5*l))
  rotated_topz <- cbind(rotated, (z + 0.5*l))
  rotated <- rbind(rotated_botz, rotated_topz)
  return(rotated)
}

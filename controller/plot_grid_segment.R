#!/usr/bin/env Rscript

# 确保sf, terra, dplyr 包已安装
if (!requireNamespace("sf", quietly = TRUE)) install.packages("sf")
if (!requireNamespace("terra", quietly = TRUE)) install.packages("terra")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

library(sf)
library(terra)
library(dplyr)

# 从命令行参数获取输入
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript this_script.R num_rows num_cols input_pts", call. = FALSE)
}

num_rows <- as.integer(args[1])
num_cols <- as.integer(args[2])
input_pts <- args[3]

# 设置默认输出目录
out_dir <- "data/out"

# 读取点文件
points_layer <- st_read(input_pts)

# 检查点的数量是否为4
if (length(points_layer$geometry) != 4) {
  stop("The input point shapefile must contain exactly 4 points.", call. = FALSE)
}

# 创建网格
grid <- st_make_grid(points_layer, n = c(num_cols, num_rows))

# 转换坐标系
point_shp <- st_cast(st_make_grid(points_layer, n = c(1, 1)), "POINT")
sourcexy <- rev(point_shp[1:4]) %>% st_transform(st_crs(points_layer))
Targetxy <- points_layer %>% st_transform(st_crs(points_layer))

# 计算控制点
controlpoints <- as.data.frame(cbind(st_coordinates(sourcexy), st_coordinates(Targetxy)))

# 线性模型
linMod <- lm(formula = cbind(controlpoints[, 3], controlpoints[, 4]) ~ controlpoints[, 1] + controlpoints[, 2], data = controlpoints)

# 计算参数
parameters <- matrix(linMod$coefficients[2:3, ], ncol = 2)
intercept <- matrix(linMod$coefficients[1, ], ncol = 2)

# 应用变换
geometry <- grid * parameters + intercept

# 创建网格shapefile
grid_shapefile <- st_sf(geometry, crs = st_crs(points_layer)) %>% mutate(ID = seq(1, length(geometry)))

# 打印结果
print(grid_shapefile)

# 写入磁盘
fn <- file.path(out_dir, "plots_grid.shp")
st_write(grid_shapefile, fn, delete_layer = TRUE) # 覆盖写入

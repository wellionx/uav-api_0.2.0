#!/bin/bash
# 合并所有分割部分
cat yolov9_maize_tassel_best_model.pt.part* > yolov9_maize_tassel_best_model.pt

# 验证文件大小
ls -lh yolov9_maize_tassel_best_model.pt

# 计算MD5校验值(可选)
md5sum yolov9_maize_tassel_best_model.pt

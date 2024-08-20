#!/bin/bash

# 定义默认训练集的路径和图片列表文件
DEFAULT_TRAIN_PATH="./Dataset/TrainSet"
DEFAULT_IMG_LIST="./Dataset/train_list.txt"

# 从命令行参数获取训练集路径和图片列表文件路径
TRAIN_PATH=${1:-$DEFAULT_TRAIN_PATH}
IMG_LIST=${2:-$DEFAULT_IMG_LIST}

# 定义Python脚本的路径
SCRIPT_PATH="./main.py"

# 定义Python解释器的路径（可选）
# PYTHON_INTERP="/usr/bin/python3"

# 运行Python脚本，并传递训练集路径和图片列表文件路径作为参数
# 如果定义了PYTHON_INTERP，则使用它来运行脚本
# 如果没有定义，则默认使用系统的python命令
if [ -n "$PYTHON_INTERP" ]; then
    $PYTHON_INTERP $SCRIPT_PATH --train_path $TRAIN_PATH --img_list $IMG_LIST
else
    python $SCRIPT_PATH --train_path $TRAIN_PATH --img_list $IMG_LIST
fi

# 检查脚本是否成功执行
if [ $? -eq 0 ]; then
    echo "脚本执行成功。"
else
    echo "脚本执行失败。" >&2
    exit 1
fi

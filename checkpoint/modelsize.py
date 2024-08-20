import jittor as jt
from jittor import nn
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import joblib
import numpy as np
import argparse

jt.flags.use_cuda = 1

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='B')

args = parser.parse_args()

# 加载模型和预处理器
model, preprocess = clip.load("./ViT-B-32.pkl")

# 确保模型被正确加载并初始化
model.eval()

# 计算 ViT 模型的参数量
total_params_vit = sum(p.numel() for p in model.parameters())
print(f"ViT模型参数量: {total_params_vit}")

# 加载保存的分类器模型
model_load_path = './best_model.pkl'
classifier = joblib.load(model_load_path)
print(f"模型已从 {model_load_path} 加载")

# ############################################  #
weights = classifier.coef_
intercepts = classifier.intercept_

# 计算分类器模型的参数总量
total_params_classifier = np.sum([w.size for w in weights]) + np.sum([i.size for i in intercepts])

# 输出总的参数量（ViT + 分类器）
total_params = total_params_vit + total_params_classifier
print(f"所有模型总的参数量（ViT + Bestmodel）: {total_params}")

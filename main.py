from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
import numpy as np
import joblib
import yaml  # 导入 yaml 模块以加载配置文件

from jittor import transform
import jittor as jt

from model.datasets import build_dataset
from model.utils_dataset import build_data_loader
from model.model import *
from .model.utils import *

jt.flags.use_cuda = 1

def main():
    # 加载配置文件
    cfg = load_config('./configs/jittor.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='B')
    args = parser.parse_args()

    clip_model, preprocess = clip.load(cfg['backbone'])

    # 加载并处理类名
    new_classes = preprocess_classes(cfg['preprocess_classes'])
    text = clip.tokenize(new_classes)
    text_features = clip_model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # 相当于clip_weights
    clip_weights = text_features

    # 加载训练集数据和特征（train_features, train_labels）
    imgs_dir = cfg['root_path']
    train_labels = open(cfg['train_text']).read().splitlines()
    train_imgs = [l.split(' ')[0] for l in train_labels]
    train_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_labels]
    cnt = {}
    new_train_imgs = []
    new_train_labels = []
    for i in range(len(train_imgs)):
        label = int(train_labels[i].numpy())
        if label not in cnt:
            cnt[label] = 0
        if cnt[label] < 4:
            new_train_imgs.append(train_imgs[i])
            new_train_labels.append(train_labels[i])
            cnt[label] += 1
    train_features = load_features(imgs_dir, new_train_imgs, preprocess, clip_model)
    train_labels = jt.cat(new_train_labels).numpy()

    # 加载测试集数据特征（test_features）
    split = 'TestSet' + args.split
    imgs_dir = 'Dataset/' + split
    test_imgs = os.listdir(imgs_dir)
    test_features = load_features(imgs_dir, test_imgs, preprocess, clip_model)

    # 处理缓存键和值（cache_keys, cache_values）的特征
    dataset = build_dataset(cfg['dataset'], cfg['dataset_path'], cfg['shots'])
    train_tranform = transform.Compose([
        transform.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transform.InterpolationMode.BICUBIC),
        transform.RandomHorizontalFlip(p=0.5),
        transform.ToTensor(),
        transform.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = extract_few_shot_feature(cfg, clip_model, train_loader_cache)
    
    #加载train_loader_F
    # dataset = build_dataset(cfg['dataset'], cfg['dataset_path'], cfg['shots'])
    # train_tranform = transform.Compose([
    #     transform.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transform.InterpolationMode.BICUBIC),
    #     transform.RandomHorizontalFlip(p=0.5),
    #     transform.ToTensor(),
    #     transform.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
    # train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True) 
    
    # 调用Jittor_FewZero_Network函数进行优化和预测
    # Jittor_FewZero_Network_T(cfg, cache_keys, cache_values, train_features, train_labels, test_features, clip_weights, clip_model, train_loader_F)
    Jittor_FewZero_Network(cfg, cache_keys, cache_values, train_features, train_labels, test_features, clip_weights)
    print("程序运行完毕，结果已保存至result.txt")

if __name__ == "__main__":
    main()

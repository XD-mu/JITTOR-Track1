import os
from tqdm import tqdm
import joblib
import yaml  # 导入 yaml 模块以加载配置文件
from PIL import Image

import jittor as jt
import jittor.nn as nn
from jittor import Function as F

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def preprocess_classes(classes_file):
    classes = open(classes_file).read().splitlines()
    new_classes = []
    for c in classes:
        c = c.split(' ')[0]
        if c.startswith('Animal'):
            c = c[7:]
        if c.startswith('Thu-dog'):
            c = c[8:]
        if c.startswith('Caltech-101'):
            c = c[12:]
        if c.startswith('Food-101'):
            c = c[9:]
        if c.startswith('Stanford-Cars'):
            c = c[14:]
        c = 'a high-resolution photo of a ' + c + ', showcasing its unique features.'
        new_classes.append(c)
    return new_classes

def load_features(imgs_dir, imgs_list, preprocess, clip_model):
    features = []
    with jt.no_grad():
        for img in tqdm(imgs_list):
            img_path = os.path.join(imgs_dir, img)
            image = Image.open(img_path)
            image = preprocess(image).unsqueeze(0)
            image_features = clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
    return jt.cat(features).numpy()

def extract_few_shot_feature(cfg, clip_model, train_loader_cache):
    cache_keys = []
    cache_values = []
    with jt.no_grad():
        for augment_idx in range(cfg['augment_epoch']):
            train_features = []
            print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.cuda()
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(jt.concat(train_features, dim=0).unsqueeze(0))
        
    cache_keys = jt.concat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = nn.functional.one_hot(jt.concat(cache_values, dim=0)).half()
    jt.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    jt.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    return cache_keys, cache_values

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=True, training_free=True):
    
    feat_dim, cate_num = clip_weights.shape
    text_feat = clip_weights.t().unsqueeze(1)
    cache_feat = cache_keys.reshape(cate_num, cfg['shots'], feat_dim)
    
    save_path = 'caches/{}'.format(cfg['dataset'])
    save_file = '{}/criterion_{}_{}shot.pt'.format(save_path, cfg['backbone'].replace('/', ''), cfg['shots'])
    
    if os.path.exists(save_file):
        print('Loading criterion...')
        sim = jt.load(save_file)
    elif only_use_txt:
        print('Calculating criterion...')
        
        feats = text_feat.squeeze()
        
        sim_sum = jt.zeros((feat_dim)).cuda()
        count = 0
        for i in range(cate_num):
            for j in range(cate_num):
                if i != j:
                    sim_sum += feats[i, :] * feats[j, :]
                    count += 1
        sim = sim_sum / count
        jt.save(sim, save_file)
    else:
        print('Calculating criterion...')
        
        feats = jt.cat([text_feat, cache_feat], dim=1)
        samp_num = feats.shape[1]
        
        sim_sum = jt.zeros((feat_dim)).cuda()
        count = 0
        for i in range(cate_num):
            for j in range(cate_num):
                for m in range(samp_num):
                    for n in range(samp_num):
                        if i != j:
                            sim_sum += feats[i, m, :] * feats[j, n, :]
                            count += 1
        sim = sim_sum / count
        jt.save(sim, save_file)

    criterion = (-1) * cfg['w'][0] * sim + cfg['w'][1] * jt.var(clip_weights, dim=1)
    
    if training_free:
        _, indices = jt.topk(criterion, k=cfg['training_free_feat_num'])
    else: 
        _, indices = jt.topk(criterion, k=cfg['training_feat_num'])
    return indices

def load_text_feature(cfg):
    save_path = cfg['cache_dir'] + "/text_weights_cupl_t.pt"
    clip_weights = jt.load(save_path)
    return clip_weights

def load_few_shot_feature(cfg):
    cache_keys = jt.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    cache_values = jt.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    return cache_keys, cache_values

def loda_val_test_feature(cfg, split):
    features = jt.load(cfg['cache_dir'] + "/" + split + "_f.pt")
    labels = jt.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    return features, labels

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def accuracy(shot_logits, cache_values, topk=(1,)):
    target = cache_values.topk(max(topk), 1, True, True)[1].squeeze()
    pred = shot_logits.topk(max(topk), 1, True, True)[1].squeeze()
    idx = (target != pred)
    return idx

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * nn.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()
    
class Jittor_FewZero_Network_Training(nn.Module):
    def __init__(self, cfg, clip_weights, clip_model, cache_keys):
        super(Jittor_FewZero_Network_Training, self).__init__()
        self.shots = cfg['shots']
        self.feat_dim, self.cate_num = clip_weights.shape
        
        self.value_weights = nn.Parameter(jt.ones([self.cate_num*cfg['shots'], 1]).half().cuda(), requires_grad=True)
        self.indices = cal_criterion(cfg, clip_weights, cache_keys)

        self.res = nn.Parameter(jt.zeros([self.cate_num, cfg['training_feat_num']]).half().cuda(), requires_grad=True)
        self.feat_num = cfg['training_feat_num']
        
    def forward(self, cache_keys, clip_weights, cache_values):
        
        res_keys = self.res.unsqueeze(1).repeat(1, self.shots, 1).reshape(-1, self.feat_num)
        new_cache_keys = cache_keys.clone()
        new_cache_keys = new_cache_keys.reshape(-1, self.feat_dim)
        new_cache_keys[:, self.indices] = new_cache_keys[:, self.indices] + res_keys
    
        res_text = self.res.t()
        new_clip_weights = clip_weights.clone()
        new_clip_weights[self.indices, :] = clip_weights[self.indices, :] + res_text 
        new_cache_values = cache_values * self.value_weights
       
        return new_cache_keys.half(), new_clip_weights.half(), new_cache_values.half()
    
    
    

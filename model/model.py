from tqdm import tqdm
import numpy as np

import jittor as jt
from jittor import Function as F
import jittor.nn as nn

from utils import *


def Jittor_FewZero_Network(cfg, cache_keys, cache_values, val_features, val_labels, test_features, clip_weights):
    
    print(test_features.shape)
    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num).cuda()
    cache_keys = cache_keys.t().reshape(cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim).cuda()
    
    cache_keys, cache_values = cache_keys.reshape(-1, feat_dim), cache_values.reshape(-1, cate_num)
    
    cfg['w'] = cfg['w_training_free']
    indices = cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=False)
    
    new_clip_weights = clip_weights[indices, :]
    new_cache_keys = cache_keys[:, indices]
    new_test_features = test_features[:, indices]
    new_val_features = val_features[:, indices]
    
    new_clip_weights = new_clip_weights / new_clip_weights.norm(dim=0, keepdim=True)
    new_cache_keys = new_cache_keys /  new_cache_keys.norm(dim=-1, keepdim=True)
    new_test_features = new_test_features /  new_test_features.norm(dim=-1, keepdim=True)
    new_val_features = new_val_features /  new_val_features.norm(dim=-1, keepdim=True)
    
    # Zero-shot CLIP
    R_fW = 100. * test_features @ clip_weights
    
    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']
    
    # Calculate R_f'F'
    R_fF = new_test_features @ new_cache_keys.t()
    
    # Calculate R_F'W'
    key_logits = new_cache_keys @ new_clip_weights
    key_logits = key_logits.softmax(1)
    cache_div = jt.sum(cache_values * jt.log2((cache_values + 1e-6) / (key_logits + 1e-6)), dim=1)[:, None]
    R_FW = (cache_div * gamma).exp()
    soft_cache_values = cache_values * R_FW
    
    cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ soft_cache_values
    
    ape_logits = R_fW + cache_logits * alpha
    
    best_search_acc = 0
    R_fF = new_val_features @ new_cache_keys.t()
    R_fW = 100. * val_features @ clip_weights
    best_beta, best_alpha, best_gamma = 0, 0, 0
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
    gamma_list = [i * cfg['search_scale'][2] / cfg['search_step'][2] for i in range(cfg['search_step'][2])]
    for beta in beta_list:
        for alpha in alpha_list:
            for gamma in gamma_list:
                with jt.no_grad():
                    soft_cache_values = cache_values * (cache_div * gamma).exp()                    
                    cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ soft_cache_values
                    ape_logits = R_fW + cache_logits * alpha
                acc = cls_acc(ape_logits, val_labels)
                if acc > best_search_acc:
                    print("新的最佳设置, alpha: {:.2f}, beta: {:.2f}, gamma: {:.2f}; accuracy: {:.2f}".format(alpha, beta, gamma, acc))
                    best_search_acc = acc
                    best_alpha, best_beta, best_gamma = alpha, beta, gamma
    print("\n查找后，最佳val准确率: {:.2f}.\n".format(best_search_acc))
    
    R_fW = 100. * test_features @ clip_weights
    R_fF = new_test_features @ new_cache_keys.t()
    
    soft_cache_values = cache_values * (cache_div * best_gamma).exp()
    cache_logits = ((-1) * (best_beta - best_beta * R_fF)).exp() @ soft_cache_values
    
    ape_logits = R_fW + cache_logits * best_alpha
    # output the result.txt
    with open('result.txt', 'w') as save_file:
        for i, prediction in enumerate(ape_logits.tolist()):
            prediction = np.asarray(prediction)
            top5_idx = prediction.argsort()[-1:-6:-1]
            save_file.write(test_features[i] + ' ' +
                            ' '.join(str(idx) for idx in top5_idx) + '\n')   
    
def Jittor_FewZero_Network_T(cfg, cache_keys, cache_values, val_features, val_labels, test_features, clip_weights, clip_model, train_loader_F):
    
    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num)
    cache_keys = cache_keys.t().reshape(cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim)
    
    cfg['w'] = cfg['w_training']
    cache_keys, cache_values = cache_keys.reshape(-1, feat_dim), cache_values.reshape(-1, cate_num)
    adapter = Jittor_FewZero_Network_Training(cfg, clip_weights, clip_model, cache_keys).cuda()
    
    optimizer = jt.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=cfg['eps'], weight_decay=1e-1)
    scheduler = jt.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    Loss = SmoothCrossEntropy()
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0
    
    for train_idx in range(cfg['train_epoch']):
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with jt.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
            R_fF = image_features @ new_cache_keys.half().t()
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
            R_fW = 100. * image_features @ new_clip_weights
            ape_logits = R_fW + cache_logits * alpha

            loss = Loss(ape_logits, target)

            acc = cls_acc(ape_logits, target)
            correct_samples += acc / 100 * len(ape_logits)
            all_samples += len(ape_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        adapter.eval()
        with jt.no_grad():
            new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)

            R_fF = val_features @ new_cache_keys.half().t()
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
            R_fW = 100. * val_features @ new_clip_weights
            ape_logits = R_fW + cache_logits * alpha
        acc = cls_acc(ape_logits, val_labels)

        print("**** Jittor_FewZero_Network-T's 测试准确率: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            jt.save(adapter, cfg['cache_dir'] + "/Jittor_FewZero_Network-T_" + str(cfg['shots']) + "shots.pt")
    
    adapter = jt.load(cfg['cache_dir'] + "/Jittor_FewZero_Network-T_" + str(cfg['shots']) + "shots.pt")
    print(f"**** 微调后，Jittor_FewZero_Network-T's最佳测试准确率: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")
    
    print("\n-------- 查找训练集上超参数 --------")
    best_search_acc = 0
    best_beta, best_alpha = 0, 0
    beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
    for beta in beta_list:
        for alpha in alpha_list:
            with jt.no_grad():
                new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
                
                R_fF = val_features @ new_cache_keys.half().t()
                cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
                R_fW = 100. * val_features @ new_clip_weights
                ape_logits = R_fW + cache_logits * alpha
            acc = cls_acc(ape_logits, val_labels)
            if acc > best_search_acc:
                print("新的最佳设置, alpha: {:.2f}, beta: {:.2f}; accuracy: {:.2f}".format(alpha, beta, acc))
                best_search_acc = acc
                best_alpha, best_beta = alpha, beta
    print("\n查找后，最佳val准确率: {:.2f}.\n".format(best_search_acc))
    
    print("\n-------- Evaluating on the test set. --------")
    with jt.no_grad():
        new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
        
        R_fF = test_features @ new_cache_keys.half().t()
        cache_logits = ((-1) * (best_beta - best_beta * R_fF)).exp() @ R_FW
        R_fW = 100. * test_features @ new_clip_weights
        ape_logits = R_fW + cache_logits * best_alpha
    # output the result.txt
    with open('result.txt', 'w') as save_file:
        for i, prediction in enumerate(ape_logits.tolist()):
            prediction = np.asarray(prediction)
            top5_idx = prediction.argsort()[-1:-6:-1]
            save_file.write(test_features[i] + ' ' +
                            ' '.join(str(idx) for idx in top5_idx) + '\n')

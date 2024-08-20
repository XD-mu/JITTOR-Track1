import torch
import jittor as jt
clip = torch.load('RN101.pt').state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, 'RN101.pkl')
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
model, preprocess = clip.load("./checkpoint/ViT-B-32.pkl")

# 加载保存的分类器模型
model_load_path = './checkpoint/best_model.pkl'
classifier = joblib.load(model_load_path)
print(f"模型已从 {model_load_path} 加载")

weights = classifier.coef_
intercepts = classifier.intercept_

# 计算参数总量
total_params = np.sum([w.size for w in weights]) + np.sum([i.size for i in intercepts])

print(f"训练最终模型的参数量: {total_params}")

# 测试集数据加载
split = 'TestSet' + args.split
imgs_dir = 'Dataset/' + split
test_imgs = os.listdir(imgs_dir)

# 处理测试数据
print('Testing data processing:')
test_features = []
with jt.no_grad():
    for img in tqdm(test_imgs):
        img_path = os.path.join(imgs_dir, img)
        image = Image.open(img_path)
        image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)       
        test_features.append(image_features)

test_features = jt.cat(test_features).numpy()

# 使用保存的模型进行预测
with open('result.txt', 'w') as save_file:
    predictions = classifier.predict_proba(test_features)
    for i, prediction in enumerate(predictions):
        prediction = np.asarray(prediction)
        top5_idx = prediction.argsort()[-1:-6:-1]
        save_file.write(test_imgs[i] + ' ' +
                        ' '.join(str(idx) for idx in top5_idx) + '\n')
print('测试完成，结果已保存到 result.txt')

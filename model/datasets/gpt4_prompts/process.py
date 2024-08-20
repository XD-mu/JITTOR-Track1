import json
from collections import defaultdict

# 定义文件路径
input_file = 'classes_B_caption_GPT4.txt'
output_files = {
    "Food-101": "Food-101.json",
    "Caltech-101": "Caltech-101.json",
    "Animal": "Animal.json",
    "Thu-dog": "Thu-dog.json",
    "Stanford-Cars": "Stanford-Cars.json"
}

# 初始化数据结构
data = defaultdict(lambda: defaultdict(list))

# 读取并处理数据
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # 获取数据集名、类别、描述和标签
        parts = line.rsplit(' ', 1)
        description = parts[0].strip()
        label = parts[1].strip()
        
        dataset_info = description.split('_', 1)
        dataset = dataset_info[0]
        
        if dataset in output_files:
            # 使用label作为类别名称
            data[dataset][label].append(description)

# 将处理后的数据保存为JSON文件
for dataset, classes in data.items():
    output_data = {}
    for label, descriptions in classes.items():
        output_data[label] = descriptions
    
    with open(output_files[dataset], 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

print("数据处理完成，JSON文件已保存。")

import os
import random
from shutil import copyfile

def split_dataset(input_images_dir, input_labels_dir, output_dir, split_ratio=(0.7, 0.05, 0.25)):
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_images_dir) if f.endswith('.png')]
    num_images = len(image_files)

    # 随机打乱图片顺序
    random.shuffle(image_files)

    # 计算划分的数量
    num_train = int(num_images * split_ratio[0])
    num_val = int(num_images * split_ratio[1])
    num_test = num_images - num_train - num_val

    # 分割图片和标签文件
    for i, image_file in enumerate(image_files):
        if i < num_train:
            set_name = 'train'
        elif i < num_train + num_val:
            set_name = 'val'
        else:
            set_name = 'test'

        # 复制图片文件
        copyfile(os.path.join(input_images_dir, image_file), os.path.join(output_dir, 'images', set_name, image_file))

        # 构建对应的标签文件名
        label_file = os.path.splitext(image_file)[0] + '.txt'

        # 复制标签文件
        copyfile(os.path.join(input_labels_dir, label_file), os.path.join(output_dir, 'labels', set_name, label_file))

# 调用划分函数
split_dataset('./datasets/DOTAs/images', './datasets/DOTAs/labels', './datasets/DOTAs')


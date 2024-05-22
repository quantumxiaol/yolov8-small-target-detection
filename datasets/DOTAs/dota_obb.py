# 将标签 ./ground trurth  (563,478),(630,573),1 转为 1 x x x x 位于./labels
# datasets

#         |-images

#                 |--train

#                 |--val

#                 |--test

#         |-labels

#                 |--train

#                 |--val

#                 |--test

import os
import cv2

# 类别映射字典
category_mapping = {
    'plane': 0,
    'baseball-diamond': 1,
    'bridge': 2,
    'ground-track-field': 3,
    'small-vehicle': 4,
    'large-vehicle': 5,
    'ship': 6,
    'tennis-court': 7,
    'basketball-court': 8,
    'storage-tank': 9,
    'soccer-ball-field': 10,
    'roundabout': 11,
    'harbor': 12,
    'swimming-pool': 13,
    'helicopter': 14,
    'container-crane': 15
}

def convert_label_to_yolo_obb(label_file, image_width, image_height):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    yolo_labels = []
    for line in lines[2:]:  # 忽略前两行（imagesource和gsd）
        line = line.strip()
        if not line:  # 跳过空行
            print("Skipping empty line.")
            continue
        
        # 解析标签数据
        parts = line.split(' ')
        coords = [float(part) for part in parts[:8]]  # 提取坐标信息
        category = parts[8]
        difficult = int(parts[9])

        # 归一化坐标
        normalized_coords = [coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)]

        # 编码类别
        class_index = category_mapping.get(category, -1)

        # 如果类别未知或难以检测，跳过
        if class_index == -1 or difficult == 1:
            continue

        # 生成YOLOv8 OBB格式标签
        yolo_label = f"{class_index} " + " ".join(map(str, normalized_coords))
        yolo_labels.append(yolo_label)

    return yolo_labels

# 图像和标签文件夹路径
image_folder = "./datasets/DOTAs/origin_images"
label_folder = "./datasets/DOTAs/origin_labels"
output_label_folder = "./datasets/DOTAs/labels"
output_image_folder = "./datasets/DOTAs/images"


# 创建输出文件夹
os.makedirs(output_label_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

# 遍历标签文件夹
for filename in os.listdir(label_folder):
    if filename.endswith('.txt'):
        label_file = os.path.join(label_folder, filename)
        image_file = os.path.join(image_folder, os.path.splitext(filename)[0] + '.png')

        # 读取图像尺寸
        image = cv2.imread(image_file)
        if image is None:
            print(f"Image {image_file} not found or cannot be read.")
            continue

        image_height, image_width, _ = image.shape

        # 转换标签为YOLOv8 OBB格式
        yolo_labels = convert_label_to_yolo_obb(label_file, image_width, image_height)

        # 保存转换后的标签
        output_label_file = os.path.join(output_label_folder, os.path.splitext(filename)[0] + '.txt')
        with open(output_label_file, 'w') as f:
            f.write('\n'.join(yolo_labels))

        # 将图像复制到输出文件夹
        output_image_file = os.path.join(output_image_folder, os.path.basename(image_file))
        cv2.imwrite(output_image_file, image)

print("转换完成！")

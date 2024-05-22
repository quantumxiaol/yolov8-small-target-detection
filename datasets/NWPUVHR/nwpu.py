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
def convert_label_to_yolo(label_file, image_width, image_height):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    yolo_labels = []
    for line in lines:
        # 解析原始标签数据
        line = line.strip()  # 去除行两端的空格
        if not line:  # 检查是否为空行
            print("Skipping empty line.")
            continue
            
        print("Original Line:", line)  # 打印原始行内容
        
        # 查找类别信息的位置
        class_index = line.rfind(',') + 1
        coords_str = line[:class_index - 1]  # 坐标部分
        class_id = line[class_index:]  # 类别部分
        print("Coords Str:", coords_str)  # 打印分割后的坐标部分
        print("Class ID:", class_id)  # 打印分割后的类别部分
        
        # 去除坐标部分内部的额外空格
        coords_str = coords_str.replace(" ", "")
        
        coords = coords_str.strip('()').split('),(')
        x_min, y_min = map(int, coords[0].split(','))
        x_max, y_max = map(int, coords[1].split(','))

        # 计算物体的中心坐标和宽度高度
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        box_width = x_max - x_min
        box_height = y_max - y_min

        # 归一化坐标
        x_center /= image_width
        y_center /= image_height
        box_width /= image_width
        box_height /= image_height

        # 编码类别
        class_index = int(class_id)-1

        # 生成YOLOv8格式标签
        yolo_label = f"{class_index} {x_center} {y_center} {box_width} {box_height}"
        yolo_labels.append(yolo_label)

    return yolo_labels



# 图像和标签文件夹路径
image_folder = "./datasets/NWPUVHR/positive image set"
label_folder = "./datasets/NWPUVHR/ground truth"
output_label_folder = "./datasets/NWPUVHR/labels"
output_image_folder = "./datasets/NWPUVHR/images"

# 创建输出文件夹
os.makedirs(output_label_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

# 遍历标签文件夹
for filename in os.listdir(label_folder):
    if filename.endswith('.txt'):
        label_file = os.path.join(label_folder, filename)
        image_file = os.path.join(image_folder, os.path.splitext(filename)[0] + '.jpg')

        # 读取图像尺寸
        image = cv2.imread(image_file)
        image_height, image_width, _ = image.shape

        # 转换标签为YOLOv8格式
        yolo_labels = convert_label_to_yolo(label_file, image_width, image_height)

        # 保存转换后的标签
        output_label_file = os.path.join(output_label_folder, os.path.splitext(filename)[0] + '.txt')
        with open(output_label_file, 'w') as f:
            f.write('\n'.join(yolo_labels))

        # 将图像复制到输出文件夹
        output_image_file = os.path.join(output_image_folder, os.path.basename(image_file))
        cv2.imwrite(output_image_file, image)

print("转换完成！")
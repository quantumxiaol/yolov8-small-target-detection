import os
def create_image_list_txt(image_dir, output_txt):
    with open(output_txt, 'w') as f_txt:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_path = os.path.join(root, file)
                    f_txt.write(image_path + '\n')

# 汇总训练集图像信息到train.txt
create_image_list_txt('./datasets/NWPUVHR/images/train', './datasets/NWPUVHR/train.txt')

# 汇总验证集图像信息到val.txt
create_image_list_txt('./datasets/NWPUVHR/images/val', './datasets/NWPUVHR/val.txt')

# 汇总测试集图像信息到test.txt
create_image_list_txt('./datasets/NWPUVHR/images/test', './datasets/NWPUVHR/test.txt')
# DOTA 1.5

官网：
https://captain-whu.github.io/DOTA/index.html

数据集地址：
https://www.kaggle.com/datasets/chandlertimm/dota-data

可在阿里云下载
https://tianchi.aliyun.com/dataset/146406?t=1716210821741


我只使用train part 1.zip共469张图像作为数据集，在此中划分训练集、测试集、验证集

到906.png

## 文件结构

       DOTAs/
       |
       |--- origin_labels/            # 储存原始标签文件
       |   |--- P0000.txt             # 示例标签文件
       |       imagesource:GoogleEarth
       |       gsd:0.146343590398
       |       2244.0 1791.0 2254.0 1795.0 2245.0 1813.0 2238.0 1809.0 small-vehicle 1
       |
       |--- origin_images/            # 储存原始图片文件
       |   |--- P0000.png             # 示例图片（22.5 MB, 3875 x 5502）
       |
       |--- images/                   # 划分后的图片（train/val/test）
       |   |--- train/
       |   |--- val/
       |   |--- test/
       |
       |--- labels/                   # 转换格式后的标签（train/val/test）
       |   |--- train/
       |   |--- val/
       |   |--- test/
       |
       |--- train.txt                 # 存放训练集图片路径
       |--- test.txt                  # 存放测试集图片路径
       |--- val.txt                   # 存放验证集图片路径
       |
       |--- dota.py                   # 标签格式转换脚本（运行前需修改路径）
       |--- testsplit.py              # 划分训练集/测试集/验证集脚本（运行前需修改路径）
       |--- ll.py                     # 生成 train.txt、test.txt、val.txt 的脚本（运行前需修改路径）

DOTA 1.5

官网：
https://captain-whu.github.io/DOTA/index.html

数据集地址：
https://www.kaggle.com/datasets/chandlertimm/dota-data

可在阿里云下载
https://tianchi.aliyun.com/dataset/146406?t=1716210821741


我只使用train part 1.zip共469张图像作为数据集，在此中划分训练集、测试集、验证集

到906.png

|-DOTAs

    |-origin_labels 储存数据集集的标签

           |--P0000.txt imagesource:GoogleEarth
                      gsd:0.146343590398
                      2244.0 1791.0 2254.0 1795.0 2245.0 1813.0 2238.0 1809.0 small-vehicle 1

    |-origin_images 储存图片

           |--P0000.png  22.5 MB  3875 x 5502

    |-images 存放划分训练集、测试集、验证集

           |--train

           |--val

           |--test

    |-labels 转换标签格式，存放划分训练集、测试集、验证集

           |--train

           |--val

           |--test
    
    |-train.txt     存放训练集图片的路径

    |-test.txt      存放测试集图片的路径

    |-val.txt       存放验证集图片的路径

    |-dota.py       脚本    运行前需要修改路径  用于转换标签格式

    |-testsplit.py  脚本    运行前需要修改路径  用于划分训练集、测试集、验证集，将图片和标签存放在images和labels文件夹下

    |-ll.py         脚本    运行前需要修改路径  用于统计训练集、验证集、测试集的路径，生成train.txt、test.txt、val.txt

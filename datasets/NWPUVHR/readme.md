这是将NWPU VHR-10数据集用于yolov5和yolov8的脚本

yolov8需要的标签为类别 中心坐标 宽高（归一化）0 0.622651356993737 0.6503712871287128 0.06993736951983298 0.11757425742574257

NWPU VHR-10的标签为左上角坐标 右下角坐标 类别 (563,478),(630,573),1 

文件如下


    |-NWPU VHR-10 不建议有空格和分隔符
    
        |-ground truth 储存数据集集的标签

               |--001.txt (563,478),(630,573),1

        |-positive image set 储存图片

               |--001.jpg

        |-negative image set 存放可能没有检测物品的图片

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

        |-nwpu.py       脚本    运行前需要修改路径  用于转换标签格式

        |-testsplit.py  脚本    运行前需要修改路径  用于划分训练集、测试集、验证集，将图片和标签存放在images和labels文件夹下

        |-ll.py         脚本    运行前需要修改路径  用于统计训练集、验证集、测试集的路径，生成train.txt、test.txt、val.txt
        

# yolov8-small-target-detection

基于yolov8实现小目标检测，在NWPU VHR-10和DOTA上测试

使用Gradio-YOLOv8-Det进行可视化

yolov8  https://github.com/ultralytics/ultralytics

Gradio-YOLOv8-Det  https://gitee.com/CV_Lab/gradio-yolov8-det

可视化需要执行gradio_yolov8_det下的gradio_yolov8_det_v2.py。

根据要求修改/model_config/model_name_all.yaml以添加自己的模型权值

修改/cls_name/cls_name_zh.yaml以修改目标检测的标签值

文件结构
|- yolov8 解压yolov8源代码

    |-datasets 储存数据集

           |--DOTAs

           |--NWPUVHR

    |-gradio_yolov8_det 结果可视化

    |-yolov8fornwpuvhr.pt  在NWPU VHR-10上训练的一个权值

    |-yolov8.yaml  配置网络，修改nc

    |-train.py  在NWPU VHR-10训练模型
    
    |-train_DOTAs.py  在DOTA训练模型
    
    |-pred.py  评估模型





Gradio-YOLOv8-Det的原作者  曾逸夫, (2024) Gradio YOLOv8 Det (Version 2.1.0).https://gitee.com/CV_Lab/gradio-yolov8-det.git.

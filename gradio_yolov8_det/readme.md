可以在https://gitee.com/CV_Lab/gradio-yolov8-det下载源代码运行。

也可以通过pip install gradio-yolov8-det运行。

如果是下载源码方式使用2.0版本，如果无法运行，可能需要在gradio_yolov8_det_v2.py下添加

if __name__ == "__main__":
    entrypoint()

项目结构
├──  gradio-yolov8-det											# 项目名称

│    ├── gradio_yolov8_det										# 项目核心文件

│    │   ├── model_config										# 模型配置

│    │   │   ├── model_name_all.yaml							# YOLOv8 模型名称（yaml版）

│    │   │   └── model_name_custom.yaml							# 自定义模型名称（yaml版）

│    │   ├── cls_name											# 类别名称

│    │   │   ├── cls_name_zh.yaml								# 类别名称文件（yaml版-中文）

│    │   │   ├── cls_imagenet_name_zh.yaml						# ImageNet类别名称文件（yaml版-中文）

│    │   │   ├── cls_name_en.yaml								# 类别名称文件（yaml版-英文）

│    │   │   ├── cls_name_ru.yaml								# 类别名称文件（yaml版-俄语）

│    │   │   ├── cls_name_es.yaml								# 类别名称文件（yaml版-西班牙语）

│    │   │   ├── cls_name_ar.yaml								# 类别名称文件（yaml版-阿拉伯语）

│    │   │   ├── cls_name_ko.yaml								# 类别名称文件（yaml版-韩语）

│    │   │   ├── cls_name.yaml									# 类别名称文件（yaml版-中文-v0.1）

│    │   │   └── cls_name.csv									# 类别名称文件（csv版-中文）

│    │   ├── gyd_utils											# 工具包

│    │   │   ├── __init__.py									# 工具包初始化文件

│    │   │   └── fonts_opt.py									# 字体管理

│    │   ├── img_examples										# 示例图片

│    │   ├── __init__.py										# 初始化文件

│    │   ├── gradio_yolov8_det_v2.py							# v2.0.0主运行文件

│    │   └── gyd_style.css										# CSS样式文件

│    ├── setup.cfg												# pre-commit CI检查源配置文件

│    ├── .pre-commit-config.yaml								# pre-commit配置文件

│    ├── LICENSE												# 项目许可

│    ├── .gitignore												# git忽略文件

│    ├── README.md												# 项目说明

│    ├── pyproject.toml											# Python Package构建文件

│    ├── Dockerfile												# Docker构建工具

│    └── .dockerignore											# Docker忽略文件

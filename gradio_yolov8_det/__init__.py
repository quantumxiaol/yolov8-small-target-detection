# Gradio YOLOv8 Det, GPL-3.0 license
# https://gitee.com/CV_Lab/gradio-yolov8-det


from .gyd_utils.fonts_opt import is_fonts

__version__ = "v2.1.0"

print(
    f"欢迎使用 Gradio YOLOv8 Det {__version__} \n具体内容参见源码地址：https://gitee.com/CV_Lab/gradio-yolov8-det"
)

__all__ = "__version__", "is_fonts"

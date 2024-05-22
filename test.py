import os
import cv2
from ultralytics import YOLO
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
import multiprocessing

class CustomDataset(IterableDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.num_samples = len(self.image_files)

    def __iter__(self):
        return self.data_generator()

    def data_generator(self):
        for img_path in self.image_files:
            image = cv2.imread(img_path)
            yield cv2.cvtColor(image, cv2.COLOR_BGR2RGB), os.path.basename(img_path)

def train_yolov8(dataset_dir, yaml_file):
    # 初始化模型
    # model = YOLO("yolov8.yaml")  # build a new model from scratch
    model = YOLO('yolov8n.pt')  # 或者你想要使用的其他模型权重
    
    # 创建自定义数据集实例
    custom_dataset = CustomDataset(os.path.join(dataset_dir, "labels/val"))

    # 将数据生成器包装成PyTorch的DataLoader
    batch_size = 16
    data_loader = DataLoader(custom_dataset, batch_size=batch_size)

    # 使用模型进行验证
    for images, filenames in data_loader:
        # 在这里使用模型对图像进行验证
        # 这里只是一个示例，实际的验证操作应根据模型的要求进行调整
        results = model(images)
        # 处理验证结果，例如计算指标等

    # 训练模型
    model.train(
        data=yaml_file,
        epochs=100,
        imgsz=640,
        batch=16,
        project='yolov8_project',
        name='yolov8_training',
        cache=True
    )

def test_yolov8(dataset_dir, weights_path, img_dir):
    # 初始化模型
    model = YOLO(weights_path)
    
    # 执行测试
    results = model.detect(
        source=img_dir,  # 图片文件夹路径或单张图片路径
        imgsz=640,
        conf_thres=0.4,  # 置信度阈值
        iou_thres=0.5,   # IoU阈值3
    )
    
    # 显示测试结果
    results.show()

# 数据集目录和配置文件路径
dataset_dir = "./datasets/NWPUVHR"
yaml_file = os.path.join(dataset_dir, "nwpu.yaml")

# 训练YOLOv8模型
# train_yolov8(dataset_dir, yaml_file)

# 测试YOLOv8模型
weights_path = os.path.join(dataset_dir, "yolov8_training/weights/best.pt")
img_dir = os.path.join(dataset_dir, "images/test")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_yolov8(dataset_dir, yaml_file)
    test_yolov8(dataset_dir, weights_path, img_dir)

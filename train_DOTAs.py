from ultralytics import YOLO
import os
import wandb
use_wandb = False  # 设置为False关闭wandb

if use_wandb:
    wandb.init(project="my_project")
wandb.disabled = True
os.environ['WANDB_DISABLED'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


 
if __name__ == '__main__':
    # 加载模型
    model = YOLO('yolov8_DOTAs.yaml').load('yolov8n.pt')  # 从YAML构建并转移权重
    # 训练模型
    results = model.train(data='./datasets/DOTAs/dota.yaml', epochs=70,  workers=0,batch=8)
    # 如果数据集中存在大量小对象，增大输入图像的尺寸imgsz可以使得这些小对象从高分辨率中受益，更好的被检测出。

    # 保存模型为yolov8fornwpuvhr.pt
    # model.save('yolov8fornwpuvhr.pt')
 
    metrics = model.val()
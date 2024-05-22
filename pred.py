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
    # model = YOLO('yolov8.yaml').load('yolov8n.pt')  # 从YAML构建并转移权重

    # 训练模型
    # results = model.train(data='./datasets/NWPUVHR/nwpu.yaml', epochs=20, workers=0, batch=64)
    
    # 保存模型为yolov8fornwpuvhr.pt
    # model.save('yolov8fornwpuvhr.pt')
    
    # 使用验证集评估模型
    # metrics = model.val()
    model=YOLO('yolov8.yaml').load('yolov8fornwpuvhr.pt')

    # 使用测试集评估模型
    test_results = model.val(data='./datasets/NWPUVHR/nwpu.yaml', split='test')

    result=model('./datasets/NWPUVHR/images/001.jpg')
    
    print(test_results)
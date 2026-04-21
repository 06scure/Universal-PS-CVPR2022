import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from modules.io import dataio
from modules.model import model
from modules.config import config
from modules.utils import image_utils, log
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='UniPS')
parser.add_argument('--session_name', default = 'test_session',
    help='训练/测试会话名称')
parser.add_argument('--test_dir', default = '/home/user/dataset/DiLiGenT_512',
    help='测试数据集路径')
parser.add_argument('--batchsize', type=int, default = 1,
    help='训练批大小，即每批的物体数量')
parser.add_argument('--outdir', default='output/test_session_0421',
    help='输出根目录，用于保存检查点、日志和测试结果')
parser.add_argument('--pretrained', default='/home/user/code/Universal-PS-CVPR2022/output/train_session/checkpoint/20260421_144733.pt',
    help='预训练检查点目录路径，用于恢复训练或推理')
parser.add_argument('--num_agg_enc', type=int, default=3,
    help='聚合 Transformer 中编码器 SAB (集合注意力块) 的层数')
parser.add_argument('--min_nimg', type=int, default=10,
    help='测试时每个物体最少采样的输入图像数; 网络会在 [min_nimg, 总图像数] 范围内随机选取')
parser.add_argument('--num_samples', type=int, default=5000,
    help='测试时每个物体最大像素采样数; 从前景掩码内随机抽取，用于限制显存占用')

def main():
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger = log.setup_logger(outdir=args.outdir)

    # 初始化超参数
    conf = config.TestConfig()  # 从 config 模块获取配置对象

    # 初始化数据集读取
    test_data = dataio.dataio(mode='Test', args=args, conf=conf, outdir=args.outdir)
    if test_data is None:
        raise RuntimeError("Failed to load test data.")
    test_data_loader = DataLoader(
        dataset = test_data,
        batch_size = args.batchsize,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    # 初始化模型
    net = model.UniPS(args=args, device=device).to(device)
    if args.pretrained is not None:
        net.load(args.pretrained)
    else:
        raise RuntimeError("Pretrained model path must be provided for testing.")
    
    # 设置模型为测试模式
    net.set_mode(mode='Test')
    net.eval()

    # 进行测试
    result = {}
    for batch in tqdm(test_data_loader, desc=f'Testing Epoch', leave=False):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                loss, mae, normal_map, error_map = net(batch)

        # 保存输出
        cv2.imwrite(f'{test_data.data.data_workspace}/normal.png', (255 * normal_map[0,:,:,:].transpose(1,2,0)[:,:,::-1]).astype(np.uint8))  # 转换为 HWC 和 BGR 格式  
        cv2.imwrite(f'{test_data.data.data_workspace}/error.png', image_utils.make_error_heatmap(error_map))

        result[test_data.data.data_workspace] = mae

    # 输出测试结果
    for obj_name, mae in result.items():
        logger.info(f"{obj_name}: MAE = {mae:.4f} degrees")
    logger.info(f"Testing completed. avg MAE: {sum(result.values()) / len(result):.4f} degrees")
    logger.info(f"Results saved to: {args.outdir}")

if __name__ == "__main__":
    main()

import os
import cv2
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from modules.io import dataio
from modules.model import model
from modules.config import config
from torch.utils.data import DataLoader
from modules.utils import image_utils, log
from typing import Dict

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='UniPS')
parser.add_argument('--session_name', default = 'test_session',
    help='训练/测试会话名称')
parser.add_argument('--test_dir', default = '/home/user/dataset/DiLiGenT_512',
    help='测试数据集路径')
parser.add_argument('--outdir', default='output/test_session',
    help='输出根目录，用于保存检查点、日志和测试结果')
parser.add_argument('--pretrained', default=None,
    help='预训练检查点目录路径，用于恢复训练或推理')
parser.add_argument('--num_agg_enc', type=int, default=3,
    help='聚合 Transformer 中编码器 SAB (集合注意力块) 的层数')
parser.add_argument('--num_samples', type=int, default=5000,
    help='测试时每个物体最大像素采样数; 从前景掩码内随机抽取，用于限制显存占用')

def test(args: argparse.Namespace, net: model.UniPS, device: torch.device) -> Dict:
    # 初始化超参数
    conf = config.TestConfig()  # 从 config 模块获取配置对象
    # 初始化数据集读取
    test_data = dataio.dataio(mode='Test', args=args, conf=conf, outdir=args.outdir)
    if test_data is None:
        raise RuntimeError("Failed to load test data.")
    test_data_loader = DataLoader(
        dataset = test_data,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    # 设置模型为测试模式
    net.eval()
    # 保存测试结果
    result = {}
    for batch in tqdm(test_data_loader, desc=f'Testing Epoch',leave = False):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                loss, mae, normal_map, error_map = net(batch)

        result[test_data.data.data_workspace] = (loss, mae, normal_map, error_map)
    return result

def main():
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger = log.setup_logger(outdir=args.outdir)
    
    # 初始化模型
    net = model.UniPS(args=args, device=device).to(device)
    if args.pretrained is not None:
        net.load(args.pretrained)
    else:
        raise RuntimeError("Pretrained model path must be provided for testing.")
    
    # 测试
    result = test(args, net, device)

    # 输出测试结果
    for obj_name, (_, mae, normal_map, error_map) in result.items():
        logger.info(f"{os.path.basename(obj_name.rstrip(os.sep))}, MAE={mae:.4f} degrees")
        os.makedirs(obj_name, exist_ok=True)
        cv2.imwrite(f'{args.outdir}/{os.path.basename(obj_name.rstrip(os.sep))}/normal.png', (255 * normal_map[0,:,:,:].transpose(1,2,0)[:,:,::-1]).astype(np.uint8))  # 转换为 HWC 和 BGR 格式  
        cv2.imwrite(f'{args.outdir}/{os.path.basename(obj_name.rstrip(os.sep))}/error.png', image_utils.make_error_heatmap(error_map))
    logger.info(f"Testing completed. avg MAE: {sum([mae for _, (loss,mae,normal_map,error_map) in result.items()]) / len(result):.4f} degrees")
    logger.info(f"Results saved to: {args.outdir}")

if __name__ == "__main__":
    main()

import os
import logging
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.io import dataio
from modules.model import model
from modules.config import config
from modules.model import model_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='UniPS Test')
parser.add_argument('--session_name', default='test_session',
    help='测试会话名称')
parser.add_argument('--test_dir', default='/home/user/dataset/DiLiGenT/pmsData',
    help='测试数据集路径')
parser.add_argument('--batchsize', type=int, default=1,
    help='批大小')
parser.add_argument('--outdir', default='output/test_session',
    help='输出根目录')
parser.add_argument('--pretrained', default=None,
    help='checkpoint 文件路径')
parser.add_argument('--num_agg_enc', type=int, default=3,
    help='聚合 Transformer 中编码器 SAB 的层数')
parser.add_argument('--num_samples', type=int, default=6144,
    help='测试时每次聚合的最大像素块大小')
parser.add_argument('--max_test_objects', type=int, default=None,
    help='限制测试物体数量；默认测试全部物体')
parser.add_argument('--encoder_imgsize', type=int, default=256,
    help='送入编码器的图像分辨率')
parser.add_argument('--decoder_imgsize', type=int, default=512,
    help='输出阶段使用的图像分辨率')


def make_error_heatmap(error_map):
    error = error_map[0, 0]
    valid_mask = error > 0

    error_uint8 = np.zeros_like(error, dtype=np.uint8)
    error_uint8[valid_mask] = np.clip(error[valid_mask] / 90.0 * 255.0, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(error_uint8, cv2.COLORMAP_JET)
    heatmap[~valid_mask] = 0
    return heatmap


def main():
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.outdir, exist_ok=True)

    conf = config.TestConfig()
    test_data = dataio.dataio('Test', args, conf, args.outdir)
    test_data.loader_imgsize = (args.decoder_imgsize, args.decoder_imgsize)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    net = model.UniPS(
        device=device,
        encoder_size=args.encoder_imgsize,
        decoder_size=args.decoder_imgsize,
        num_samples=args.num_samples,
        num_agg_enc=args.num_agg_enc,
    ).to(device)
    
    model_utils.load_checkpoint(
        net=net, device=device, checkpoint_path=args.pretrained)

    result = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing', leave=False):
            _, mae, normal_map, error_map = net(batch=batch, mode='test')
            normal_map = normal_map.detach().cpu().numpy()
            error_map = error_map.detach().cpu().numpy()
            mae_value = float(mae.detach().cpu().item())

            data_workspace = test_loader.dataset.data.data_workspace
            cv2.imwrite(
                f'{data_workspace}/normal.png',
                255 * normal_map[0].transpose(1, 2, 0)[:, :, ::-1].astype('uint8')
            )
            cv2.imwrite(
                f'{data_workspace}/error.png',
                make_error_heatmap(error_map)
            )
            result[data_workspace] = mae_value

    for obj_name, mae in result.items():
        logger.info(f'{obj_name}: MAE = {mae:.4f} degrees')
    logger.info(f'Testing completed. avg MAE: {sum(result.values()) / len(result):.4f} degrees')


if __name__ == '__main__':
    main()

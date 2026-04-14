import os
import logging
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.io import dataio
from modules.model import model
from modules.config import config
from modules.model import model_utils

try:
    import swanlab
    swanlab_available = True
except ImportError:
    swanlab_available = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='UniPS Train')
parser.add_argument('--session_name', default='train_session',
    help='训练会话名称')
parser.add_argument('--training_dir', default='/home/user/dataset/PSWild',
    help='训练数据集路径')
parser.add_argument('--epoch', type=int, default=10,
    help='训练轮数')
parser.add_argument('--batchsize', type=int, default=1,
    help='批大小')
parser.add_argument('--outdir', default='output/train_session',
    help='输出根目录，用于保存检查点、日志和测试结果')
parser.add_argument('--pretrained', default=None,
    help='checkpoint 文件路径；从头训练时留空')
parser.add_argument('--num_agg_enc', type=int, default=3,
    help='聚合 Transformer 中编码器 SAB 的层数')
parser.add_argument('--min_nimg', type=int, default=2,
    help='训练时每个物体最少采样的输入图像数')
parser.add_argument('--num_samples', type=int, default=6144,
    help='训练时每个物体最大像素采样数')
parser.add_argument('--lr', type=float, default=0.0001,
    help='AdamW 优化器初始学习率')
parser.add_argument('--lr_scheduler', default='step', choices=['step', 'cos'],
    help='学习率调度器类型: step 或 cos')
parser.add_argument('--encoder_imgsize', type=int, default=256,
    help='送入编码器的图像分辨率')
parser.add_argument('--decoder_imgsize', type=int, default=512,
    help='输出阶段使用的图像分辨率')


def main():
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.outdir, exist_ok=True)
    conf = config.TrainConfig()

    train_data = dataio.dataio('Train', args, conf, args.outdir)
    train_data.loader_imgsize = (args.decoder_imgsize, args.decoder_imgsize)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    net = model.UniPS(
        device=device,
        min_nimg=args.min_nimg,
        encoder_size=args.encoder_imgsize,
        decoder_size=args.decoder_imgsize,
        num_samples=args.num_samples,
        num_agg_enc=args.num_agg_enc,
    ).to(device)
    optimizer, scheduler = model_utils.build_optimizer_and_scheduler(net, args)
    start_epoch = model_utils.load_checkpoint(
        net=net, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        checkpoint_path=args.pretrained, 
        device=device)

    if swanlab_available:
        swanlab.init(
            project='Universal-PS',
            name=f"{args.session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=args.__dict__,
            logdir=os.path.join(args.outdir, 'logs'),
        )

    for epoch in range(start_epoch, args.epoch):
        epoch_loss = 0.0
        epoch_mae = 0.0
        step_count = 0

        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{args.epoch}', leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            loss, mae, _, _ = net(batch=batch, mode='train')
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            mae_value = float(mae.detach().cpu().item())
            epoch_loss += loss_value
            epoch_mae += mae_value
            step_count += 1

            avg_loss = epoch_loss / step_count
            avg_mae = epoch_mae / step_count
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'mae': f'{mae_value:.4f}',
                'avg_mae': f'{avg_mae:.4f}',
            })

            if swanlab_available:
                swanlab.log({
                    'train/loss': loss_value,
                    'train/avg_loss': avg_loss,
                    'train/mae': mae_value,
                    'train/avg_mae': avg_mae,
                    'train/lr': optimizer.param_groups[0]['lr'],
                })

        scheduler.step()
        checkpoint_path = model_utils.save_checkpoint(net, optimizer, scheduler, epoch + 1, args)
        logger.info(
            f'Epoch {epoch + 1}/{args.epoch} completed. '
            f'Average Loss: {epoch_loss / max(step_count, 1):.4f}. '
            f'Average MAE: {epoch_mae / max(step_count, 1):.4f}. '
            f'Checkpoint saved to {checkpoint_path}. '
            f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )


if __name__ == '__main__':
    main()

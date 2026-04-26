import torch
import swanlab
import argparse
from tqdm import tqdm
from test import test
from datetime import datetime
from modules.io import dataio
from modules.utils import log
from modules.config import config
from torch.utils.data import DataLoader
from modules.model import model, model_utils

parser = argparse.ArgumentParser(description='UniPS')
parser.add_argument('--session_name', default = 'train_session',
    help='训练会话名称')
parser.add_argument('--training_dir', default = '/home/user/dataset/PSWild',
    help='训练数据集路径')
parser.add_argument('--epoch', type=int, default = 20,
    help='训练轮数')
parser.add_argument('--batchsize', type=int, default = 2,
    help='训练批大小，即每批的物体数量')
parser.add_argument('--outdir', default='output/train_session_global',
    help='输出根目录，用于保存检查点、日志和测试结果')
parser.add_argument('--pretrained', default=None,
    help='预训练检查点目录路径，用于恢复训练或推理')
parser.add_argument('--num_agg_enc', type=int, default=3,
    help='聚合 Transformer 中编码器 SAB (集合注意力块) 的层数')
parser.add_argument('--min_nimg', type=int, default=1,
    help='训练时每个物体最少采样的输入图像数; 网络会在 [min_nimg, 总图像数] 范围内随机选取')
parser.add_argument('--num_samples', type=int, default=2048,
    help='训练时每个物体最大像素采样数; 从前景掩码内随机抽取，用于限制显存占用')
parser.add_argument('--lr', type=float, default=0.00001,
    help='AdamW 优化器初始学习率，统一应用于编码器、聚合模块和预测头')
parser.add_argument('--lr_scheduler', default='step',
    help='学习率调度器类型: "step" (每3轮衰减0.8) 或 "cos" (余弦退火，30轮) (默认: step)')
parser.add_argument('--save_freq', type=int, default=10,
    help='模型保存频率，每多少个epoch保存一次 (默认: 5)')
parser.add_argument('--train_with_test', default=True, 
    help='是否在训练时使用测试数据集测试')
parser.add_argument('--test_dir', default = '/home/user/dataset/DiLiGenT_512',
    help='测试数据集路径')
parser.add_argument('--enable_swanlab', default = True,
    help='是否启用 SwanLab 日志记录')

def main():
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger = log.setup_logger(outdir=args.outdir)

    swanlab.init(
        project="UniPS",
        name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config = args.__dict__, 
        logdir = str(args.outdir + '/logs'),
        mode='disabled' if not args.enable_swanlab else 'cloud', 
    )

    # 初始化超参数
    conf = config.TrainConfig()

    # 初始化数据集读取
    train_data = dataio.dataio(mode='Train', args=args, conf=conf, outdir=args.outdir)
    if train_data is None:
        raise RuntimeError("Failed to load train data.")

    train_data_loader = DataLoader(
        dataset=train_data, 
        batch_size=args.batchsize, 
        shuffle=True, 
        num_workers=12, 
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True)
    
    # 初始化模型
    net = model.UniPS(args, device).to(device)
    if args.pretrained is None:
        logger.info("No pretrained model specified. Training from scratch.")
        net.init_weights()
    net.train()

    # 创建 optimizer / scheduler
    optimizer, scheduler = model_utils.build_optimizer_scheduler(
        net,
        lr=args.lr,
        scheduler_type=args.lr_scheduler
    )

    # 恢复 checkpoint
    epoch = 0
    if args.pretrained is not None:
        epoch = net.load(
            outdir=args.pretrained,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        logger.info(f"Resume from epoch {epoch}")

    #开始训练
    global_step = 0
    losses = 0.0
    for epoch in range(epoch, args.epoch):
        pbar = tqdm(train_data_loader,desc=f'Train Epoch {epoch+1}/{args.epoch}')
        for batch in pbar:
            with torch.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                optimizer.zero_grad()
                loss, _, _, _ = net(batch)
                loss.backward()
                optimizer.step()
            losses += loss.item()
            global_step += 1

            pbar.set_postfix(
                ordered_dict={
                    'loss': f'{loss.item():.4f}', 
                    'avg_loss': f'{losses/global_step:.4f}',
                })
            
            swanlab.log(data={
                'train/loss': loss.item(),
                'train/avg_loss': losses/global_step,
                'train/lr': scheduler.get_last_lr()[0],
            }, step=global_step)

        # 每个epoch结束后测试
        if args.train_with_test:
            result = test(args, net, device)
            avg_mae = sum([mae for _, (_,mae,_,_) in result.items()]) / len(result)
            logger.info(f"Average test MAE: {avg_mae:.4f} degrees")
            swanlab.log(data={
                'test/mae': avg_mae,
            }, step=global_step)
        
        # 保存权重文件
        savedir = args.outdir + '/checkpoint/'
        if (epoch + 1) % args.save_freq == 0:
            net.save(
                outdir=savedir,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
            )
            logger.info(f'Model saved to {savedir}.')
        
        net.train()
        scheduler.step()    # 更新学习率
        logger.info(f'Epoch {epoch+1}/{args.epoch} completed. Average Loss: {losses/global_step:.4f}.')

    swanlab.finish()

if __name__ == '__main__':
    main()
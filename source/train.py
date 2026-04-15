import torch
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from modules.io import dataio
from modules.model import model, model_utils
from modules.config import config
from torch.utils.data import DataLoader

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

parser = argparse.ArgumentParser(description='UniPS')
parser.add_argument('--session_name', default = 'train_session',
    help='训练会话名称')
parser.add_argument('--training_dir', default = '/home/user/dataset/PSWild',
    help='训练数据集路径')
parser.add_argument('--epoch', type=int, default = 10,
    help='训练轮数')
parser.add_argument('--batchsize', type=int, default = 1,
    help='训练批大小，即每批的物体数量')
parser.add_argument('--outdir', default='output/train_session',
    help='输出根目录，用于保存检查点、日志和测试结果')
parser.add_argument('--pretrained', default='/home/user/code/vggt_unips/output/train_session/checkpoint/20260415_214320_model.pt',
    help='预训练检查点目录路径，用于恢复训练或推理')
parser.add_argument('--num_agg_enc', type=int, default=3,
    help='聚合 Transformer 中编码器 SAB (集合注意力块) 的层数')
parser.add_argument('--min_nimg', type=int, default=2,
    help='训练时每个物体最少采样的输入图像数; 网络会在 [min_nimg, 总图像数] 范围内随机选取')
parser.add_argument('--num_samples', type=int, default=6144,
    help='训练时每个物体最大像素采样数; 从前景掩码内随机抽取，用于限制显存占用')
parser.add_argument('--lr', type=float, default=0.00001,
    help='AdamW 优化器初始学习率，统一应用于编码器、聚合模块和预测头')
parser.add_argument('--lr_scheduler', default='step',
    help='学习率调度器类型: "step" (每3轮衰减0.8) 或 "cos" (余弦退火，30轮) (默认: step)')
parser.add_argument('--lr_init_scale', type=float, default=1.0,
    help='初始学习率的缩放因子，启动时将 lr 乘以该系数; 在微调预训练模型时有用 (默认: 1.0)')
parser.add_argument('--grad_loss_weight', type=float, default=0.05,
    help='低分辨率梯度差辅助损失的权重 μ；总损失在 canonical 阶段额外加入 μL_g')

def main():
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # swanlab_available = False

    if swanlab_available:
        swanlab.init(
            project="Universal-PS",
            name=f"UniPS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config = args.__dict__, 
            logdir = str(args.outdir + '/logs'),
        )

    # 初始化超参数
    conf = config.TrainConfig()

    # 初始化数据集读取
    train_data = dataio.dataio('Train', args, conf, args.outdir)
    if train_data is None:
        raise RuntimeError("Failed to load train data.")

    train_data_loader = DataLoader(
        dataset = train_data, 
        batch_size = args.batchsize, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True)
    
    # 初始化模型
    net = model.UniPS(args, device).to(device)
    if args.pretrained is not None:
        net.load(args.pretrained)
    else:
        net.init_weights()
    net.set_mode('Train')

    # 构建优化器和学习率调度器
    optimizer, scheduler = model_utils.build_optimizer_scheduler(
        net,
        lr=args.lr,
        scheduler_type=args.lr_scheduler
    )

    #开始训练
    global_step = 0
    losses = 0
    for epoch in range(args.epoch):
        with torch.autocast(device_type=device.type, enabled = False):
            pbar = tqdm(train_data_loader,desc=f'Train Epoch {epoch+1}/{args.epoch}', leave=False)
            for batch in pbar:
                optimizer.zero_grad()
                loss, mae, _, _ = net(batch)
                loss.backward()
                optimizer.step()
                losses += loss
                global_step += 1

                pbar.set_postfix(
                    {'loss': f'{loss:.4f}', 
                     'avg_loss': f'{losses/global_step:.4f}',
                    })
                
                if swanlab_available:
                    swanlab.log({
                        'loss': loss,
                        'avg_loss': losses/global_step,
                        'mae': float(mae)
                    })

        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        savedir = args.outdir + '/checkpoint/' + time
        net.save(savedir)   # 每个epoch后保存模型
        scheduler.step()    # 更新学习率
        logger.info(f'Epoch {epoch+1}/{args.epoch} completed. Average Loss: {losses/global_step:.4f}. Models saved to {savedir}.')

if __name__ == '__main__':
    main()
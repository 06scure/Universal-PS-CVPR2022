import torch
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from modules.io import dataio
from modules.model import model
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
parser.add_argument('--agg_type', default='Transformer', choices=['Transformer', 'Pooling'],
    help='聚合解码器类型，将逐图像特征融合为全局光照上下文')
parser.add_argument('--epoch', type=int, default = 1,
    help='训练轮数')
parser.add_argument('--batchsize', type=int, default = 1,
    help='训练批大小，即每批的物体数量')
parser.add_argument('--outdir', default='output/train_session',
    help='输出根目录，用于保存检查点、日志和测试结果')
parser.add_argument('--pretrained', default='/home/user/code/Universal-PS-CVPR2022/output/train_session/checkpoint/20260413_143851',
    help='预训练检查点目录路径，用于恢复训练或推理')
parser.add_argument('--num_agg_enc', type=int, default=3,
    help='聚合 Transformer 中编码器 SAB (集合注意力块) 的层数')
parser.add_argument('--min_nimg', type=int, default=4,
    help='训练时每个物体最少采样的输入图像数; 网络会在 [min_nimg, 总图像数] 范围内随机选取')
parser.add_argument('--num_samples', type=int, default=4096,
    help='训练时每个物体最大像素采样数; 从前景掩码内随机抽取，用于限制显存占用')
parser.add_argument('--lr', type=float, default=0.0001,
    help='AdamW 优化器初始学习率，统一应用于编码器、聚合模块和预测头')
parser.add_argument('--lr_scheduler', default='step',
    help='学习率调度器类型: "step" (每3轮衰减0.8) 或 "cos" (余弦退火，30轮) (默认: step)')
parser.add_argument('--lr_init_scale', type=float, default=1.0,
    help='初始学习率的缩放因子，启动时将 lr 乘以该系数; 在微调预训练模型时有用 (默认: 1.0)')
parser.add_argument('--encoder_imgsize', type=int, default=256,
    help='送入 Swin Transformer 编码器的图像分辨率 (高=宽); 输入图像在编码前会被缩放到此尺寸')
parser.add_argument('--decoder_imgsize', type=int, default=512,
    help='送入解码器的图像分辨率 (高=宽); 输入图像在编码前会被缩放到此尺寸')

def main():
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # swanlab_available = False   # 测试模式

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
    train_data.loader_imgsize = (args.decoder_imgsize, args.decoder_imgsize)
    train_data_loader = DataLoader(
        dataset = train_data, 
        batch_size = args.batchsize, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True)
    
    # 初始化模型
    net = model.Net(args, device)
    if args.pretrained is not None:
        net.load_models(args.pretrained)
    net.set_mode('Train')

    #开始训练
    global_step = 0
    losses = 0
    for epoch in range(args.epoch):
        with torch.autocast(device_type=device.type, enabled = False):
            pbar = tqdm(train_data_loader,desc=f'Train Epoch {epoch+1}/{args.epoch}', leave=False)
            for batch in pbar:
                loss, mae, _, _ = net.step(
                    batch,
                decoder_imgsize=(args.decoder_imgsize, args.decoder_imgsize),
                encoder_imgsize=(args.encoder_imgsize, args.encoder_imgsize))

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
                        'mae': mae
                    })

        #每个epoch后保存模型
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        savedir = args.outdir + '/checkpoint/' + time
        net.save_models(savedir)
        net.scheduler_step()
        logger.info(f'Epoch {epoch+1}/{args.epoch} completed. Average Loss: {losses/global_step:.4f}. Models saved to {savedir}.')

if __name__ == '__main__':
    main()
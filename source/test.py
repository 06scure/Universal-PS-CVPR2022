import cv2
import torch
import logging
import argparse
from tqdm import tqdm
from modules.io import dataio
from modules.model import model
from modules.config import config
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='UniPS')
parser.add_argument('--session_name', default = 'test_session',
    help='训练/测试会话名称')
parser.add_argument('--test_dir', default = '/home/user/dataset/DiLiGenT_512',
    help='测试数据集路径')
parser.add_argument('--agg_type', default='Transformer', choices=['Transformer', 'Pooling'],
    help='聚合解码器类型，将逐图像特征融合为全局光照上下文')
parser.add_argument('--batchsize', type=int, default = 1,
    help='训练批大小，即每批的物体数量')
parser.add_argument('--outdir', default='output/test_session',
    help='输出根目录，用于保存检查点、日志和测试结果')
parser.add_argument('--pretrained', default='/home/user/code/Universal-PS-CVPR2022/output/pswild_train_session/checkpoint/20260412_190410',
    help='预训练检查点目录路径，用于恢复训练或推理')
parser.add_argument('--num_agg_enc', type=int, default=3,
    help='聚合 Transformer 中编码器 SAB (集合注意力块) 的层数')
parser.add_argument('--min_nimg', type=int, default=10,
    help='训练时每个物体最少采样的输入图像数; 网络会在 [min_nimg, 总图像数] 范围内随机选取')
parser.add_argument('--num_samples', type=int, default=2500,
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

    # 初始化超参数
    conf = config.TestConfig()  # 从 config 模块获取配置对象

    # 初始化数据集读取
    test_data = dataio.dataio('Test', args, conf, args.outdir)
    if test_data is None:
        raise RuntimeError("Failed to load test data.")
    test_data.loader_imgsize = (args.decoder_imgsize, args.decoder_imgsize)
    test_data_loader = DataLoader(
        dataset = test_data, 
        batch_size = args.batchsize, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True)

    # 初始化模型
    net = model.Net(args, device)
    if args.pretrained is not None:
        net.load_models(args.pretrained)
    net.set_mode('Test')

    # 进行测试
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc=f'Testing Epoch', leave=False):
            # mask = batch[2]  # 获取原始mask (B, 1, H, W)
            loss, output, input = net.step( # output = [B, 3, h, w]
                batch, 
                decoder_imgsize=(args.decoder_imgsize, args.decoder_imgsize),
                encoder_imgsize=(args.encoder_imgsize, args.encoder_imgsize))

            # 保存输入和输出
            cv2.imwrite(f'{test_data.data.data_workspace}/input.png', 255 * input[0,:,:,:].transpose(1,2,0)[:,:,::-1])
            cv2.imwrite(f'{test_data.data.data_workspace}/normal.png', 255 * output[0,:,:,:].transpose(1,2,0)[:,:,::-1])

            #TODO: 输出误差热力图
    
    logger.info("Testing completed.")

if __name__ == '__main__':
    main()

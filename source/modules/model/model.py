from .model_utils import *
from ..utils.ind2sub import ind2coords
import os
import torch
import logging
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_, trunc_normal_
from typing import Optional, Tuple

from .utils.folked import Transformer
from .utils.folked import swin_transformer
from .utils.folked import uper

logger = logging.getLogger(__name__)

class PredictionHead(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(PredictionHead, self).__init__()
        modules_regression = []
        modules_regression.append(nn.Linear(dim_input, dim_input//2))
        modules_regression.append(nn.ReLU(inplace=False))
        modules_regression.append(nn.Linear(dim_input//2, dim_output))
        self.regression = nn.Sequential(*modules_regression)

    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):# or isinstance(m, nn.Linear):
                    kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.bias.data.zero_()
                    m.weight.data.fill_(1.0)

    def forward(self, x):
        # [B, C] -> [B, 3]
        return self.regression(x)

class Encoder(nn.Module):
    def __init__(self, input_nc):
        super(Encoder, self).__init__()
        back = []
        fuse = []

        in_channels = (96, 192, 384, 768)
        back.append(swin_transformer.SwinTransformer(in_chans=input_nc))

        fuse.append(uper.UPerHead(in_channels = in_channels))
        attn = []
        for i in range(len(in_channels)):
            attn.append(self.attn_block(in_channels[i]))
        self.attn = nn.Sequential(*attn)

        self.backbone = nn.Sequential(*back)
        self.fusion = nn.Sequential(*fuse)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):# or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def attn_block(self, dim, num_attn = 1):
        attn = []
        for k in range(num_attn):
            attn.append(Transformer.SAB(dim, dim, num_heads=8, ln=False, attention_dropout = 0.1, dim_feedforward = 2 * dim))
        return nn.Sequential(*attn)

    def conv_block(self, in_planes, out_planes, kernel_size):
        conv = nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                            kernel_size=kernel_size, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=False),
            )
        return conv

    def forward(self, x):
        """
        Arg:
            image [B, N, C, H, W]
        Return:
            feature [B, N, Cout, H/4, W/4]
        """
        feats = []  # 每帧之间的swin transformer特征,为 [96, 192, 384, 768]四个尺度的特征图
        for k in range(x.shape[1]):
            feats.append(self.backbone(x[:, k, :, :, :]))

        out = [] # layer first
        # 帧间 communication
        for l in range(len(feats[0])):  # 同一尺度的特征图进行跨帧融合
            in_fuse = []
            for k in range(x.shape[1]): # 每个视角的第l层特征图
                in_fuse.append(feats[k][l])
            in_fuse = torch.stack(in_fuse, dim=1)
            B, N, C, H, W = in_fuse.size()
            in_fuse = in_fuse.permute(0,3,4,1,2).reshape(-1, N, C) # [B*H*W, N, C]
            # 将拼接好的特征图送入 SAB 进行跨视图融合
            out_fuse = self.attn[l](in_fuse).reshape(B, H, W, N, C).permute(0,3,4,1,2) # [B, N, C, H, W]
            out.append(out_fuse)

        feats = []  # SAB 融合后的特征图
        for k in range(x.shape[1]):
            feats.append((out[0][:,k,:,:,:], out[1][:,k,:,:,:], out[2][:,k,:,:,:], out[3][:,k,:,:,:]))

        outs = []   # 送入 UPerHead 进行多尺度融合，输出统一维度的特征图
        for k in range(x.shape[1]):
            outs.append(self.fusion(feats[k]))
        feats = torch.stack(outs, 1) # [B, N, C, H/4, W/4], h,w =256
        return feats

class UniPS(nn.Module):
    def __init__(self, args, device):
        super(UniPS, self).__init__()
        self.device = device
        self.min_nimg = getattr(args, 'min_nimg', 2)   # 输入图像
        self.num_samples = getattr(args, 'num_samples', 4096) # 采样像素
        self.model_name = args.session_name

        self.encoder_size = getattr(args, 'encoder_imgsize', 256)
        self.decoder_size = getattr(args, 'decoder_imgsize', 512)

        self.encoder = Encoder(4)

        self.aggregation = Transformer.AggregationBlock(
            dim_input = 256 + 3, 
            num_outputs = 1, 
            dim_hidden=384, 
            dim_feedforward = 1024, 
            num_heads=8, 
            ln=True, 
            attention_dropout=0.1)

        self.prediction = PredictionHead(384, 3) # No urcainty

        self.criterionL2 = nn.MSELoss(reduction = 'sum')

    def forward(self, batch) -> Tuple[torch.Tensor, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Args:
            batch: [img, nml, mask]
                img: [B, N, 3, H, W]
                nml: [B, 3, H, W]
                mask: [B, 1, H, W]
        Return:
            - loss: scalar loss value for backpropagation / evaluation
            - mae: scalar mean angular error in degrees for evaluation
            - normal_map: [B, 3, H, W] tensor for visualization, where the 3 channels are predicted normals mapped to [0, 1]
            - error_map: [B, 1, H, W] tensor for visualization, where the single channel is the per-pixel angular error to the GT normal (in degrees)
        """

        # dataloader 输出的多视角图像维度是 [B, H, N, C, W]，这里统一转成 [B, N, C, H, W]
        img = batch[0].permute(0, 4, 1, 2, 3).to(self.device)# [B, N, C, H, W]
        nml = batch[1].to(self.device)  # [B, 3, H, W]
        mask = batch[2].to(self.device) # [B, 1, H, W]

        min_nimg = self.min_nimg
        if self.training is True and img.shape[1] >= min_nimg:
            # 训练阶段随机抽取部分输入视角，增强模型对输入视角数量变化的鲁棒性
            numI = np.random.randint(img.shape[1]-min_nimg+1)+min_nimg
            imgid = np.random.permutation(range(img.shape[1]))[:numI]
            img = img[:, imgid, :, :, :]


        """ Feature Encoding Stage"""
        B, N, C, H, W = img.shape

        img_ = img.reshape(-1, C, H, W)
        img_ = F.interpolate(img_, size=self.encoder_size, mode='bilinear', align_corners=False).reshape(B, N, C, self.encoder_size, self.encoder_size)
        mask_ = F.interpolate(mask, size=self.encoder_size, mode='nearest')
        data = torch.cat([img_ * mask_.unsqueeze(1).expand(-1, img.shape[1], -1, -1, -1), mask_.unsqueeze(1).expand(-1, img.shape[1], -1, -1, -1)], dim=2)

        feats = self.encoder(data) # [B, N, 256, H/4, W/4]，每个视角一张特征图

        """Process at Canonical Resolution"""
        B, N, C, H, W = feats.shape

        img_ = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        img_ = F.interpolate(img_, size= (H, W), mode='bilinear', align_corners=False).reshape(img.shape[0], img.shape[1], img.shape[2], H, W)
        m = F.interpolate(mask, size = (H, W), mode='nearest')
        # GT 法线也被缩放到 canonical resolution，并重新归一化到单位球面
        n = F.normalize(F.interpolate(nml, size = (H, W), mode='bilinear', align_corners=False), p=2, dim=1)

        loss = torch.tensor(0.0, device=self.device)
        nout = torch.zeros(B, H * W, 3).to(self.device)

        for b in range(B):
            # m_: 每个像素是否有效；n_: 对应像素的 GT normal
            m_ = m[b, :, :, :].reshape(-1, H * W).permute(1,0)
            n_ = n[b, :, :, :].reshape(-1, H * W).permute(1,0)
            ids = torch.nonzero(m_>0)[:,0]
            # f: 每个有效像素在 N 个视角上的 encoder 特征，形状 [num_valid, N, C]
            f = feats[b, :, :, :, :].reshape(-1, C, H * W).permute(2, 0, 1)
            f = f[ids, :, :]
            n_ = n_[ids, :]
            # o: 每个有效像素在 N 个视角上的 RGB 观测，形状 [num_valid, N, 3]
            o = img_[b, :, :, :, :].reshape(-1, 3, H * W).permute(2, 0, 1)
            o = o[ids, :, :]
            # 聚合器输入是 [RGB, feature] 的拼接，沿着视角维进行跨视图融合
            x = torch.cat([o, f], dim=2)
            feat_gg = self.aggregation(x)
            out_nml = self.prediction(feat_gg)
            nout_ = F.normalize(out_nml[:, :3],dim=1, p=2)
            nout[b, ids, :] = nout_
            # 低分辨率下做了一次损失
            loss += self.criterionL2(nout_, n_) / len(ids)

        """Prediction at Original Resolution"""
        img_ = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        img_ = F.interpolate(img_, size= self.decoder_size, mode='bilinear', align_corners=False).reshape(img.shape[0], img.shape[1], img.shape[2], self.decoder_size, self.decoder_size)
        m = F.interpolate(mask, size = self.decoder_size, mode='nearest')
        n = F.normalize(F.interpolate(nml, size = self.decoder_size, mode='bilinear', align_corners=False), p=2, dim=1)

        B = img.shape[0]
        N = img.shape[1]
        C = feats.shape[2] + img.shape[2]
        H = self.decoder_size
        W = self.decoder_size

        mae = torch.tensor(0.0, device=self.device)

        if self.training is True:
            nout = torch.zeros(B, H * W, 3).to(self.device)
            numMaxSamples = self.num_samples
            mae_sum = torch.tensor(0.0, device=self.device)
            mae_count = 0
            for b in range(B):
                m_ = m[b, :, :, :].reshape(-1, H * W).permute(1,0)
                n_ = n[b, :, :, :].reshape(-1, H * W).permute(1,0)

                ids = torch.nonzero(m_.squeeze(1) > 0, as_tuple=False).squeeze(1)
                if ids.numel() == 0:
                    continue
                if ids.numel() > numMaxSamples:
                    # 高分辨率阶段只随机采样部分有效像素，避免显存/算力开销过大
                    perm = torch.randperm(ids.numel(), device=ids.device)[:numMaxSamples]
                    ids = ids[perm]

                coords = ind2coords((H, W), ids)

                feat = feats[b, :, :, :, :]
                n_ = n_[ids, :]

                x = []
                for k in range(N):
                    # 用 grid_sample 在高分辨率坐标处双线性采样低分辨率 encoder 特征
                    f = F.grid_sample(feat[[k], :, :, :], coords.to(self.device), mode='bilinear', align_corners=False).squeeze().permute(1,0)
                    o = img_[b, k, :, :, :]
                    o = o.reshape(o.shape[0], o.shape[1] * o.shape[2]).permute(1,0)
                    o = o[ids, :]
                    x.append(torch.cat([o, f], dim=1))
                x = torch.stack(x, 1)

                feat_gg = self.aggregation(x)
                out_nml = self.prediction(feat_gg)
                nout_ = F.normalize(out_nml[:, :3],dim=1, p=2)
                nout[b, ids, :] = nout_
                # 逐像素又做了一次损失
                loss += self.criterionL2(nout_, n_) / ids.numel()
                mae_sum += angular_error(nout_, n_).sum()
                mae_count += ids.numel()

            mae = mae_sum / max(mae_count, 1)
            nout_high = nout.permute(0, 2, 1).reshape(B, 3, H, W)
            mask_high = m

        if self.training is False:
            nout = torch.zeros(B, H * W, 3).to(self.device)
            loss = torch.tensor(0.0, device=self.device)
            numMaxSamples = 10000
            for b in range(B):
                m_ = m[b, :, :, :].reshape(-1, H * W).permute(1,0)
                n_ = n[b, :, :, :].reshape(-1, H * W).permute(1,0)
                ids = torch.nonzero(m_.squeeze(1) > 0, as_tuple=False).squeeze(1)
                id_chunks = torch.split(ids, numMaxSamples) if ids.numel() > 0 else (ids,)
                feat = feats[b, :, :, :, :]
                for ids_chunk in id_chunks:
                    if ids_chunk.numel() == 0:
                        continue
                    x = []
                    coords = ind2coords((H, W), ids_chunk)
                    for k in range(N):
                        f = F.grid_sample(feat[[k], :, :, :], coords.to(self.device), mode='bilinear', align_corners=False).squeeze().permute(1,0)
                        o = img_[b, k, :, :, :]
                        o = o.reshape(o.shape[0], o.shape[1] * o.shape[2]).permute(1,0)
                        o = o[ids_chunk, :]
                        x.append(torch.cat([o, f], dim=1))
                    x = torch.stack(x, 1)
                    feat_gg = self.aggregation(x)
                    out_nml = self.prediction(feat_gg)
                    nout_ = F.normalize(out_nml[:, :3],dim=1, p=2)
                    nout[b, ids_chunk, :] = nout_
            nout_high = nout.permute(0, 2, 1).reshape(B, 3, H, W)
            mask_high = m
            mae = angular_error(nout_high, n, mask_high)

        # 返回可视化结果
        if self.training is False:
            normal_map = get_normal_map(nout_high, mask_high)
            error_map = get_error_map(nout_high, n, mask_high)
            return loss / B, mae.detach().cpu().item() / B, normal_map.detach().cpu().numpy(), error_map.detach().cpu().numpy()
        else:
            return loss / B, mae.detach().cpu().item() / B, None, None
    
    def save(self, 
             outdir:str,
             optimizer:Optional[torch.optim.Optimizer] = None,
             scheduler:Optional[torch.optim.lr_scheduler.LRScheduler] = None,
             epoch:int = 0):
        os.makedirs(outdir, exist_ok = True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(outdir, f'{timestamp}.pt')
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch,
        }, checkpoint_path)

        return checkpoint_path
    
    def load(self,
            outdir:str,
            optimizer:Optional[torch.optim.Optimizer] = None,
            scheduler:Optional[torch.optim.lr_scheduler.LRScheduler] = None,
            device:torch.device = torch.device('cuda')):
        if outdir is None:
            raise ValueError("Checkpoint directory must be provided for loading.")

        checkpoint = torch.load(outdir, map_location=device)
        self.load_state_dict(state_dict=checkpoint['state_dict'])

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(state_dict=checkpoint['optimizer'])

        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(state_dict=checkpoint['scheduler'])

        epoch = int(checkpoint.get('epoch', 0))
        return epoch
    
    def init_weights(self):
        self.encoder.init_weights()
        self.aggregation.init_weights()
        self.prediction.init_weights()
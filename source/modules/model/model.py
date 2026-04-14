from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_, trunc_normal_

from ..utils.ind2sub import ind2coords
from .model_utils import angular_error
from .utils import Transformer
from .utils.folked import swin_transformer
from .utils.folked import uper


class PredictionHead(nn.Module):
    def __init__(self, dim_input: int, dim_output: int) -> None:
        super().__init__()
        self.regression = nn.Sequential(
            nn.Linear(dim_input, dim_input // 2),
            nn.ReLU(inplace=False),
            nn.Linear(dim_input // 2, dim_output),
        )

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regression(x)


class Encoder(nn.Module):
    def __init__(self, input_nc: int) -> None:
        super().__init__()
        in_channels = (96, 192, 384, 768)
        self.backbone = nn.Sequential(
            swin_transformer.SwinTransformer(in_chans=input_nc)
        )
        self.fusion = nn.Sequential(
            uper.UPerHead(in_channels=in_channels)
        )
        self.attn = nn.Sequential(*[self.attn_block(dim) for dim in in_channels])

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def attn_block(self, dim: int, num_attn: int = 1) -> nn.Sequential:
        return nn.Sequential(*[
            Transformer.SAB(
                dim,
                dim,
                num_heads=8,
                ln=False,
                attention_dropout=0.1,
                dim_feedforward=2 * dim,
            )
            for _ in range(num_attn)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.backbone(x[:, view_idx]) for view_idx in range(x.shape[1])]

        fused_layers = []
        for layer_idx in range(len(feats[0])):
            layer_features = torch.stack(
                [feats[view_idx][layer_idx] for view_idx in range(x.shape[1])],
                dim=1,
            )
            batch_size, num_views, channels, height, width = layer_features.shape
            layer_features = layer_features.permute(0, 3, 4, 1, 2).reshape(-1, num_views, channels)
            fused = self.attn[layer_idx](layer_features)
            fused = fused.reshape(batch_size, height, width, num_views, channels).permute(0, 3, 4, 1, 2)
            fused_layers.append(fused)

        fused_views = [
            tuple(layer[:, view_idx] for layer in fused_layers)
            for view_idx in range(x.shape[1])
        ]
        outputs = [self.fusion(view_features) for view_features in fused_views]
        return torch.stack(outputs, dim=1)


class UniPS(nn.Module):
    def __init__(
        self,
        device: torch.device,
        min_nimg: int = 2,
        encoder_size: int | tuple[int, int] = 256,
        decoder_size: Optional[int | tuple[int, int]] = None,
        num_samples: int = 2048,
        max_samples: int = 10000,
        num_agg_enc: int = 3,
    ) -> None:
        super().__init__()
        self.device = device
        self.min_nimg = min_nimg
        self.encoder_imgsize = encoder_size
        self.decoder_imgsize = decoder_size
        self.num_samples = num_samples
        self.max_num_samples = max_samples

        self.encoder = Encoder(input_nc=4)
        self.aggregation = Transformer.TransformerLayer(
            dim_input=256 + 3,
            num_enc_sab=num_agg_enc,
            num_outputs=1,
            dim_hidden=384,
            dim_feedforward=1024,
            num_heads=8,
            ln=True,
            attention_dropout=0.1,
        )
        self.prediction = PredictionHead(dim_input=384, dim_output=3)
        self.criterion_l2 = nn.MSELoss(reduction='sum')
        self.init_weights()

    def init_weights(self) -> None:
        self.encoder.init_weights()
        self.aggregation.init_weights()
        self.prediction.init_weights()

    def _resolve_size(
        self,
        size: Optional[int | tuple[int, int]],
        fallback: tuple[int, int],
    ) -> tuple[int, int]:
        if size is None:
            return fallback
        if isinstance(size, int):
            return (size, size)
        return size

    def _build_tokens(
        self,
        feature_map: torch.Tensor,
        images_high: torch.Tensor,
        coords: torch.Tensor,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens = []
        for view_idx in range(images_high.shape[0]):
            sampled = F.grid_sample(
                feature_map[[view_idx]],
                coords,
                mode='bilinear',
                align_corners=False,
            )
            feature_tokens = sampled[0, :, 0, :].transpose(0, 1)
            observation_tokens = images_high[view_idx].flatten(1).transpose(0, 1)[ids]
            tokens.append(torch.cat([observation_tokens, feature_tokens], dim=1))
        return torch.stack(tokens, dim=1)

    def forward(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode not in ('train', 'test'):
            raise ValueError("Mode must be one of ['train', 'test']")

        self.train(mode == 'train')

        images = batch[0].permute(0, 4, 1, 2, 3).to(device=self.device)
        normal = batch[1].to(device=self.device)
        mask = batch[2].to(device=self.device)

        if mode == 'train' and images.shape[1] >= self.min_nimg:
            num_images = torch.randint(
                low=self.min_nimg,
                high=images.shape[1] + 1,
                size=(1,),
                device=images.device,
            ).item()
            image_indices = torch.randperm(images.shape[1], device=images.device)[:num_images]
            images = images.index_select(1, image_indices)

        batch_size, num_views, num_channels, image_height, image_width = images.shape
        encoder_size = self._resolve_size(self.encoder_imgsize, (image_height, image_width))
        decoder_size = self._resolve_size(self.decoder_imgsize, (image_height, image_width))

        images_flat = images.reshape(-1, num_channels, image_height, image_width)

        images_encoder = F.interpolate(
            images_flat,
            size=encoder_size,
            mode='bilinear',
            align_corners=False,
        ).reshape(batch_size, num_views, num_channels, encoder_size[0], encoder_size[1])
        mask_encoder = F.interpolate(mask, size=encoder_size, mode='nearest')
        encoder_input = torch.cat([
            images_encoder * mask_encoder.unsqueeze(1).expand(-1, num_views, -1, -1, -1),
            mask_encoder.unsqueeze(1).expand(-1, num_views, -1, -1, -1),
        ], dim=2)
        feats = self.encoder(encoder_input)

        feat_batch, feat_views, feat_channels, feat_height, feat_width = feats.shape
        images_canonical = F.interpolate(
            images_flat,
            size=(feat_height, feat_width),
            mode='bilinear',
            align_corners=False,
        ).reshape(batch_size, num_views, num_channels, feat_height, feat_width)
        mask_canonical = F.interpolate(mask, size=(feat_height, feat_width), mode='nearest')
        normal_canonical = F.normalize(
            F.interpolate(normal, size=(feat_height, feat_width), mode='bilinear', align_corners=False),
            p=2,
            dim=1,
        )

        loss = images.new_tensor(0.0)
        for batch_idx in range(feat_batch):
            mask_pixels = mask_canonical[batch_idx].reshape(-1, feat_height * feat_width).permute(1, 0)
            normal_pixels = normal_canonical[batch_idx].reshape(-1, feat_height * feat_width).permute(1, 0)
            valid_ids = torch.nonzero(mask_pixels.squeeze(1) > 0, as_tuple=False).squeeze(1)
            if valid_ids.numel() == 0:
                continue

            feature_tokens = feats[batch_idx].reshape(feat_views, feat_channels, feat_height * feat_width).permute(2, 0, 1)[valid_ids]
            normal_tokens = normal_pixels[valid_ids]
            observation_tokens = images_canonical[batch_idx].reshape(num_views, num_channels, feat_height * feat_width).permute(2, 0, 1)[valid_ids]

            aggregated = self.aggregation(torch.cat([observation_tokens, feature_tokens], dim=2))
            predicted = F.normalize(self.prediction(aggregated)[:, :3], dim=1, p=2)
            loss = loss + self.criterion_l2(predicted, normal_tokens) / valid_ids.numel()

        output_height, output_width = decoder_size
        images_high = F.interpolate(
            images_flat,
            size=decoder_size,
            mode='bilinear',
            align_corners=False,
        ).reshape(batch_size, num_views, num_channels, output_height, output_width)
        mask_high = F.interpolate(mask, size=decoder_size, mode='nearest')
        normal_high = F.normalize(
            F.interpolate(normal, size=decoder_size, mode='bilinear', align_corners=False),
            p=2,
            dim=1,
        )

        nout = images.new_zeros((batch_size, output_height * output_width, 3))
        mae = images.new_tensor(0.0)

        if mode == 'train':
            mae_sum = images.new_tensor(0.0)
            mae_count = 0
            for batch_idx in range(batch_size):
                mask_pixels = mask_high[batch_idx].reshape(-1, output_height * output_width).permute(1, 0)
                normal_pixels = normal_high[batch_idx].reshape(-1, output_height * output_width).permute(1, 0)
                valid_ids = torch.nonzero(mask_pixels.squeeze(1) > 0, as_tuple=False).squeeze(1)
                if valid_ids.numel() == 0:
                    continue
                if valid_ids.numel() > self.num_samples:
                    keep = torch.randperm(valid_ids.numel(), device=valid_ids.device)[:self.num_samples]
                    valid_ids = valid_ids[keep]

                coords = ind2coords((output_height, output_width), valid_ids).to(self.device)
                tokens = self._build_tokens(feats[batch_idx], images_high[batch_idx], coords, valid_ids)
                aggregated = self.aggregation(tokens)
                predicted = F.normalize(self.prediction(aggregated)[:, :3], dim=1, p=2)

                target_normals = normal_pixels[valid_ids]
                nout[batch_idx, valid_ids] = predicted
                loss = loss + self.criterion_l2(predicted, target_normals) / valid_ids.numel()
                mae_sum = mae_sum + angular_error(predicted, target_normals).sum()
                mae_count += valid_ids.numel()

            mae = mae_sum / max(mae_count, 1)
        else:
            for batch_idx in range(batch_size):
                mask_pixels = mask_high[batch_idx].reshape(-1, output_height * output_width).permute(1, 0)
                valid_ids = torch.nonzero(mask_pixels.squeeze(1) > 0, as_tuple=False).squeeze(1)
                for valid_chunk in torch.split(valid_ids, self.max_num_samples):
                    if valid_chunk.numel() == 0:
                        continue
                    coords = ind2coords((output_height, output_width), valid_chunk).to(self.device)
                    tokens = self._build_tokens(feats[batch_idx], images_high[batch_idx], coords, valid_chunk)
                    aggregated = self.aggregation(tokens)
                    predicted = F.normalize(self.prediction(aggregated)[:, :3], dim=1, p=2)
                    nout[batch_idx, valid_chunk] = predicted

        nout_high = nout.permute(0, 2, 1).reshape(batch_size, 3, output_height, output_width)
        normal_map = 0.5 * (nout_high + 1) * mask_high
        dot = torch.sum(nout_high * normal_high, dim=1, keepdim=True).clamp(-1.0 + 1.0e-12, 1.0 - 1.0e-12)
        error_map = torch.rad2deg(torch.acos(dot)) * mask_high

        if mode == 'test':
            mae = angular_error(nout_high, normal_high, mask_high)

        return loss, mae, normal_map, error_map

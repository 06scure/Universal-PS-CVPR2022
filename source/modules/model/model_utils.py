import os
import torch
from datetime import datetime

def angular_error(x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        dot = torch.sum(x1 * x2 * mask, dim=1, keepdim=True)
    else:
        dot = torch.sum(x1 * x2, dim=1, keepdim=True)

    eps = 1e-12
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    error = torch.rad2deg(torch.acos(dot)).abs()

    if mask is not None:
        return torch.sum(error * mask) / (torch.sum(mask) + 1e-8)
    return error

def build_optimizer_and_scheduler(net, args):
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    if args.lr_scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=30,
            eta_min=0,
            last_epoch=-1,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.8,
        )

    return optimizer, scheduler


def load_checkpoint(net, device, checkpoint_path=None, scheduler=None, optimizer=None):
    if checkpoint_path is None:
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    required_keys = ('state_dict', 'optimizer', 'scheduler', 'epoch', 'args')
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(f'Checkpoint is missing required keys: {missing_keys}')

    net.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = int(checkpoint['epoch'])
    return start_epoch


def save_checkpoint(net, optimizer, scheduler, epoch, args):
    checkpoint_dir = os.path.join(args.outdir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(checkpoint_dir, f'{timestamp}_module.pt')

    torch.save({
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'args': vars(args),
    }, checkpoint_path)
    return checkpoint_path
import os
import torch
import numpy as np
from typing import Tuple

def loadmodel(model, filename):
    params = torch.load('%s' % filename)
    model.load_state_dict(params,strict=False)
    print('Load %s' % filename)
    return model

def loadoptimizer(optimizer, filename):
    params = torch.load('%s' % filename)
    optimizer.load_state_dict(params)
    print('Load %s' % filename)
    return optimizer

def loadscheduler(scheduler, filename):
    params = torch.load('%s' % filename)
    scheduler.load_state_dict(params)
    print('Load %s' % filename)
    return scheduler

def savemodel(model, filename):
    print('Save %s' % filename)
    torch.save(model.state_dict(), filename)

def saveoptimizer(optimizer, filename):
    print('Save %s' % filename)
    torch.save(optimizer.state_dict(), filename)

def savescheduler(scheduler, filename):
    print('Save %s' % filename)
    torch.save(scheduler.state_dict(), filename)

def optimizer_setup_Adam(net, lr = 0.0001, init=True, stype='step'):
    print(f'optimizer (Adam) lr={lr}')
    if init==True:
        net.init_weights()
    net = torch.nn.DataParallel(net)
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    optimizer = torch.optim.Adam(optim_params, betas=(0.9, 0.999), weight_decay=0)
    if stype == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0, last_epoch=-1)
        print('cosine aneealing learning late scheduler')
    if stype == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
        print('step late scheduler x0.8 decay')
    return net, optimizer, scheduler

def optimizer_setup_SGD(net, lr = 0.01, momentum= 0.9, init=True):
    print(f'optimizer (SGD with momentum) lr={lr}')
    if init==True:
        net.init_weights()
    net = torch.nn.DataParallel(net)
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    return net, torch.optim.SGD(optim_params, momentum=momentum, weight_decay=1e-4, nesterov=True)

def optimizer_setup_AdamW(net, lr = 0.001, init=True, stype='step'):
    print(f'optimizer (AdamW) lr={lr}')
    if init==True:
        net.init_weights()
    net = torch.nn.DataParallel(net)
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    optimizer = torch.optim.AdamW(optim_params, betas=(0.9, 0.999), weight_decay=0.01)
    if stype == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0, last_epoch=-1)
        print('cosine aneealing learning late scheduler')
    if stype == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
        print('step late scheduler x0.8 decay')
    return net, optimizer, scheduler

def mode_change(net, Training):
    if Training == True:
        for param in net.parameters():
            param.requires_grad = True
        net.train()
    if Training == False:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records,
            'args': args}
    torch.save(state, os.path.join(save_path, 'checkp_%d.pth.tar' % (epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_%d_rec.pth.tar' % (epoch)))

def masking(img, mask):
    # img [B, C, H, W]
    # mask [B, 1, H, W] [0,1]
    img_masked = img * mask.expand((-1, img.shape[1], -1, -1))
    return img_masked

def print_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print('# parameters: %d' % params)


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

def get_error_map(pred_nml, gt_nml, mask):
    dot = torch.sum(pred_nml * gt_nml, dim=1, keepdim=True).clamp(-1.0 + 1.0e-12, 1.0 - 1.0e-12)
    error_map = (torch.acos(dot) * 180.0 / torch.pi) * mask
    return error_map

def get_normal_map(pred_nml, mask):
    normal_map = torch.Tensor(0.5 * (pred_nml + 1) * mask)
    return normal_map

def write_errors(filepath, error, trainid, numimg, objname = []):
    from datetime import datetime
    dt_now = datetime.now()
    print(filepath)

    if len(objname) > 0:
        with open(filepath, 'a') as f:
            f.write('%s %03d %s %02d %.2f\n' % (dt_now, numimg, objname, trainid, error))
    else:
        with open(filepath, 'a') as f:
            f.write('%s %03d %02d %.2f\n' % (dt_now, numimg, trainid, error))


def gradient_difference_loss(pred:torch.Tensor, 
                             target:torch.Tensor, 
                             mask:torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred: [B, 3, H, W] 预测法线（单位向量）
        target: [B, 3, H, W] GT 法线（单位向量）
        mask: [B, 1, H, W] 有效区域掩码
    Returns:
        g(n) = ||(n[x+1]-n[x-1])/2||_1 + ||(n[y+1]-n[y-1])/2||_1
    """
    if pred.shape[-2] < 3 or pred.shape[-1] < 3:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_dx = (pred[:, :, 1:-1, 2:] - pred[:, :, 1:-1, :-2]) * 0.5
    pred_dy = (pred[:, :, 2:, 1:-1] - pred[:, :, :-2, 1:-1]) * 0.5
    target_dx = (target[:, :, 1:-1, 2:] - target[:, :, 1:-1, :-2]) * 0.5
    target_dy = (target[:, :, 2:, 1:-1] - target[:, :, :-2, 1:-1]) * 0.5

    pred_grad = pred_dx.abs().sum(dim=1, keepdim=True) + pred_dy.abs().sum(dim=1, keepdim=True)
    target_grad = target_dx.abs().sum(dim=1, keepdim=True) + target_dy.abs().sum(dim=1, keepdim=True)

    grad_mask = mask[:, :, 1:-1, 1:-1]
    valid = grad_mask.sum()
    if valid.item() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    grad_error = torch.norm(pred_grad - target_grad, p=2, dim=1, keepdim=True)
    return (grad_error * grad_mask).sum() / (valid + 1e-8)


def build_optimizer_scheduler(net:torch.nn.Module, 
                              lr = 0.0001,
                              optimizer_type = 'AdamW',
                              scheduler_type = 'step') -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(optim_params, betas=(0.9, 0.999), weight_decay=0)
    elif optimizer_type == 'SGD':
        optimizer =  torch.optim.SGD(optim_params, momentum=0.9, weight_decay=1e-4, nesterov=True)
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(optim_params, betas=(0.9, 0.999), weight_decay=0.01)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    if scheduler_type == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0, last_epoch=-1)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return optimizer, scheduler
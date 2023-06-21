import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import losses
from skimage.metrics import structural_similarity as ssim

def bce_loss(pred, target):
    loss_function = nn.BCEWithLogitsLoss()
    loss = loss_function(pred, target)
    return loss

def ce_loss(pred, target):
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(pred, target)
    return loss


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def focal_loss(inputs, targets, alpha=.25, gamma=2) : 
    inputs = F.sigmoid(inputs)       
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1-BCE_EXP)**gamma * BCE
    return loss 

def calc_loss(pred, target, bce_weight = 0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def smp_dice_loss(pred, target):
    loss_function = getattr(losses, "DiceLoss")(mode="multilabel")
    loss = loss_function(pred, target)
    return loss

def smp_focal_loss(pred, target):
    loss_function = getattr(losses, "FocalLoss")(mode="multilabel")
    loss = loss_function(pred, target)
    return loss

def tversky_loss(pred, target):
    loss_function = getattr(losses, "TverskyLoss")(mode="multilabel", alpha=0.6, beta=0.4)
    loss = loss_function(pred, target)
    return loss
    
def mse_loss(pred, target):
    loss_function = nn.MSELoss()
    loss = loss_function(pred, target)
    return loss

def smp_jaccard_loss(pred, target):
    loss_function = getattr(losses, "JaccardLoss")(mode="multilabel")
    loss = loss_function(pred, target)
    return loss

def comb_loss(pred, target):
    bce = bce_loss(pred, target)
    dice = smp_dice_loss(pred, target)
    jaccard = smp_jaccard_loss(pred, target)
    loss = (0.1*bce) + (0.6*dice) + (0.3*jaccard)
    return loss

def ssim_loss(output, target):
    output_gray = F.rgb_to_grayscale(output).cpu()
    target_gray = F.rgb_to_grayscale(target).cpu()
    output_gray = output_gray.detach().numpy()
    target_gray = target_gray.detach().numpy()
    loss = []
    for i in range(16):
        ssim_value, diff = ssim(output_gray[i][0], target_gray[i][0],data_range=output_gray[i].max() - output_gray[i].min(), full=True)
        ssim_loss = 1 - ssim_value
        loss.append(ssim_loss)

    return sum(loss)/16

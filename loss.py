import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import losses

def bce_loss(pred, target):
    loss_function = nn.BCEWithLogitsLoss()
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

def mse_loss(pred, target):
    loss_function = nn.MSELoss()
    loss = loss_function(pred, target)
    return loss
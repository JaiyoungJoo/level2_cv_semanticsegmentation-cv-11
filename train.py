import os
import random
import datetime
import argparse
from importlib import import_module
import ssl
# external library
import numpy as np
from tqdm.auto import tqdm
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
# visualization
import wandb

# dataset
from dataset import XRayDataset
from dataset import get_transform
ssl._create_default_https_context = ssl._create_unverified_context

# for time check
import time
from pytz import timezone

# exp setting
BATCH_SIZE = 8
LR = 1e-4
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
my_table = wandb.Table(
    columns=["epoch"] + CLASSES)

def check_path(path):
    # 가중치 저장 경로 설정
    if not os.path.isdir(path):                                                           
        os.makedirs(path)

def make_dataset(debug="False"):
    # dataset load
    tf = A.Resize(512, 512)
    train_transform, val_transform = get_transform()
    if args.transform=='True':
        train_dataset = XRayDataset(is_train=True, transforms=train_transform)
        valid_dataset = XRayDataset(is_train=False, transforms=val_transform)
    else:
        train_dataset = XRayDataset(is_train=True, transforms=tf)
        valid_dataset = XRayDataset(is_train=False, transforms=tf)
    if debug=="True":
        train_subset_size = int(len(train_dataset) * 0.1)

        # Create a random train subset of the original dataset
        train_subset_indices = torch.randperm(len(train_dataset))[:train_subset_size]
        train_dataset = Subset(train_dataset, train_subset_indices)

        # Calculate the number of samples for the valid subset
        valid_subset_size = int(len(valid_dataset) * 0.1)

        # Create a random valid subset of the original dataset
        valid_subset_indices = torch.randperm(len(valid_dataset))[:valid_subset_size]
        valid_dataset = Subset(valid_dataset, valid_subset_indices)
        
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    return [train_loader, valid_loader]

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, args):
    
    output_path = os.path.join(args.save_dir, f"{args.model}_{args.encoder}_{args.loss}_{args.epochs}_step_1_acctest.pt")    #아래의 wandb쪽의 name과 동시 수정할것
    torch.save(model, output_path)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def wandb_config(args):
    wandb.init(config={'batch_size':BATCH_SIZE,
                    'learning_rate':LR,                 #차차 args.~~로 update할 것
                    'seed':args.seed,
                    'max_epoch':args.epochs},
            project='Segmentation',
            entity='aivengers_seg',
            name=f'{args.model}_{args.encoder}_{args.loss}_{args.epochs}_step_1_acctest'
            )

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()
    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    row = np.concatenate((np.array([epoch]), np.array(dices_per_class)))
    my_table.add_data(*row)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice

def train(model, data_loader, val_loader, criterion, optimizer, args):
    if args.wandb=="True":
        wandb_config(args)
    else:
        print("#########################################################")
        print("wandb not logging....")
        print("#########################################################")
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    seed = 0
    for epoch in range(args.epochs):
        if args.seed == 'up'and epoch % 5 == 0:
            seed += 1
            set_seed(seed)
            data_loader, valid_loader = make_dataset()


        model.train()
        step_count = 0
        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            # inference
            outputs = model(images)
            
            # loss 계산
            loss = criterion(outputs, masks)
            loss = loss / args.acc_steps # 경사를 경사 누적 스텝 수로 나눔
            loss.backward()
            step_count += 1
            if step_count % args.acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count = 0
            # 경사 업데이트 없이 스텝을 끝마치기 전에 경사 초기화
            if step_count != 0 and (step_count % args.acc_steps) != 0:
                optimizer.zero_grad()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{args.epochs}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
            train={'Loss':round(loss.item(),4)}
            if args.wandb=="True":
                wandb.log(train, step = epoch)
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.val_every == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {args.save_dir}")
                best_dice = dice
                save_model(model, args)

            val={'avg_dice':dice,
                 'best_dice':best_dice}
            
            if args.wandb=="True":
                wandb.log(val, step = epoch)

    wandb.log({"Table Name": my_table}, step=epoch) 

def main(args):
    # criterion = nn.BCEWithLogitsLoss()
    criterion = getattr(import_module("loss"), args.loss)
    if args.model == 'Pretrained_torchvision':
        model = getattr(import_module("model"), args.model)(model = args.model_path)
    elif args.model == 'Pretrained_smp':
        model = getattr(import_module("model"), args.model)(model = args.model_path)
    else : 
        model = getattr(import_module("model"), args.model)(encoder = args.encoder)

    optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)

    if args.seed != 'up':
        set_seed(args.seed)
        train_loader, valid_loader = make_dataset(args.debug)
    else:
        set_seed(0)
        train_loader, valid_loader = make_dataset()

    check_path(args.save_dir)
    train(model, train_loader, valid_loader, criterion, optimizer, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=21, help="random seed (default: 21)")
    parser.add_argument("--loss", type=str, default="bce_loss")
    parser.add_argument("--model", type=str, default="FCN")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--wandb", type=str, default="True")
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--save_dir", type=str, default="/opt/ml/input/weights/")
    parser.add_argument("--model_path", type=str, default="/opt/ml/input/weights/albumentation/FPN_densenet161_150.pt")
    parser.add_argument("--debug", type=str, default="False")
    parser.add_argument("--transform",type=str, default="False")
    parser.add_argument("--acc_steps", type=int, default="1")

    args = parser.parse_args()
    if args.model == 'Pretrained_torchvision' or 'Pretrained_smp':
        args.save_dir = os.path.join(args.save_dir, args.model_path.split('/')[-1].split('.')[0])
    else:
        args.save_dir = os.path.join(args.save_dir, args.model)

    start = time.time()
    print(args)
    main(args)
    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds=sec)
    result_list = str(datetime.timedelta(seconds=sec)).split(".")
    print(result_list[0])
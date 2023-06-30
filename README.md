# Hand Bone Image Segmentation
This project is the Naver Boost Camp CV11 team's submission code for Hand Bone Image Segmentation competition.  
Given an image of hand bone, it is a matter of segmenting 29 bone parts.



# Team Members

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/hykhhijk">
            <img src="https://avatars.githubusercontent.com/u/58303938?v=4" alt="김용희 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/HipJaengYiCat">
          <img src="https://avatars.githubusercontent.com/u/78784633?v=4" alt="박승희 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/imsmile2000">
          <img src="https://avatars.githubusercontent.com/u/69185594?v=4" alt="이윤표 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/junha-lee">
          <img src="https://avatars.githubusercontent.com/u/44857783?v=4" alt="이준하 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/JaiyoungJoo">
          <img src="https://avatars.githubusercontent.com/u/103994779?v=4" alt="주재영 프로필" width=120 height=120 />
        </a>
      </td>
    </tr>
    <tr>
      <td align="center">
        <a href="https://github.com/hykhhijk">
          김용희
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/HipJaengYiCat">
          박승희
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/imsmile2000">
          이윤표
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/junha-lee">
          이준하
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/JaiyoungJoo">
          주재영
        </a>
      </td>
    </tr>
  </table>
</div>

<br/>
<div id="5"></div>
 
# Environment
- OS : Linux Ubuntu 18.04.5
- GPU : Tesla V100 (32GB)


# Folder Structure
```bash
├─eda
├─ensemble
├─mmdetection
├─mmdetection3
├─UniverseNet
├─yolov8
├─multilabel_kfold.py
└─streamlit
```
<br></br>

# Usage

## Install Requirements

- `pip install -r requirements.txt`


<br></br>

## train.sh
1. Move the path to the tools folder where the train.sh file is located

2. Write python3 train.py command in train.sh file
    ```bash
    python3 train.py --model FCN --loss bce_loss --epochs 100
    ```
    All arguments for config training
   ```bash
    python3 train.py [-h] [--seed SEED] [--loss LOSS] [--model MODEL] [--epochs EPOCHS] [--val_every VAL_EVERY] [--train_batch TRAIN_BATCH] [--train_workers TRAIN_WORKERS] [--wandb WANDB] [--encoder ENCODER]
                [--save_dir SAVE_DIR] [--model_path MODEL_PATH] [--debug DEBUG] [--transform TRANSFORM] [--acc_steps ACC_STEPS] [--dataclean DATACLEAN]
    ```

4. Run
    ```bash
    nohup sh train.sh
    ```


# Result

## Loss experiment result
**Metric** : Dice coefficient

| **model** | **encoder**   | **loss**                                       | **wandb**  | **public** |
|-------|-----------|--------------------------------------------|--------|--------|
| MAnet | resnet101 | bce                                        | 0.9516 | 0.9456 |
| MAnet | resnet101 | smp_dice                                   | 0.9513 | 0.9489 |
| MAnet | resnet101 | calc                                       | 0.9523 | 0.9484 |
| MAnet | resnet101 | smp_focal                                  | 0.9506 | 0.9452 |
| MAnet | resnet101 | bce_with_logit                             | 0.9492 |        |
| MAnet | resnet101 | jaccard                                    | 0.9538 | 0.9496 |
| MAnet | resnet101 | tversky                                    | 0.9524 | 0.9490 |
| MAnet | resnet101 | comb(bce: 0.5, dice: 0.5)                  | 0.9526 | 0.9500 |
| MAnet | resnet101 | comb1(bce: 0.33, dice:0.33, jaccard: 0.33) | 0.9524 | 0.9490 |
| MAnet | resnet101 | comb2(bce: 1, dice: 3, jaccard: 6)         | 0.9528 | 0.9500 |
| MAnet | resnet101 | comb3(bce: 0.1, dice:0.6, jaccard: 0.3)    | 0.9531 | 0.9503 |  

We use comb3 as our loss. You can use it by set argument when training "--loss comb_loss"

      
## Augmentation experiment result
**Metric** : Dice coefficient

| **augmentation**                     | **wandb**      | **리더보드**                          |
|--------------------------------------|---------------------------|---------------------------------------|
| ElasticTransform(300)                | 0.9527                    | 0.9507                                |
| Rotate(limit=45)                     | 0.9524                    | 0.9502                                |
| RandomContrast(limit=[0,0.5], p=1)   | 0.9617                    | 0.9497 (inference transform: 0.9500)  |
| RandomContrast (limit=0.2, p=0.5)    | 0.9515                    | 0.9484                                |
| RandomContrast(limit=[0,0.5], p=0.5) | 0.9493                    | 0.9471                                |
| Normalize                            | 0.9486                    | 0.9465                                |
| ElasticTransform(400)                | 0.9524                    | 0.9450                                |
| Crop                                 | 0.9479                    |                                       |
| CenterCrop                           | 0.9343                    | 0.9243 (inference transform: 0.2927)  |
| Equalize and remove black (200)      | 0.6426(100) →0.7891 (200) |                                       |  

We use ElasticTransform(300), Rotate(limit=45) and RandomContrast(limit=[0,0.5], p=1) as our final augmentation.  



## TTA experiment result
**Metric** : Dice coefficient

|                                 **TTA**                               |     **리더보드**    |
|:-----------------------------------------------------------------:|:------------------:|
|                        TTA   적용 안했을 때                       |        0.9710      |
|                           HorizontalFlip                          |        0.9710      |
|            HorizontalFlip,     Multiply([0.9,1,1.1,1.2])           |        0.9717      |
|     HorizontalFlip,     Multiply([0.9,1,1.1,1.2]),     Rotate 90    |        0.9585      |

We use HorizontalFlip,     Multiply([0.9,1,1.1,1.2]) as our final TTA combination  


## Final Solution
**Metric** : Dice coefficient
![image](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-11/assets/58303938/89b2b744-3c2e-4bcf-aba1-d1bd8d95d123)

- Final submission : Public : 0.9743(2nd) / Private : 0.9749(2nd)

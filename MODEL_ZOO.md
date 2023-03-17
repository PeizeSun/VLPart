# VLPart model zoo

This file documents a collection of models reported in our paper.
The training time was measured on with 8 NVIDIA V100 GPUs & NVLink.

#### How to Read the Tables

The "Name" column contains a link to the config file. 

To train a model, run 

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml
``` 

To evaluate a model with a trained/ pretrained model, run 

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
``` 

Before training, make sure [Preparing Datasets](datasets) and [Preparing Models](models) are well-prepared.


### Cross-dataset part segmentation on PartImageNet

| Config              | All (40) | quadruped: head | quadruped: head | quadruped: head | quadruped: head | Training time |Download|
|---------------------|:--------:|:---------------:|:-------:|:-----------:|:-------:|:--------:|:------:|
| [pascal_part]()     |          |       |         |             |         |
| [+ IN-S11 label]()  |          
| [+ Parsed IN-S11]() |

| Config              | All (40) | quadruped: head | quadruped: head | quadruped: head | quadruped: head | Training time |Download|
|---------------------|:--------:|:---------------:|:-------:|:-----------:|:-------:|:--------:|:------:|
| [pascal_part]()     |          |       |         |             |         |
| [+ LVIS_PACO]()     |          
| [+ Parsed IN-S11]() |
<br>

### Cross-category part segmentation within Pascal Part

| Config               | All (93) | Base (57) | Novel (36) | dog:head | dog: torso | dog: leg | dog: paw | dog: tail | Training time |Download|
|----------------------|:--------:|:---------:|:----------:|:--------:|:----------:|:--------:|:--------:|:---------:|:--------:|:------:|
| [pascal_part_base]() |          |           |            |          |            |
| [+ VOC object]()     |          
| [+ IN-S20 label]()   |
| [+ Parsed IN-S20]()  |
<br>

### Open-vocabulary object detection and part segmentation

R50 Mask R-CNN:

| Config                   | COCO AP/AP50 | LVIS AP/APr | PartImageNet AP/AP50 |  Pascal Part AP/AP50 |PACO AP | Training time |Download|
|--------------------------|:------------:|:-----------:|:-------:|:--------:|:------:|:--------:|:------:|
| [joint]()                |  39.2/61.3   |  28.1/20.8  | 29.7 | 19.4/42.7 |  10.8  |
| [joint+IN-L]()           |          
| [joint+IN-L+Parsed IN]() |


| Name               | COCO AP/AP50 | LVIS AP/APr | PartImageNet AP/AP50 |  Pascal Part AP/AP50 |PACO AP |
|--------------------|:------------:|:-----------:|:-------:|:--------:|:------:|
| Dataset-specific   |  39.2/61.3   |  28.1/20.8  | 29.7 | 19.4/42.7 |  10.8  |
| Config             |          
| Training Time      |
| Download           |

```
# before training joint+IN-L+Parsed IN, generate Parsed IN by:
python train_net.py --config-file  --eval-only 
python train_net.py --config-file  --eval-only 
```


SwinBase Cascade Mask R-CNN:

| Config                   | COCO AP/AP50 | LVIS AP/APr | PartImageNet AP/AP50 |  Pascal Part AP/AP50 |PACO AP | Training time |Download|
|--------------------------|:------------:|:-----------:|:-------:|:--------:|:------:|:--------:|:------:|
| [joint]()                |  39.2/61.3   |  28.1/20.8  | 29.7 | 19.4/42.7 |  10.8  |
| [joint+IN-L]()           |          
| [joint+IN-L+Parsed In]() |

| Name              | COCO AP/AP50 | LVIS AP/APr | PartImageNet AP/AP50 |  Pascal Part AP/AP50 |PACO AP |
|-------------------|:------------:|:-----------:|:-------:|:--------:|:------:|
| Dataset-specific  |  39.2/61.3   |  28.1/20.8  | 29.7 | 19.4/42.7 |  10.8  |
| Config            |          
| Training Time     |
| Download          |

```
# before training joint+IN-L+Parsed IN, generate Parsed IN by:
python train_net.py --config-file  --eval-only 
python train_net.py --config-file  --eval-only 
```

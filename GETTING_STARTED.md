## Getting Started

This document provides a brief intro of the usage of ClusterDet.



### Inference Demo on Image

Pick a model and its config file from [model zoo](MODEL_ZOO.md), run:
```
python demo/demo.py --config-file configs/coco/faster_rcnn_r50_coco.yaml \
  --input input1.jpg input2.jpg \
  --output output_image \ 
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```

For edge map demo, run:
```
python demo/demo.py --config-file configs/proposal/recall_edge2box_coco_lvis.yaml \
  --input datasets/coco/val2017/* \
  --opts MODEL.GROUP_HEADS.SIZE_DIVISIBILITY 0 VIS_PERIOD 1
```
The visualiztion will be saved in the folder `output_edge_map`.



### Training & Evaluation in Command Line

We provide `train_net.py` to train all the configs provided.

```
python train_net.py --num-gpus 8 \
  --config-file configs/coco/faster_rcnn_r50_coco.yaml
```

The configs are made for 8-GPU training.
To train on 1 GPU, you need to figure out learning rate, batch size and training iterations by yourself, 
as it has not been checked to reproduce performance.
```
python train_net.py \
  --config-file configs/coco/faster_rcnn_r50_coco.yaml --num-gpus 1 \
  SOLVER.IMS_PER_BATCH REASONABLE_VALUE SOLVER.BASE_LR REASONABLE_VALUE SOLVER.STEPS "(REASONABLE_VALUE,)"
```

To evaluate a model's performance, use:
```
python train_net.py --num-gpus 8 \
  --config-file configs/coco/faster_rcnn_r50_coco.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

More detailed command lines are in [Model Zoo](./MODEL_ZOO.md).

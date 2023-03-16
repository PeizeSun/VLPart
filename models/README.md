# Prepare models for ClusterDet

### HED Pretrained Models
Download [hed_checkpoint.pt](https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip) to `DETIC_ROOT/models/`.
```
cd $ClusterDet_ROOT/models/
wget https://cseweb.ucsd.edu/~weijian/static/datasets/hed/hed-data.tar
tar xvf ./hed-data.tar
mv data/hed_checkpoint.pt .
```
convert it to d2 format:
```
cd $ClusterDet_ROOT
python tools/convert-pt-to-detectron2.py --pt_path models/hed_checkpoint.pt --output_path models/hed_checkpoint.pth
```


###  DenseCL Pretrained models:

Download [densecl_r50_imagenet_200ep.pth](https://cloudstor.aarnet.edu.au/plus/s/hdAg5RYm8NNM2QP/download) and [densecl_r101_imagenet_200ep.pth](https://cloudstor.aarnet.edu.au/plus/s/4sugyvuBOiMXXnC/download) from [DenseCL](https://github.com/WXinlong/DenseCL) repo:
```
$ClusterDet_ROOT/models/
    densecl_r50_imagenet_200ep.pth
    densecl_r101_imagenet_200ep.pth
```
convert them to d2 format:
```
cd $ClusterDet_ROOT
python tools/convert-pretrain-to-detectron2.py models/densecl_r50_imagenet_200ep.pth models/densecl_r50_imagenet_200ep.pkl
python tools/convert-pretrain-to-detectron2.py models/densecl_r101_imagenet_200ep.pth models/densecl_r101_imagenet_200ep.pkl
```

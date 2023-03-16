# Prepare datasets for ClusterDet

Before preparing datasets, please make sure the models are prepared.

Download the datasets from the official websites and place or sim-link them under `$Detic_ROOT/datasets/`. 

```
$Detic_ROOT/datasets/
    metadata/
    coco/
    lvis/
    partimagenet/
    pascal_part/
    paco/
    objects365/
    VOC2007/
    cc3m/
    imagenet21k/
```
`metadata/` is our preprocessed meta-data (included in the repo). See the below [section](#Metadata) for details.



Please follow the following instruction to pre-process individual datasets.


### COCO and LVIS

```
$ClusterDet_ROOT/datasets/
    coco/
        train2017/
        val2017/
        annotations/
            captions_train2017.json
            instances_train2017.json 
            instances_val2017.json
    lvis/
        lvis_v1_train.json
        lvis_v1_val.json
        lvis_v1_minival_inserted_image_name.json
```
Download lvis_v1_minival_inserted_image_name.json from
```
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/coco/annotations/lvis_v1_minival_inserted_image_name.json
```

### Zero-shot COCO and LVIS

```
cd $ClusterDet_ROOT/
python tools/lvis_remove_rare.py --ann datasets/lvis/lvis_v1_train.json
python tools/lvis_get_cat_info.py --ann datasets/lvis/lvis_v1_train.json

python tools/coco_split_ovrcnn.py
python tools/lvis_get_cat_info.py --ann datasets/coco/zero-shot/instances_train2017_seen_2_oriorder.json

```

### PartImageNet

The PartImageNet folder should look like:
```
$ClusterDet_ROOT/datasets/
    partimagenet/
        train/
            n01440764
            n01443537
            ...            
        val/
            n01484850
            n01614925
            ...
        train.json
        val.json
```

convert them into coco annotation format.

~~~
cd $ClusterDet_ROOT/
python tools/partimagenet_format_json.py --old_path datasets/partimagenet/train.json --new_path datasets/partimagenet/train_format.json
python tools/partimagenet_format_json.py --old_path datasets/partimagenet/val.json --new_path datasets/partimagenet/val_format.json
~~~


### PASCAL Part

Download pascal_part annotations and images from
```
wget http://roozbehm.info/pascal-parts/trainval.tar.gz
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
```

The PASCAL Part folder should look like:
```
$ClusterDet_ROOT/datasets/
    pascal_part/
        Annotations_Part/
            2008_000002.mat
            2008_000003.mat
            ...
            2010_006086.mat
        VOCdevkit/
            VOC2010/
```

convert them into coco annotation format.

~~~
cd $ClusterDet_ROOT/
python tools/pascal_part_mat2json.py
python tools/pascal_part_mat2json.py --split train.txt --ann_out datasets/pascal_part/train.json
python tools/pascal_part_mat2json.py --only_base --split train.txt --ann_out datasets/pascal_part/train_base.json
python tools/pascal_part_mat2json.py --only_dog --split train.txt --ann_out datasets/pascal_part/train_dog.json

~~~

### PACO

Download paco annotations and images according to [paco](https://github.com/facebookresearch/paco).

The PACO folder should look like:
```
$ClusterDet_ROOT/datasets/
    paco/
        annotations/
            paco_ego4d_v1_test.json
            paco_ego4d_v1_train.json
            paco_ego4d_v1_val.json
            paco_lvis_v1_test.json
            paco_lvis_v1_train.json
            paco_lvis_v1_val.json
        images/
            000cd456-ff8d-499b-b0c1-4acead128a8b_000024.jpeg
            000cd456-ff8d-499b-b0c1-4acead128a8b_000681.jpeg
            ...    
```


### COCO Caption

We follow [Detic](https://github.com/facebookresearch/Detic) to pre-process COCO Caption annotations: 

```
cd $ClusterDet_ROOT/
python tools/get_coco_tags.py --cc_ann datasets/coco/annotations/captions_train2017.json --cat_path datasets/coco/annotations/instances_val2017.json --allcaps --convert_caption --out_path datasets/coco/zero-shot/captions_train2017_tags_allcaps.json 
```


The converted files should be like 

```
$ClusterDet_ROOT/datasets/
    coco/
        zero-shot/
            captions_train2017_tags_allcaps.json
```


### CC3M
We use [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md
) to download CC3M. 

The downloaded and unzipped files should be like
```
$ClusterDet_ROOT/datasets/
    cc3m/
        download/
            000000/
                0000000000.jpg
                0000000000.json
                0000000000.txt
                ...                             
            000000.parquet
            000000_stats.json
            000000.tar
            ...
            000331           
            000331.parquet
            000331_stats.json
            000331.tar           
```
Prepare json files for CC3M
```
 # the whole cc3m will be divided into 34 json files, 
 # each json includes 10 folders' images, 
 # each json has about 50k images (it should be 100k images, but many CC3M download urls are invalid ...)
 python tools/cc3m_prepare.py
```

### ImageNet21K

The ImageNet21K folder should look like:
```
$ClusterDet_ROOT/datasets/
    imagenet21k/
        ImageNet-21K/
            n01593028.tar
            n01593282.tar
            ...
```

We first unzip the overlapping classes of LVIS (we will directly work with the .tar file for the rest classes) and convert them into LVIS annotation format.

~~~
cd $ClusterDet_ROOT/
mkdir datasets/imagenet/annotations
mkdir datasets/imagenet/ImageNet-LVIS
python tools/unzip_imagenet_lvis.py --dst_path datasets/imagenet/ImageNet-LVIS
python tools/create_imagenetlvis_json.py --imagenet_path datasets/imagenet/ImageNet-LVIS --out_path datasets/imagenet/annotations/imagenet_lvis_image_info.json
~~~
This creates `datasets/imagenet/annotations/imagenet_lvis_image_info.json`.


# Attention !!! Below datasets are not ready...

### Open-world COCO
We follow [OLN](https://github.com/mcahny/object_localization_network) to create the open-world COCO split and pre-process the annotations:
```
cd $ClusterDet_ROOT/
python tools/get_coco_split_voc.py
python tools/get_lvis_cat_info.py --ann datasets/coco/openworld/instances_train2017_voc.json
```


### Zero-shot COCO and LVIS

We follow [OVR-CNN](https://github.com/alirezazareian/ovr-cnn) to create the zero-shot COCO and LVIS split: 

```
cd $ClusterDet_ROOT/
python tools/get_coco_split_ovrcnn.py
python tools/get_coco_zeroshot.py --data_path datasets/coco/zero-shot/instances_train2017_seen_2.json
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_train2017_seen_2.json
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_val2017_all_2.json
python tools/get_lvis_cat_info.py --ann datasets/coco/zero-shot/instances_train2017_seen_2_oriorder.json
python tools/get_lvis_cat_info.py --ann datasets/coco/zero-shot/instances_train2017_seen_2_del.json

python tools/remove_lvis_rare.py --ann datasets/lvis/lvis_v1_train.json
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train.json
```


### COCO Caption

We follow [Detic](https://github.com/facebookresearch/Detic) to pre-process COCO Caption annotations: 

```
cd $ClusterDet_ROOT/
python tools/get_coco_tags.py --cc_ann datasets/coco/annotations/captions_train2017.json --cat_path datasets/coco/annotations/instances_val2017.json --allcaps --convert_caption --out_path datasets/coco/zero-shot/captions_train2017_tags_allcaps.json 
python tools/get_VLDet_tags.py --cc_ann datasets/coco/annotations/captions_train2017.json --cat_path datasets/metadata/coco_nouns_4764.txt --allcaps --convert_caption --out_path datasets/coco/zero-shot/captions_train2017_nouns_4764tags_allcaps.json
```


The converted files should be like 

```
$ClusterDet_ROOT/datasets/
    coco/
        openworld/
            instances_train2017_voc_cat_info.json
            instances_train2017_voc.json
        zero-shot/
            captions_train2017_nouns_4764tags_allcaps.json
            captions_train2017_tags_allcaps.json
            instances_train2017_seen_2_oriorder_cat_info.json
            instances_train2017_seen_2_oriorder.json
            instances_train2017_seen_2.json
            instances_val2017_all_2_oriorder.json
            instances_val2017_all_2.json
    lvis/
        lvis_v1_train_cat_info.json
        lvis_v1_train_norare.json
```
### ADE20k-full

The ADE20k-full folder should look like:
```
$ClusterDet_ROOT/datasets/
    ADE20K_2021_17_01/
        images/            
        index_ade20k.pkl
        objects.txt
```

convert them into coco annotation format.

~~~
cd $ClusterDet_ROOT/
python tools/prepare_ade20k_full_entity_seg.py
~~~

### UVO

Pre-processing UVO is too complicated, so we suggest:

- download json annotations `UVO_frame_val.json` from [UVO google drive](https://drive.google.com/drive/folders/1dz2aSAy50tT95I3oWYjVsiJhDwdEE40s)
- download images from our prepared google drive [val_images.zip](https://drive.google.com/file/d/1qBcs6UcFUtBt_xFItZsr-bG5vJRJeiTl/view?usp=share_link) and unzip it to `collect_val_images`

The UVO folder should look like:
```
$ClusterDet_ROOT/datasets/
    UVO/
        collect_val_images/            
        UVO_frame_val.json
        val_images.zip
```


### Conceptual Caption

Download the dataset from [this](https://ai.google.com/research/ConceptualCaptions/download) page and place them as:
```
$ClusterDet_ROOT/datasets/
    cc3m/
        Train_GCC-training.tsv
```

Run the following command to download the images and convert the annotations to LVIS format (Note: download images takes long).

~~~
cd $ClusterDet_ROOT/
python tools/download_cc.py --ann datasets/cc3m/Train_GCC-training.tsv --save_image_path datasets/cc3m/training/ --out_path datasets/cc3m/train_image_info.json
python tools/get_cc_tags.py
~~~

This creates `datasets/cc3m/train_image_info_tags.json`.



### Metadata

```
$Detic_ROOT/datasets/
    metadata/
        coco_clip_a+cname.npy
        lvis_v1_clip_a+cname.npy
        imagenet_lvis_wnid.txt
```

`*_clip_a+cname.npy` is the pre-computed CLIP embeddings for each datasets.
They are created by (taking LVIS as an example)
~~~
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val.json --out_path metadata/lvis_v1_clip_a+cname.npy
~~~

`imagenet_lvis_wnid.txt` is the list of matched classes between ImageNet-21K and LVIS.

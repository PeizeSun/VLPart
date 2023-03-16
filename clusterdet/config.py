# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_clusterdet_config(cfg):
    """
    Add config for ClusterDet.
    """
    cfg.DINO = CN()
    cfg.DINO.MODEL_TYPE = 'dino_vits8'
    cfg.DINO.STRIDE = 4
    cfg.DINO.LAYER = 11
    cfg.DINO.THRESH = 0.05
    cfg.DINO.FACET = 'key'
    cfg.DINO.BUILD_BASEDATA = False
    cfg.DINO.BASEDATA_SAVE_DIR = 'output_basedata'
    cfg.DINO.BASEDATA_ANN_PATH = 'datasets/pascal_part/train_base_one.json'
    cfg.DINO.BASEDATA_IMS_PATH = 'datasets/pascal_part/VOCdevkit/VOC2010/JPEGImages/'
    cfg.DINO.PIXEL_NORM = True
    cfg.DINO.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.DINO.PIXEL_STD = [0.229, 0.224, 0.225]
    cfg.DINO.MIN_SIZE_TEST = 224

    cfg.MODEL.ROI_BOX_HEAD.PART_CLS_COST_WEIGHT = 2.0
    cfg.MODEL.ROI_BOX_HEAD.PART_LOC_COST_WEIGHT = 5.0
    cfg.MODEL.ROI_BOX_HEAD.PART_LOSS_TYPE = 'max_score'
    cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS_GROUP = [False]
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS_GROUP = [True]
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH_GROUP = ['datasets/metadata/lvis_v1_train_cat_info.json']
    cfg.MODEL.ROI_BOX_HEAD.OBJ2PART_MAPPING = 'datasets/metadata/pascal_obj2part_mapping.json'
    cfg.MODEL.ROI_BOX_HEAD.PART2PROXY_MAPPING = 'datasets/metadata/pascal_part2proxy_mapping.json'
    cfg.MODEL.ROI_BOX_HEAD.METAPART_WEIGHT_PATH = 'datasets/metadata/pascal_metapart_clip_a+cname.npy'
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (0.125,)
    cfg.MODEL.SUPERVISION = 'fully_supervised'
    cfg.MODEL.EVAL_ATTR = False
    cfg.MODEL.EVAL_PER = False
    cfg.MODEL.EVAL_PROPOSAL = False
    cfg.MODEL.PIXEL_NORM = True
    cfg.MODEL.CLIP_TYPE = 'RN50'
    cfg.MODEL.FREEZE_CLIP = True
    cfg.MODEL.FREEZE_CLIP_TEXT = True
    cfg.MODEL.FREEZE_CLIP_VISUAL = False
    cfg.MODEL.IS_GROUNDING = False
    cfg.MODEL.ROI_BOX_HEAD.RETURN_ALL_SCORE = False
    cfg.MODEL.ROI_BOX_HEAD.MULT_OBJECT_SCORE = False
    cfg.MODEL.ROI_BOX_HEAD.REFINE_BOX = False
    cfg.MODEL.ROI_BOX_HEAD.BOX_SCORE_REG_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.BOX_SCORE_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.USE_REFINEMENT_SCORE = False
    cfg.MODEL.ROI_BOX_HEAD.NOUNS_LOSS_WEIGHT = 0.01
    cfg.MODEL.ROI_BOX_HEAD.BOX2PART_LOSS_WEIGHT = 0.1
    cfg.MODEL.ROI_BOX_HEAD.PARSED_PART_LOSS_WEIGHT = 0.1
    cfg.MODEL.RPN.NORM = ''
    cfg.DATASETS.BACK_RATIO = 8
    cfg.INPUT.CROP.PRE_SIZE = [400, 500, 600]
    cfg.MODEL.PROPOSAL_SELFSUP_GENERATOR = CN()
    cfg.MODEL.PROPOSAL_SELFSUP_GENERATOR.NAME = 'RPNSelfsup'

    # CLIP_FPN
    cfg.MODEL.CLIP_FPN = CN()
    cfg.MODEL.CLIP_FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.CLIP_FPN.STRIDES = [4, 8, 16, 32]
    cfg.MODEL.CLIP_FPN.IN_CHANNELS = [256, 512, 1024, 2048]
    cfg.MODEL.CLIP_FPN.OUT_CHANNELS = 256
    cfg.MODEL.CLIP_FPN.NORM = ""
    cfg.MODEL.CLIP_FPN.FUSE_TYPE = "sum"

    cfg.MODEL.LOAD_PROPOSALS_TEST = False
    cfg.MODEL.PSEUDO_VIS_ON = 0
    cfg.MODEL.RPN_VIS_ON = 0
    cfg.MODEL.MASK_EVAL_ON = False

    cfg.MODEL.CONDINST = CN()
    cfg.MODEL.CONDINST.SIZES_OF_INTEREST = [64, 128, 256, 512]
    cfg.MODEL.CONDINST.SIZES_OF_INTEREST_MIN_LEVEL = 3
    cfg.MODEL.CONDINST.SIZES_OF_INTEREST_MAX_LEVEL = 6

    # the downsampling ratio of the final instance masks to the input image
    cfg.MODEL.CONDINST.MASK_OUT_STRIDE = 4
    cfg.MODEL.CONDINST.BOTTOM_PIXELS_REMOVED = -1

    # if not -1, we only compute the mask loss for MAX_PROPOSALS random proposals PER GPU
    cfg.MODEL.CONDINST.MAX_PROPOSALS = -1
    # if not -1, we only compute the mask loss for top `TOPK_PROPOSALS_PER_IM` proposals
    # PER IMAGE in terms of their detection scores
    cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM = -1

    cfg.MODEL.CONDINST.MASK_FEAT = CN()
    cfg.MODEL.CONDINST.MASK_FEAT.OUT_CHANNELS = 8
    cfg.MODEL.CONDINST.MASK_FEAT.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.CONDINST.MASK_FEAT.CHANNELS = 128
    cfg.MODEL.CONDINST.MASK_FEAT.NORM = "BN"
    cfg.MODEL.CONDINST.MASK_FEAT.NUM_CONVS = 4

    cfg.MODEL.CONDINST.MASK_HEAD = CN()
    cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS = 16
    cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 3
    cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS = False

    cfg.MODEL.BOXINST = CN()
    cfg.MODEL.BOXINST.ENABLED = False
    cfg.MODEL.BOXINST.PAIRWISE = CN()
    cfg.MODEL.BOXINST.PAIRWISE.SIZE = 3
    cfg.MODEL.BOXINST.PAIRWISE.DILATION = 2
    cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3
    cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000

    # Open-world
    cfg.OPENWORLD = CN()
    cfg.OPENWORLD.EVALUATE_CLASS_AGNOSTIC = True
    cfg.OPENWORLD.EVALUATE_CLASS_AWARE = False
    cfg.OPENWORLD.EVALUATE_EXCLUDE_KNOWN = True
    cfg.OPENWORLD.OLN_OFFICIAL = False
    cfg.MODEL.RPN.OPENWORLD_IOU_THRESHOLDS = [0.1, 0.3]
    cfg.MODEL.RPN.OPENWORLD_IOU_LABELS = [0, -1, 1]
    cfg.MODEL.RPN.GROUP_BOX_ALL_LOSS = False
    cfg.MODEL.RPN.GROUP_BOX_LOSS = 0.1
    cfg.MODEL.ROI_BOX_HEAD.CLASS_AGNOSTIC = False
    cfg.MODEL.ROI_BOX_HEAD.CLASS_AGNOSTIC_REG_LOSS = True
    cfg.MODEL.ROI_BOX_HEAD.GROUP_BOX_SCORE_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.GROUP_BOX_REG_LOSS = False

    # Group head configs
    cfg.WITH_PIXEL_AFFINITY = False  # Turn on supervision from pixel_affinity
    cfg.MODEL.SEPERATE_BACKBONE = False
    cfg.MODEL.GROUP_GENERATOR = CN()
    cfg.MODEL.GROUP_GENERATOR.NAME = "GroupRPN"
    cfg.MODEL.GROUP_HEADS = CN()
    cfg.MODEL.GROUP_HEADS.NAME = "FCNHead"
    cfg.MODEL.GROUP_HEADS.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.GROUP_HEADS.OUT_CHANNELS = 8
    cfg.MODEL.GROUP_HEADS.NUM_CHANNELS = 256
    cfg.MODEL.GROUP_HEADS.NUM_CONVS = 4
    cfg.MODEL.GROUP_HEADS.POS_WEIGHT = 0.05

    cfg.MODEL.GROUP_HEADS.NUM_CLASSES = 80
    cfg.MODEL.GROUP_HEADS.SIZE_DIVISIBILITY = 0
    cfg.MODEL.GROUP_HEADS.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.GROUP_HEADS.EDGE_THRESHOLDS = [0.5, 0.7, 0.9]
    cfg.MODEL.GROUP_HEADS.EDGE_LEVELS = [-1]
    cfg.MODEL.GROUP_HEADS.PROPOSAL_NUMS = [2, 4, 8, 16, 32, 64, 128]
    cfg.MODEL.GROUP_HEADS.SCALE_FACTORS = [1.0, 0.5, 0.25]
    cfg.MODEL.GROUP_HEADS.BOX_AREA_THRESH = 100.0
    cfg.MODEL.GROUP_HEADS.BOX_OUTPUT_STRIDE = 4
    cfg.MODEL.GROUP_HEADS.BOX_TO_IMAGE_RATIO_THRESH = 0.9
    cfg.MODEL.GROUP_HEADS.BOX_SCORE_THRESH = 0.1
    cfg.MODEL.GROUP_HEADS.BOX_IOU_THRESH = 0.5
    cfg.MODEL.GROUP_HEADS.IOU_BY_GT_THRESH = 0.5
    cfg.MODEL.GROUP_HEADS.BOX_TOPK_SCORE_THRESH = 0.7
    cfg.MODEL.GROUP_HEADS.BOX_PER_LEVEL = 100
    cfg.MODEL.GROUP_HEADS.BOX_TOPK = 3
    cfg.MODEL.GROUP_HEADS.MASK_TOPK = 3
    cfg.MODEL.GROUP_HEADS.HIERARCHY_GROUP = True
    cfg.MODEL.GROUP_HEADS.DOWN_SAMPLE_RATIO = 2
    cfg.MODEL.GROUP_HEADS.EVALUATE = False

    # clip head configs
    cfg.MODEL.RECOGNITION_HEADS = CN()
    cfg.MODEL.RECOGNITION_HEADS.CLIP_TYPE = 'ViT-B/32'
    cfg.MODEL.RECOGNITION_HEADS.CATEGORY_WEIGHT_PATH = 'datasets/metadata/coco_clip_a+cname.npy'
    cfg.MODEL.RECOGNITION_HEADS.CROP_BOX_SCALES = [1.0, ]
    cfg.MODEL.RECOGNITION_HEADS.IMAGE_SCALES = [0.5, 1.0, 2.0]
    cfg.MODEL.RECOGNITION_HEADS.CANONICAL_BOX_SIZE = 224
    cfg.MODEL.RECOGNITION_HEADS.SOFTMAX_T = 0.01
    cfg.MODEL.RECOGNITION_HEADS.BOX_SCORE_THRESH = 0.05
    cfg.MODEL.RECOGNITION_HEADS.BOX_IOU_THRESH = 0.5
    cfg.MODEL.RECOGNITION_HEADS.BOX_TOPK = 100
    cfg.MODEL.RECOGNITION_HEADS.NAME = 'Res5RecognitionROIHeads'
    cfg.MODEL.RECOGNITION_HEADS.OUT_CHANNELS = 1024
    cfg.MODEL.RECOGNITION_HEADS.IN_FEATURES = ['res4']
    cfg.MODEL.RECOGNITION_HEADS.POOLER_RESOLUTION = 14
    cfg.MODEL.RECOGNITION_HEADS.POOLER_SCALES = 16
    cfg.MODEL.RECOGNITION_HEADS.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.RECOGNITION_HEADS.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.RECOGNITION_HEADS.USE_SIGMOID_CE = False
    cfg.MODEL.RECOGNITION_HEADS.IMAGE_LABEL_LOSS = 'oicr'
    cfg.MODEL.RECOGNITION_HEADS.IMAGE_LOSS_WEIGHT = 1.0
    cfg.MODEL.RECOGNITION_HEADS.REGION_LOSS_WEIGHT = 3.0
    cfg.MODEL.RECOGNITION_HEADS.WITH_REFINEMENT_SCORE = False
    cfg.MODEL.RECOGNITION_HEADS.REFINEMENT_IOU = 0.5
    cfg.MODEL.RECOGNITION_HEADS.MINING_SCORE_THRESH = 0.2
    cfg.MODEL.RECOGNITION_HEADS.MINING_IOU_THRESH = 0.5
    cfg.MODEL.RECOGNITION_HEADS.PRIOR_PROB = 0.01
    cfg.MODEL.RECOGNITION_HEADS.NOVEL_MASK_PATH = 'datasets/metadata/lvis_novel_masks.npy'
    cfg.MODEL.RECOGNITION_HEADS.BACKGROUND_WEIGHT_PATH = 'datasets/metadata/bg_clip_RN50.npy'
    cfg.MODEL.RECOGNITION_HEADS.BASE_COEFFICIENT = 0.35
    cfg.MODEL.RECOGNITION_HEADS.NOVEL_COEFFICIENT = 0.65

    # ann generator
    cfg.MODEL.ANN_GENERATOR = False
    cfg.MODEL.ANN_CLASS_AGNOSTIC = False
    cfg.MODEL.CLASS_FILTER = False
    cfg.MODEL.ONE_BOX_PER_CLASS = False
    cfg.OUTPUT_ANN_DIR = 'datasets/grouping.json'

    # Open-vocabulary classifier
    cfg.WITH_IMAGE_LABELS = False  # Turn on co-training with classification data
    cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS = False  # Use fixed classifier for open-vocabulary detection
    cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS_GROUP = False
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'datasets/metadata/lvis_v1_clip_a+cname.npy'
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH_GROUP = ['datasets/metadata/lvis_v1_clip_a+cname.npy']
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH = 'datasets/metadata/lvis_v1_clip_a+cname.npy'
    cfg.MODEL.ROI_BOX_HEAD.PART_WEIGHT_PATH = 'datasets/metadata/paco_clip_RN50_a+cname.npy'
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM = 1024
    cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT = True
    cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP = 50.0
    cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS = False
    cfg.MODEL.ROI_BOX_HEAD.USE_BIAS = 0.0  # >= 0: not use


    cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False  # CenterNet2
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False  # Federated Loss
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = 'datasets/metadata/lvis_v1_train_cat_info.json'
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5

    cfg.MODEL.ROI_BOX_HEAD.FUSE_CLIP_SCORE = False

    # Classification data configs
    cfg.MODEL.ROI_BOX_CASCADE_HEAD.IMAGE_LABEL_LOSSES = ['max_size', 'max_size', 'max_size']
    cfg.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS = 'max_size'  # max, softmax, sum
    cfg.MODEL.ROI_BOX_HEAD.IMAGE_LOSS_WEIGHT = 0.1
    cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE = 1.0
    cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX = False  # Used for image-box loss and caption loss
    cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS = 128  # num proposals for image-labeled data
    cfg.MODEL.ROI_BOX_HEAD.WITH_SOFTMAX_PROP = False  # Used for WSDDN
    cfg.MODEL.ROI_BOX_HEAD.WITH_REFINEMENT_SCORE = False  # Used for OICR
    cfg.MODEL.ROI_BOX_HEAD.REFINEMENT_IOU = 0.5  # Used for OICR
    cfg.MODEL.ROI_BOX_HEAD.REGION_LOSS_WEIGHT = 3.0  # Used for OICR
    cfg.MODEL.ROI_BOX_HEAD.WITH_GLOBAL_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.GLOBAL_LOSS_WEIGHT = 0.1
    cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP = False  # Used for WSDDN
    cfg.MODEL.ROI_BOX_HEAD.SOFTMAX_WEAK_LOSS = False  # Used when USE_SIGMOID_CE is False
    cfg.MODEL.ROI_BOX_HEAD.NOUNS_WEIGHT_PATH = 'datasets/metadata/caption_4764nouns_clip_RN50_a+cname.npy'

    # Caption data configs
    cfg.MODEL.TEXT_ENCODER_TYPE = 'ViT-B/32'
    cfg.MODEL.TEXT_ENCODER_DIM = 512
    cfg.MODEL.WITH_CAPTION = False
    cfg.MODEL.CAP_BATCH_RATIO = 4  # Ratio between detection data and caption data
    cfg.MODEL.SYNC_CAPTION_BATCH = False  # synchronize across GPUs to enlarge # "classes"
    cfg.MODEL.ROI_BOX_HEAD.WITH_CAPTION_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.CAPTION_LOSS_WEIGHT = 1.0  # Caption loss weight
    cfg.MODEL.ROI_BOX_HEAD.NEG_CAP_WEIGHT = 0.125  # Caption loss hyper-parameter

    cfg.MODEL.ROI_HEADS.MASK_WEIGHT = 1.0
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False  # For demo only

    # dynamic class sampling when training with 21K classes
    cfg.MODEL.DYNAMIC_CLASSIFIER = False
    cfg.MODEL.NUM_SAMPLE_CATS = 50

    # Different classifiers in testing, used in cross-dataset evaluation
    cfg.MODEL.RESET_CLS_TESTS = False
    cfg.MODEL.TEST_CLASSIFIERS = []
    cfg.MODEL.TEST_NUM_CLASSES = []
    cfg.MODEL.DATASET_LOSS_WEIGHT = []

    # Backbones
    cfg.MODEL.BACKBONE.CONV_BODY = "VGG16"

    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'T'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (1, 2, 3)  # FPN stride 8 - 32

    cfg.MODEL.TIMM = CN()
    cfg.MODEL.TIMM.BASE_NAME = 'resnet50'
    cfg.MODEL.TIMM.OUT_LEVELS = (3, 4, 5)
    cfg.MODEL.TIMM.NORM = 'FrozenBN'
    cfg.MODEL.TIMM.FREEZE_AT = 0
    cfg.MODEL.TIMM.PRETRAINED = False

    # Multi-dataset dataloader
    cfg.DATALOADER.DATASET_RATIO = [1, 1]  # sample ratio
    cfg.DATALOADER.USE_RFS = [False, False]
    cfg.DATALOADER.MULTI_DATASET_GROUPING = False  # Always true when multi-dataset is enabled
    cfg.DATALOADER.DATASET_ANN = ['box', 'box']  # Annotation type of each dataset
    cfg.DATALOADER.DATASET_AUG = ['weak', 'weak']
    cfg.DATALOADER.USE_DIFF_BS_SIZE = False  # Use different batchsize for each dataset
    cfg.DATALOADER.DATASET_BS = [8, 32]  # Used when USE_DIFF_BS_SIZE is on
    cfg.DATALOADER.DATASET_INPUT_SIZE = [896, 384]  # Used when USE_DIFF_BS_SIZE is on
    cfg.DATALOADER.DATASET_INPUT_SCALE = [(0.1, 2.0), (0.5, 1.5)]  # Used when USE_DIFF_BS_SIZE is on
    cfg.DATALOADER.DATASET_MIN_SIZES = [(640, 800), (320, 400)]  # Used when USE_DIFF_BS_SIZE is on
    cfg.DATALOADER.DATASET_MAX_SIZES = [1333, 667]  # Used when USE_DIFF_BS_SIZE is on
    cfg.DATALOADER.USE_TAR_DATASET = False  # for ImageNet-21K, directly reading from unziped files
    cfg.DATALOADER.TARFILE_PATH = 'datasets/imagenet/metadata-22k/tar_files.npy'
    cfg.DATALOADER.TAR_INDEX_DIR = 'datasets/imagenet/metadata-22k/tarindex_npy'
    cfg.DATALOADER.CAPTION_PARSER = False

    # Custom solver
    cfg.SOLVER.USE_CUSTOM_SOLVER = False
    cfg.SOLVER.OPTIMIZER = 'SGD'
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0  # Used in DETR
    cfg.SOLVER.CUSTOM_MULTIPLIER = 1.0  # Used in DETR
    cfg.SOLVER.CUSTOM_MULTIPLIER_NAME = []  # Used in DETR

    # EfficientDetResizeCrop config
    cfg.INPUT.CUSTOM_AUG = ''
    cfg.INPUT.TRAIN_SIZE = 640
    cfg.INPUT.TEST_SIZE = 640
    cfg.INPUT.SCALE_RANGE = (0.1, 2.)

    # FP16
    cfg.FP16 = False
    cfg.FIND_UNUSED_PARAM = True

    # deprecated
    cfg.MODEL.CLUSTER_HEADS = CN()
    cfg.MODEL.CLUSTER_HEADS.NAME = "GroupOnFPN"
    cfg.MODEL.CLUSTER_HEADS.NUM_PARTS = 1000
    cfg.MODEL.CLUSTER_HEADS.NUM_OBJECTS = 300
    cfg.MODEL.CLUSTER_HEADS.TWO_STAGE_OBJECTNESS = False
    cfg.MODEL.CLUSTER_HEADS.PART_OBJECT_ENABLE = False
    cfg.MODEL.CLUSTER_HEADS.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.CLUSTER_HEADS.OUT_CHANNELS = 8
    cfg.MODEL.CLUSTER_HEADS.NUM_CONVS = 4
    cfg.MODEL.CLUSTER_HEADS.MASK_OUT_STRIDE = 4

    cfg.MODEL.CLUSTER_HEADS.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.CLUSTER_HEADS.CLASS_WEIGHT = 2.0
    cfg.MODEL.CLUSTER_HEADS.GIOU_WEIGHT = 2.0
    cfg.MODEL.CLUSTER_HEADS.L1_WEIGHT = 5.0
    cfg.MODEL.CLUSTER_HEADS.DICE_WEIGHT = 5.0
    cfg.MODEL.CLUSTER_HEADS.MASK_WEIGHT = 5.0
    cfg.MODEL.CLUSTER_HEADS.BIPARTITE_MATCH = False
    cfg.MODEL.CLUSTER_HEADS.IOU_THRESHOLDS = [0.3]
    cfg.MODEL.CLUSTER_HEADS.IOU_LABELS = [0, 1]

    cfg.TEACHER_STUDENT = CN()
    cfg.TEACHER_STUDENT.BURN_UP_STEP = 10000
    cfg.TEACHER_STUDENT.TEACHER_UPDATE_ITER = 1
    cfg.TEACHER_STUDENT.EMA_KEEP_RATE = 0.9996
    cfg.TEACHER_STUDENT.PSEUDO_LOSS_WEIGHT = 2.0
    cfg.TEACHER_STUDENT.EVAL_TEACHER = False

    cfg.WEAKLYSUP = CN()
    cfg.WEAKLYSUP.UNSUP_CLASS_INDEX = -1
    cfg.WEAKLYSUP.NUM_PROS = 32

    cfg.OPENWORLD.UNSUP_CLASS_INDEX = -1
    cfg.OPENWORLD.UNSUP_THRESH = 0.2
    cfg.OPENWORLD.STRONG_INPUT = False
    cfg.OPENWORLD.SELF_TRAIN = False
    cfg.OPENWORLD.PART_CLUSTER = False

    cfg.VIS = CN()
    cfg.VIS.SCORE = True
    cfg.VIS.BOX = True
    cfg.VIS.MASK = True
    cfg.VIS.LABELS = True

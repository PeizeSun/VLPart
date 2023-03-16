# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from detectron2.modeling import RPN_HEAD_REGISTRY, PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.proposal_generator.rpn import build_rpn_head
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from detectron2.modeling.box_regression import Box2BoxTransform, Box2BoxTransformLinear, _dense_box_regression_loss
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels


@RPN_HEAD_REGISTRY.register()
class OLNRPNHead(nn.Module):
    @configurable
    def __init__(
        self, *, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1,)
    ):
        super().__init__()
        cur_channels = in_channels
        # Keeping the old variable names and structure for backwards compatiblity.
        # Otherwise the old checkpoints will fail to load.
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            # 3x3 conv for the hidden representation
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels
        
        assert num_anchors == 1, "ctrness only support num_anchors as 1"
        # # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)
                
        # Keeping the order of weights initialization same for backwards compatiblility.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {
            "in_channels": in_channels,
            "num_anchors": num_anchors[0],
            "box_dim": box_dim,
            "conv_dims": cfg.MODEL.RPN.CONV_DIMS,
        }

    def forward(self, features: List[torch.Tensor]):
        pred_objectness_logits = []
        pred_anchor_deltas = []

        for x in features:
            t = self.conv(x)
            # We add L2 normalization for training stability
            t = F.normalize(t, p=2, dim=1)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class OLNRPN(nn.Module):
    """
    Object Localization Network, introduced by:
    Paper:
        Learning Open-World Object Proposals without Learning to Classify
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon, Weicheng Kuo
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        anchor_localization_matcher: Matcher,
        num_classes: int,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        box_reg_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.0,
        official_implement: bool = False, 
    ):
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.anchor_localization_matcher = anchor_localization_matcher
        self.num_classes = num_classes
        
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.official_implement = official_implement
        
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            # "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "box2box_transform": Box2BoxTransformLinear(),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["anchor_localization_matcher"] = Matcher(
            cfg.MODEL.RPN.OPENWORLD_IOU_THRESHOLDS, cfg.MODEL.RPN.OPENWORLD_IOU_LABELS, allow_low_quality_matches=True
        )
        ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        
        ret["official_implement"] = cfg.OPENWORLD.OLN_OFFICIAL
        return ret

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    def _subsample_labels_oln(self, label):
        """
        _subsample_labels with positive_fraction as 1
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, 1.0, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label
    
    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, 
        anchors: List[Boxes], 
        gt_instances: List[Instances],  
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        anchors = Boxes.cat(anchors)
        gt_boxes = [x.gt_boxes[x.gt_classes < self.num_classes] for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        gt_labels_oln = []
        matched_gt_boxes_oln = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            matched_idxs_oln, gt_labels_i_oln = retry_if_cuda_oom(self.anchor_localization_matcher)(match_quality_matrix)
            
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            gt_labels_i_oln = gt_labels_i_oln.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1
                gt_labels_i_oln[~anchors_inside_image] = -1
                
            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)
            gt_labels_i_oln = self._subsample_labels_oln(gt_labels_i_oln)
            
            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                matched_gt_boxes_i_oln = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
                matched_gt_boxes_i_oln = gt_boxes_i[matched_idxs_oln].tensor
            
            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
            
            gt_labels_oln.append(gt_labels_i_oln)  # N,AHW
            matched_gt_boxes_oln.append(matched_gt_boxes_i_oln)
        
        return gt_labels, matched_gt_boxes, gt_labels_oln, matched_gt_boxes_oln

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        # pred_ctrness_logits: List[torch.Tensor], 
        gt_labels_oln: List[torch.Tensor],
        gt_boxes_oln: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()

        if self.box_reg_loss_type in ["iou", "linear_iou"]:
            if num_pos_anchors < 1:
                localization_loss = cat(pred_anchor_deltas, dim=1).sum() * 0
            else:
                localization_loss = _iou_loss(
                    anchors,
                    self.box2box_transform,
                    pred_anchor_deltas,
                    gt_boxes,
                    pos_mask,
                    loss_type=self.box_reg_loss_type,
                )
        else:
            localization_loss = _dense_box_regression_loss(
                anchors,
                self.box2box_transform,
                pred_anchor_deltas,
                gt_boxes,
                pos_mask,
                box_reg_loss_type=self.box_reg_loss_type,
                smooth_l1_beta=self.smooth_l1_beta,
            )
        # valid_mask = gt_labels >= 0
        # objectness_loss = F.binary_cross_entropy_with_logits(
        #     cat(pred_objectness_logits, dim=1)[valid_mask],
        #     gt_labels[valid_mask].to(torch.float32),
        #     reduction="sum",
        # )
        
        gt_labels_oln = torch.stack(gt_labels_oln)  # (N, sum(Hi*Wi*Ai))        
        valid_mask_oln = gt_labels_oln >= 0
        pos_mask_oln = gt_labels_oln == 1
        neg_mask_oln = gt_labels_oln == 0
        num_pos_anchors = pos_mask_oln.sum().item()
        
        objectness_targets = torch.zeros_like(gt_labels_oln, dtype=torch.float)
        if num_pos_anchors < 1:
            objectness_loss = cat(pred_objectness_logits, dim=1).sum() * 0
        else:
            objectness_targets_pos = self.compute_ctrness_targets(anchors, gt_boxes_oln, pos_mask_oln)
            objectness_targets[pos_mask_oln] = objectness_targets_pos
    
            pred_objectness = cat(pred_objectness_logits, dim=1)    # (N, R)
            objectness_loss = smooth_l1_loss(
                pred_objectness.sigmoid()[valid_mask_oln], 
                objectness_targets[valid_mask_oln], 
                beta=0.0, 
                reduction="sum"
            )
        
        normalizer = self.batch_size_per_image * num_images
        losses = {
            # "loss_rpn_cls": objectness_loss * 0.0 / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
            "loss_rpn_objectness": objectness_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses
    
    
    @torch.no_grad()
    def compute_ctrness_targets(self, anchors: List[Boxes], gt_boxes: List[torch.Tensor], pos_mask_oln):
        anchors = Boxes.cat(anchors).tensor  # Rx4
        reg_targets = [self.box2box_transform.get_deltas(anchors, m) for m in gt_boxes]
        reg_targets = torch.stack(reg_targets, dim=0)  # NxRx4
        
        reg_targets_pos = reg_targets[pos_mask_oln]
        
        nonvalid_targets = torch.min(reg_targets_pos, -1)[0] <= 0
        reg_targets_pos[nonvalid_targets, :] = 0
                
        left_right = reg_targets_pos[:, [0, 2]]
        top_bottom = reg_targets_pos[:, [1, 3]]
        ctrness = (left_right.min(dim=-1)[0] / (left_right.max(dim=-1)[0] + 1e-12)) * (top_bottom.min(dim=-1)[0] / (top_bottom.max(dim=-1)[0] + 1e-12))
        
        return torch.sqrt(ctrness) 
    
    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):

        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        # pred_ctrness_logits = [
        #     # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
        #     score.permute(0, 2, 3, 1).flatten(1)
        #     for score in pred_ctrness_logits
        # ]
        
        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes, gt_labels_oln, gt_boxes_oln = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, 
                pred_objectness_logits, 
                gt_labels, pred_anchor_deltas, gt_boxes,
                gt_labels_oln, gt_boxes_oln
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    def predict_proposals(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            if self.official_implement:            
                proposals_i = tblr2bboxes(anchors_i, pred_anchor_deltas_i, normalizer=1.0)
            else:
                proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals
    
    
def tblr2bboxes(priors,
                tblr,
                normalizer=4.0,
                normalize_by_wh=True,
                max_shape=None,
                clip_border=True):
    
    if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'
    assert priors.size(0) == tblr.size(0)
    loc_decode = tblr * normalizer
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    if normalize_by_wh:
        wh = priors[:, 2:4] - priors[:, 0:2]
        w, h = torch.split(wh, 1, dim=1)
        loc_decode[:, :2] *= h  # tb
        loc_decode[:, 2:] *= w  # lr
    top, bottom, left, right = loc_decode.split((1, 1, 1, 1), dim=1)
    xmin = prior_centers[:, 0].unsqueeze(1) - left
    xmax = prior_centers[:, 0].unsqueeze(1) + right
    ymin = prior_centers[:, 1].unsqueeze(1) - top
    ymax = prior_centers[:, 1].unsqueeze(1) + bottom
    boxes = torch.cat((xmin, ymin, xmax, ymax), dim=1)
    if clip_border and max_shape is not None:
        boxes[:, 0].clamp_(min=0, max=max_shape[1])
        boxes[:, 1].clamp_(min=0, max=max_shape[0])
        boxes[:, 2].clamp_(min=0, max=max_shape[1])
        boxes[:, 3].clamp_(min=0, max=max_shape[0])
    return boxes


def _iou_loss(
    anchors, 
    box2box_transform, 
    pred_anchor_deltas, 
    gt_boxes, 
    pos_mask, 
    loss_type='linear_iou', 
    weight=None,
    eps=1e-6):
    
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
    else:
        anchors = cat(anchors)
    
    pos_anchors = torch.stack([anchors for _ in range(len(pred_anchor_deltas[0]))])[pos_mask]
    pos_pred_deltas = torch.stack([k for k in cat(pred_anchor_deltas, dim=1)])[pos_mask]
    pos_targets = torch.stack(gt_boxes)[pos_mask]
    
    pos_reg_targets = box2box_transform.get_deltas(pos_anchors, pos_targets)
    valid_targets = torch.min(pos_reg_targets, -1)[0] > 0
    
    if valid_targets.sum() < 1:
        return pos_pred_deltas.sum() * 0
    
    pos_anchors = pos_anchors[valid_targets]
    pos_pred_deltas = pos_pred_deltas[valid_targets]
    pos_targets = pos_targets[valid_targets]
    
    pos_pred = box2box_transform.apply_deltas(pos_pred_deltas, pos_anchors)

    pred = pos_pred
    target = pos_targets
    
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_right - target_left) * \
                  (target_bottom - target_top)
    pred_area = (pred_right - pred_left) * \
                (pred_bottom - pred_top)

    w_intersect = (torch.min(pred_right, target_right) - torch.max(pred_left, target_left)).clamp(min=0) 
    h_intersect = (torch.min(pred_bottom, target_bottom) - torch.max(pred_top, target_top)).clamp(min=0)
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect
    ious = area_intersect / torch.max(area_union, area_union.new_tensor([eps]))

    if loss_type == 'iou':
        losses = -torch.log(ious)
    elif loss_type == 'linear_iou':
        losses = 1 - ious
    else:
        raise NotImplementedError

    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum()
    else:
        assert losses.numel() != 0
        return losses.sum()

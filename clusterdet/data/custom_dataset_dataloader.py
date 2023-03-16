# Copyright (c) Facebook, Inc. and its affiliates.
# Part of the code is from https://github.com/xingyizhou/UniDet/blob/master/projects/UniDet/unidet/data/multi_dataset_dataloader.py (Apache-2.0 License)
import copy
import logging
import numpy as np
import operator
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.utils.data as torchdata
import json
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import _log_api_usage, log_first_n

from detectron2.config import configurable
from detectron2.data import samplers
from torch.utils.data.sampler import BatchSampler, Sampler
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import get_detection_dataset_dicts, build_batch_data_loader
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler, InferenceSampler
from detectron2.data.build import worker_init_reset_seed, print_instances_class_histogram
from detectron2.data.build import filter_images_with_only_crowd_annotations
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.data.build import check_metadata_consistency
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.utils import comm
import itertools
import math
from collections import defaultdict
from typing import Optional

logger = logging.getLogger('detectron2.clusterdet.data.custom_dataset_dataloader')


def _custom_test_loader_from_config(cfg, dataset_name, mapper=None):
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS_TEST
        else None,
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler": InferenceSampler(len(dataset))
        if not isinstance(dataset, torchdata.IterableDataset)
        else None,
    }


@configurable(from_config=_custom_test_loader_from_config)
def build_custom_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def trivial_batch_collator(batch):
    return batch


def _custom_train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    if 'MultiDataset' in sampler_name:
        dataset_dicts = get_detection_dataset_dicts_with_source(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
    else:
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is not None:
        pass
    elif sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "MultiDatasetSampler":
        sampler = MultiDatasetSampler(
            dataset_dicts,
            dataset_ratio = cfg.DATALOADER.DATASET_RATIO,
            use_rfs = cfg.DATALOADER.USE_RFS,
            dataset_ann = cfg.DATALOADER.DATASET_ANN,
            repeat_threshold = cfg.DATALOADER.REPEAT_THRESHOLD,
        )
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset_dicts,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        'multi_dataset_grouping': cfg.DATALOADER.MULTI_DATASET_GROUPING,
        'use_diff_bs_size': cfg.DATALOADER.USE_DIFF_BS_SIZE,
        'dataset_bs': cfg.DATALOADER.DATASET_BS,
        'num_datasets': len(cfg.DATASETS.TRAIN)
    }


@configurable(from_config=_custom_train_loader_from_config)
def build_custom_train_loader(
        dataset, *, mapper, sampler, 
        total_batch_size=16,
        aspect_ratio_grouping=True, 
        num_workers=0,
        num_datasets=1,
        multi_dataset_grouping=False,
        use_diff_bs_size=False,
        dataset_bs=[]
    ):
    """
    Modified from detectron2.data.build.build_custom_train_loader, but supports
    different samplers
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    if multi_dataset_grouping:
        return build_multi_dataset_batch_data_loader(
            use_diff_bs_size,
            dataset_bs,
            dataset,
            sampler,
            total_batch_size,
            num_datasets=num_datasets,
            num_workers=num_workers,
        )
    else:
        return build_batch_data_loader(
            dataset,
            sampler,
            total_batch_size,
            aspect_ratio_grouping=aspect_ratio_grouping,
            num_workers=num_workers,
        )


def build_multi_dataset_batch_data_loader(
    use_diff_bs_size, dataset_bs,
    dataset, sampler, total_batch_size, num_datasets, num_workers=0
):
    """
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=None,
        collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict
    if use_diff_bs_size:
        return DIFFMDAspectRatioGroupedDataset(
            data_loader, dataset_bs, num_datasets)
    else:
        return MDAspectRatioGroupedDataset(
            data_loader, batch_size, num_datasets)


def get_detection_dataset_dicts_with_source(
    dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None
):
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    
    for source_id, (dataset_name, dicts) in \
        enumerate(zip(dataset_names, dataset_dicts)):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
        for d in dicts:
            d['dataset_source'] = source_id

        if "annotations" in dicts[0]:
            try:
                class_names = MetadataCatalog.get(dataset_name).thing_classes
                check_metadata_consistency("thing_classes", dataset_name)
                print_instances_class_histogram(dicts, class_names)
            except AttributeError:  # class names are not available for this dataset
                pass

    assert proposal_files is None

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    return dataset_dicts


class MultiDatasetSampler(Sampler):
    def __init__(
        self, 
        dataset_dicts, 
        dataset_ratio,
        use_rfs,
        dataset_ann,
        repeat_threshold=0.001,
        seed: Optional[int] = None,
        ):
        """
        """
        sizes = [0 for _ in range(len(dataset_ratio))]
        for d in dataset_dicts:
            sizes[d['dataset_source']] += 1
        logger.info('dataset sizes {}'.format(sizes))
        
        self.sizes = sizes
        assert len(dataset_ratio) == len(sizes), \
            'length of dataset ratio {} should be equal to number if dataset {}'.format(
                len(dataset_ratio), len(sizes)
            )
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        
        self.dataset_ids = torch.tensor(
            [d['dataset_source'] for d in dataset_dicts], dtype=torch.long)

        dataset_weight = [torch.ones(s) * max(sizes) / s * r / sum(dataset_ratio) \
            for i, (r, s) in enumerate(zip(dataset_ratio, sizes))]
        dataset_weight = torch.cat(dataset_weight)

        rfs_factors = []
        st = 0
        for i, s in enumerate(sizes):
            if use_rfs[i]:
                if dataset_ann[i] == 'box':
                    rfs_func = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency
                else:
                    rfs_func = repeat_factors_from_tag_frequency
                rfs_factor = rfs_func(
                    dataset_dicts[st: st + s],
                    repeat_thresh=repeat_threshold)
                rfs_factor = rfs_factor * (s / rfs_factor.sum())
            else:
                rfs_factor = torch.ones(s)
            rfs_factors.append(rfs_factor)
            st = st + s
        rfs_factors = torch.cat(rfs_factors)

        self.weights = dataset_weight * rfs_factors
        self.sample_epoch_size = len(self.weights)


    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size)


    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            ids = torch.multinomial(
                self.weights, self.sample_epoch_size, generator=g, 
                replacement=True)
            nums = [(self.dataset_ids[ids] == i).sum().int().item() \
                for i in range(len(self.sizes))]
            yield from ids


class MDAspectRatioGroupedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, batch_size, num_datasets):
        """
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2 * num_datasets)]

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            aspect_ratio_bucket_id = 0 if w > h else 1
            bucket_id = d['dataset_source'] * 2 + aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]


class DIFFMDAspectRatioGroupedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, batch_sizes, num_datasets):
        """
        """
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self._buckets = [[] for _ in range(2 * num_datasets)]

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            aspect_ratio_bucket_id = 0 if w > h else 1
            bucket_id = d['dataset_source'] * 2 + aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_sizes[d['dataset_source']]:
                yield bucket[:]
                del bucket[:]


def repeat_factors_from_tag_frequency(dataset_dicts, repeat_thresh):
    """
    """
    category_freq = defaultdict(int)
    for dataset_dict in dataset_dicts:
        cat_ids = dataset_dict['pos_category_ids']
        for cat_id in cat_ids:
            category_freq[cat_id] += 1
    num_images = len(dataset_dicts)
    for k, v in category_freq.items():
        category_freq[k] = v / num_images

    category_rep = {
        cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
        for cat_id, cat_freq in category_freq.items()
    }

    rep_factors = []
    for dataset_dict in dataset_dicts:
        cat_ids = dataset_dict['pos_category_ids']
        rep_factor = max({category_rep[cat_id] for cat_id in cat_ids}, default=1.0)
        rep_factors.append(rep_factor)

    return torch.tensor(rep_factors, dtype=torch.float32)

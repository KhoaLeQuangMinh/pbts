"""nnU-Net's data loader with ONE change: crops that reach outside the case are padded with that case's
background value per channel (raw intensity 0 after our z-scoring, stored in data/head/records), not 0.
After our normalisation 0 means "average head brightness", so nnU-Net's zero padding would draw a bright
frame around every whole-head sample. Labels are padded with -1 exactly as in nnU-Net.

generate_train_batch is copied from nnunetv2 (training/dataloading/data_loader.py, installed version) with only
the data-padding line changed; tests/test_train.py checks the padded region holds the background values.
"""
import json
from functools import lru_cache

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from threadpoolctl import threadpool_limits

from pbts import config as C


@lru_cache(maxsize=None)
def pad_values(case: str):
    return tuple(json.load(open(C.HEAD_DIR / "records" / f"{case}.json"))["pad_value"])


def pad_with_background(data, bbox, case):
    cropped = crop_and_pad_nd(data, bbox, np.nan)
    pv = pad_values(case)
    for c in range(cropped.shape[0]):
        ch = cropped[c]
        ch[np.isnan(ch)] = pv[c]
    return cropped


class PBTSDataLoader(nnUNetDataLoader):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = None
        seg_all = None
        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]
                    bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
                    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

                    # ---- the only change vs nnU-Net: background padding instead of 0 ----
                    data_cropped = torch.from_numpy(pad_with_background(np.asarray(data), bbox, i)).float()
                    seg_cropped = torch.from_numpy(crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)
                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    if self.transforms is not None:
                        transformed = self.transforms(**{'image': data_cropped, 'segmentation': seg_cropped})
                        data_sample, seg_sample = transformed['image'], transformed['segmentation']
                    else:
                        data_sample, seg_sample = data_cropped, seg_cropped

                    if data_all is None:
                        data_all = torch.empty((self.batch_size, *data_sample.shape), dtype=torch.float32)
                    data_all[j] = data_sample
                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

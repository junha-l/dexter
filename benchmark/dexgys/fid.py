import glob
import json
import os
from pathlib import Path

import torch
import tqdm
import trimesh
from point_e.evals.fid_is import compute_statistics
from torch.functional import Tensor
from torch.utils.data import DataLoader, Dataset

from dexter.utils.shadowhand import ShadowHandModel

from ..common.feature_extractor import get_model, normalize_point_clouds
from ..common.report import report


def get_obj_points(oid, data_root_path, use_downsample=True, key="align"):
    data_dir = os.path.join(data_root_path, "meshes")
    meta_dir = os.path.join(data_dir, "metaV2")
    obj_suffix_path = "align_ds" if use_downsample else "align"
    real_meta = json.load(open(os.path.join(meta_dir, "object_id.json"), "r"))
    virtual_meta = json.load(open(os.path.join(meta_dir, "virtual_object_id.json"), "r"))

    if oid in real_meta:
        obj_name = real_meta[oid]["name"]
        obj_path = os.path.join(data_dir, "OakInkObjectsV2")
    else:
        obj_name = virtual_meta[oid]["name"]
        obj_path = os.path.join(data_dir, "OakInkVirtualObjectsV2")

    obj_mesh_path = list(
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj"))
        + glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply"))
    )
    if len(obj_mesh_path) > 1:
        obj_mesh_path = [p for p in obj_mesh_path if key in os.path.split(p)[1]]
    assert len(obj_mesh_path) == 1
    obj_path = obj_mesh_path[0]
    obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
    bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
    obj_trimesh.vertices = obj_trimesh.vertices - bbox_center
    points = trimesh.sample.sample_surface(obj_trimesh, 1024)
    points = torch.tensor(points[0], dtype=torch.float32)
    return points


class DictDataset(Dataset):
    def __init__(self, data_list, data_root_path, is_pred=True):
        self.data = data_list
        self.data_root_path = data_root_path
        self.is_pred = is_pred

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = {}
        if self.is_pred:
            data["predictions"] = torch.tensor(self.data[index]["predictions"]).squeeze()
            while data["predictions"].shape[0] != 28:
                data["predictions"] = data["predictions"][0]
        else:
            data["dex_grasp"] = torch.tensor(self.data[index]["targets"]).squeeze()

        data["obj_pc"] = get_obj_points(self.data[index]["obj_id"], self.data_root_path)
        return data

    @staticmethod
    def collate_fn(batch):
        input_dict = {}
        for k in batch[0]:
            if isinstance(batch[0][k], Tensor):
                try:
                    input_dict[k] = torch.stack([sample[k] for sample in batch])
                except Exception:
                    input_dict[k] = [sample[k] for sample in batch]
            else:
                input_dict[k] = [sample[k] for sample in batch]
        return input_dict


def main(pred_path: str, data_path: str, batch_size: int = 60):
    """Compute FID between predicted and ground-truth grasps.

    Args:
        pred_path: path to predictions.json (GT targets are read from the same file).
        data_path: dataset root containing meshes/.
        batch_size: dataloader batch size.
    """
    device = "cuda"

    dexhandmodel = ShadowHandModel(base_dir="./assets/shadowhand", device=device)
    feature_extractor = get_model()

    with open(pred_path) as f:
        pred_data = json.load(f)
    pred_dataset = DictDataset(pred_data, data_path, is_pred=True)
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=batch_size,
        collate_fn=pred_dataset.collate_fn,
        num_workers=8,
        shuffle=False,
    )

    # GT grasps ("targets") are stored alongside predictions in pred_data, so the
    # GT dataset is built from pred_data with is_pred=False (no separate gt file needed).
    gt_dataset = DictDataset(pred_data, data_path, is_pred=False)
    gt_loader = DataLoader(
        gt_dataset,
        batch_size=batch_size,
        collate_fn=gt_dataset.collate_fn,
        num_workers=8,
        shuffle=False,
    )

    features_pred = []

    for i, batch in tqdm.tqdm(enumerate(pred_loader), total=len(pred_loader)):
        obj_pcs = batch["obj_pc"].to(device)
        dexhand_pc = dexhandmodel(batch["predictions"].to(device), with_surface_points=True)[
            "surface_points"
        ]

        input_pc = normalize_point_clouds(torch.cat([obj_pcs, dexhand_pc], dim=1)).transpose(-1, -2)
        _, _, features = feature_extractor(input_pc, features=True)
        features_pred.append(features.cpu().detach())

    features_gt = []
    for i, batch in tqdm.tqdm(enumerate(gt_loader), total=len(gt_loader)):
        obj_pcs = batch["obj_pc"].to(device)
        dexhand_pc = dexhandmodel(batch["dex_grasp"].to(device), with_surface_points=True)[
            "surface_points"
        ]

        input_pc = normalize_point_clouds(torch.cat([obj_pcs, dexhand_pc], dim=1)).transpose(-1, -2)
        _, _, features = feature_extractor(input_pc, features=True)
        features_gt.append(features.cpu().detach())

    features_pred = torch.cat(features_pred, dim=0).numpy()
    stats_p = compute_statistics(features_pred)

    features_gt = torch.cat(features_gt, dim=0).numpy()
    stats_gt = compute_statistics(features_gt)

    fid = stats_p.frechet_distance(stats_gt)
    report(Path(pred_path).parent, "FID", {"fid": float(fid)}, n=len(pred_data))


if __name__ == "__main__":
    import fire

    fire.Fire(main)

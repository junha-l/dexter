import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch_fpsample
import tqdm
import trimesh
from chamferdist import ChamferDistance
from point_e.evals.fid_is import compute_statistics
from torch.functional import Tensor
from torch.utils.data import DataLoader

from dexter.data.dexonomy import DexonomyPredictionDataset
from dexter.utils.shadowhand_mujoco import RobotKinematics

from ..common.feature_extractor import get_model, normalize_point_clouds
from ..common.report import report


def collate_fn(batch):
    input_dict = {}
    for k in batch[0]:
        if isinstance(batch[0][k], Tensor):
            try:
                input_dict[k] = torch.stack([sample[k] for sample in batch]).float()
            except Exception:
                input_dict[k] = [sample[k] for sample in batch]
        elif isinstance(batch[0][k], np.ndarray):
            input_dict[k] = torch.stack([torch.from_numpy(sample[k]) for sample in batch]).float()
        else:
            input_dict[k] = [sample[k] for sample in batch]
    return input_dict


def main(pred_path: str, data_path: str = "/datasets/dexonomy/", batch_size: int = 60):
    """Compute FID and Chamfer distance between predicted and GT grasps.

    Writes per-grasp chamfer.csv and upserts the headline means (`chamfer`, `fid`) into
    metrics.json, both next to pred_path.

    Args:
        pred_path: path to predictions.json.
        data_path: dataset root.
        batch_size: dataloader batch size.
    """
    device = "cuda"

    hand_model = RobotKinematics(xml_path="./assets/shadowhand_mujoco/right_hand.xml")
    feature_extractor = get_model()

    dataset = DexonomyPredictionDataset(
        data_path=data_path, prediction_path=pred_path, load_pcd=True
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn
    )

    exp_name = "_".join(Path(pred_path).parent.name.split("_")[:-1])

    chamfer_distance = ChamferDistance()

    features_pred = []
    features_gt = []
    chamfer_losses = []
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    for i, batch in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        obj_pcs = batch["pcd"]
        obj_pc_down = torch_fpsample.sample(obj_pcs, 1024)[0]

        pred_hand_pc_list = []
        for pred_grasp in batch["pred_grasp"]:
            hand_model.forward_kinematics(pred_grasp[7:].cpu().numpy())
            pred_hand_mesh = hand_model.get_posed_meshes(pred_grasp[:7].cpu().numpy())
            pred_hand_pc = trimesh.sample.sample_surface(pred_hand_mesh, 1024)[0]
            pred_hand_pc = torch.from_numpy(pred_hand_pc)
            pred_hand_pc_list.append(pred_hand_pc)

        pred_hand_pc = torch.stack(pred_hand_pc_list, dim=0)
        input_pc = (
            normalize_point_clouds(torch.cat([obj_pc_down, pred_hand_pc], dim=1))
            .transpose(-1, -2)
            .float()
        ).to(device)
        _, _, features = feature_extractor(input_pc, features=True)
        features_pred.append(features.cpu().detach())

        gt_hand_pc_path = os.path.join(tmp_dir, f"gt_hand_pc_{i}_{exp_name}.pt")
        if os.path.exists(gt_hand_pc_path):
            gt_hand_pc = torch.load(gt_hand_pc_path)
        else:
            gt_hand_pc_list = []
            for gt_grasp in batch["gt_grasp"]:
                hand_model.forward_kinematics(gt_grasp[7:].cpu().numpy())
                gt_hand_mesh = hand_model.get_posed_meshes(gt_grasp[:7].cpu().numpy())
                gt_hand_pc = trimesh.sample.sample_surface(gt_hand_mesh, 1024)[0]
                gt_hand_pc = torch.from_numpy(gt_hand_pc)
                gt_hand_pc_list.append(gt_hand_pc)

            gt_hand_pc = torch.stack(gt_hand_pc_list, dim=0).float()
            torch.save(gt_hand_pc, gt_hand_pc_path)

        gt_feature_path = os.path.join(tmp_dir, f"gt_feature_{i}_{exp_name}.pt")
        if os.path.exists(gt_feature_path):
            gt_features = torch.load(gt_feature_path)
        else:
            input_pc = (
                normalize_point_clouds(torch.cat([obj_pc_down, gt_hand_pc], dim=1))
                .transpose(-1, -2)
                .float()
            ).to(device)
            _, _, gt_features = feature_extractor(input_pc, features=True)
            torch.save(gt_features.cpu().detach(), gt_feature_path)

        features_gt.append(gt_features.cpu().detach())

        chamfer_loss = chamfer_distance(
            pred_hand_pc.float().to(device),
            gt_hand_pc.float().to(device),
            bidirectional=True,
            batch_reduction=None,
        )
        chamfer_losses.append(chamfer_loss)

    features_pred = torch.cat(features_pred, dim=0).numpy()
    stats_p = compute_statistics(features_pred)

    features_gt = torch.cat(features_gt, dim=0).numpy()
    stats_gt = compute_statistics(features_gt)

    fid = stats_p.frechet_distance(stats_gt)
    chamfer_losses = torch.cat(chamfer_losses, dim=0).cpu().numpy()

    # Predictions are consumed in order (shuffle=False), so obj_ids line up with the losses.
    with open(pred_path) as f:
        obj_ids = [p["obj_id"] for p in json.load(f)]
    assert len(obj_ids) == len(chamfer_losses), (len(obj_ids), len(chamfer_losses))

    out_dir = Path(pred_path).parent
    with open(out_dir / "chamfer.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["obj_id", "chamfer"])
        writer.writerows(zip(obj_ids, chamfer_losses.tolist()))

    report(out_dir, "Chamfer", {"chamfer": float(chamfer_losses.mean())}, n=len(chamfer_losses))
    report(out_dir, "FID", {"fid": float(fid)}, n=len(dataset))


if __name__ == "__main__":
    import fire

    fire.Fire(main)

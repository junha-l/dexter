import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from benchmark.common.parallel import emit, read_slice, run_parallel
from benchmark.common.report import report
from dexter.models.loss import get_cmap_loss, get_hand_chamfer_loss
from dexter.utils.rot import axis_angle_to_quaternion
from dexter.utils.shadowhand import ShadowHandModel


def load_metadata(data_path: Path) -> dict:
    """Load test.json and group grasps/guidance by 'cate.obj.action' object code."""
    with open(data_path) as f:
        all_data = json.load(f)

    metadata = {}
    for entry in all_data:
        obj_code = f"{entry['cate_id']}.{entry['obj_id']}.{entry['action_id']}"
        if obj_code in metadata:
            metadata[obj_code]["dex_grasp"].append(entry["dex_grasp"])
            metadata[obj_code]["guidance"].append(entry["guidance"])
        else:
            metadata[obj_code] = {
                **entry,
                "dex_grasp": [entry["dex_grasp"]],
                "guidance": [entry["guidance"]],
            }
    return metadata


def find_target_key(metadata: dict, scene_name: str, prompt: str):
    """Return the unique object code matching this scene and prompt, else None."""
    candidates = [
        code for code in metadata if scene_name in code and prompt in metadata[code]["guidance"]
    ]
    return candidates[0] if len(candidates) == 1 else None


def match_grasp(pred: torch.Tensor, gt: torch.Tensor, weights: dict) -> int:
    """Index of the GT grasp closest to the single predicted grasp."""

    def l1(a, b):
        return F.l1_loss(a.expand_as(b), b, reduction="none").sum(-1)

    trans_cost = l1(pred[:, :3], gt[:, :3])
    qpos_cost = l1(pred[:, 6:], gt[:, 6:])
    pred_q, gt_q = axis_angle_to_quaternion(pred[:, 3:6]), axis_angle_to_quaternion(gt[:, 3:6])
    rot_cost = 1 - (pred_q @ gt_q.T).abs().squeeze(0)
    cost = (
        weights["translation"] * trans_cost
        + weights["rotation"] * rot_cost
        + weights["qpos"] * qpos_cost
    )
    return int(cost.argmin())


def run_worker(data_path: str, weights: dict):
    """Score this worker's slice, streaming one @@RESULT@@ record per prediction.

    GPU selection and single-thread caps come from the environment set by the runner.
    """
    torch.set_num_threads(1)
    data_path = Path(data_path)
    metadata = load_metadata(data_path / "test.json")
    hand_model = ShadowHandModel(base_dir="./assets/shadowhand", device="cuda")

    for pred in read_slice():
        scene_name, prompt = pred["obj_id"], pred["guidance"]
        target_key = find_target_key(metadata, scene_name, prompt)
        if target_key is None:
            emit({"skipped": True})  # still advances the bar toward total
            continue

        obj_pc = np.load(data_path / "data" / scene_name / "xyzc.npy")[:, :3]
        obj_pc = torch.from_numpy(obj_pc).float().cuda()
        pred_grasp = torch.from_numpy(np.array(pred["predictions"])).float().unsqueeze(0)
        gt = torch.from_numpy(np.array(metadata[target_key]["dex_grasp"])).float()

        target_idx = match_grasp(pred_grasp, gt, weights)
        pred_hand = hand_model(pred_grasp.cuda(), obj_pc, with_surface_points=True)
        gt_hand = hand_model(
            gt[target_idx : target_idx + 1].cuda(), obj_pc, with_surface_points=True
        )
        gt_hand["obj_pc"] = obj_pc

        emit({
            "scene_name": scene_name,
            "prompt": prompt,
            "num_gt": len(gt),
            "match_idx": target_idx,
            "chamfer_loss": get_hand_chamfer_loss(pred_hand, gt_hand, reduce=False).item(),
            "cmap_loss": get_cmap_loss(pred_hand, gt_hand, reduce=False).item(),
        })


def main(
    pred_path: str,
    data_path: str = "/root/data/dexgys",
    gpu: int = 0,
    parallel: int = 1,
    weight_qpos: float = 1.0,
    weight_translation: float = 2.0,
    weight_rotation: float = 2.0,
    worker: bool = False,
):
    """Match each prediction to its GT grasp and score the hand-chamfer and cmap losses.

    Default (orchestrator): fan the scoring across --parallel GPU-sharing workers, then
    write per-grasp results to chamfer.csv and upsert the headline means into metrics.json,
    both next to pred_path. (Penetration is reported separately by q1.py.)

    --worker is the internal per-slice mode, set by the orchestrator; not for direct use.
    """
    weights = {"qpos": weight_qpos, "translation": weight_translation, "rotation": weight_rotation}
    if worker:
        run_worker(data_path, weights)
        return

    with open(pred_path) as f:
        predictions = json.load(f)
    records = run_parallel(
        __file__,
        predictions,
        worker_args=[
            "--pred_path",
            pred_path,
            "--data_path",
            str(data_path),
            "--weight_qpos",
            str(weight_qpos),
            "--weight_translation",
            str(weight_translation),
            "--weight_rotation",
            str(weight_rotation),
        ],  # fmt: skip
        gpu=gpu,
        parallel=parallel,
        label="Chamfer",
        fields={
            "chamfer": lambda recs: mean(
                [r["chamfer_loss"] for r in recs if "chamfer_loss" in r] or [0.0]
            ),
            "cmap": lambda recs: mean([r["cmap_loss"] for r in recs if "cmap_loss" in r] or [0.0]),
        },
    )

    results = [r for r in records if not r.get("skipped")]
    results_df = pd.DataFrame(results)

    out_dir = Path(pred_path).parent
    results_df.to_csv(out_dir / "chamfer.csv", index=False)

    metrics = {
        key.removesuffix("_loss"): float(results_df[key].mean())
        for key in ["chamfer_loss", "cmap_loss"]
    }
    report(out_dir, "Chamfer", metrics, n=len(results))


if __name__ == "__main__":
    import fire

    fire.Fire(main)

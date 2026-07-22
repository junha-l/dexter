"""Q1 grasp-quality metric for DexGYS predictions.

Orchestrator + worker in one file. The orchestrator stays light (no torch/CUDA):
it splits the predictions into ``--parallel`` slices and re-spawns this file with
``--worker`` per slice, streaming each slice to its worker over stdin. Each worker
loads the models once, evaluates its samples on the GPU, and streams one
``@@RESULT@@`` line per object back on stdout, which the orchestrator aggregates
under a single progress bar.

Ported from Grasp-as-You-Say; uses dexter's ShadowHandModel and the OakInk mesh
layout (``<data_path>/meshes/{metaV2, OakInkObjectsV2, OakInkVirtualObjectsV2}``).
Input format matches scripts/test.py: ``{"obj_id": str, "predictions": list[float]  # 28-D}``.

Usage:
    python -m benchmark.dexgys.q1 --pred-path predictions.json \
        --gpu 0 --parallel 4 --data-path /datasets/dexgys
"""

from __future__ import annotations

import csv
import glob
import json
import os
import os.path as osp
import sys
from statistics import mean

from benchmark.common.parallel import emit, read_slice, run_parallel
from benchmark.common.report import report

CFG = {
    "lambda_torque": 10,
    "m": 8,
    "mu": 1,
    "thres_contact": 0.01,
    "thres_pen": 0.005,
}

# Hand links excluded from the object->hand penetration scan.
PEN_SKIP_LINKS = {
    "robot0:forearm",
    "robot0:wrist_child",
    "robot0:ffknuckle_child",
    "robot0:mfknuckle_child",
    "robot0:rfknuckle_child",
    "robot0:lfknuckle_child",
    "robot0:thbase_child",
    "robot0:thhub_child",
}


def _heavy_import():
    """Bind torch/csdf/trimesh/scipy/dexter to module globals.

    Called only inside a worker (never the orchestrator) so the orchestrator stays
    light and GPU-free until CUDA_VISIBLE_DEVICES is set.
    """
    global np, torch, trimesh, scipy
    global compute_sdf, index_vertices_by_faces, axis_angle_to_matrix, ShadowHandModel
    import numpy as np  # noqa: F401
    import scipy.spatial  # noqa: F401
    import torch  # noqa: F401
    import trimesh  # noqa: F401
    from csdf import compute_sdf, index_vertices_by_faces  # noqa: F401

    from dexter.utils.rot import axis_angle_to_matrix  # noqa: F401
    from dexter.utils.shadowhand import ShadowHandModel  # noqa: F401


# ============================================================================
# Compute core (runs inside a worker)
# ============================================================================
class ObjectModel:
    """Loads an OakInk object mesh and exposes its SDF / surface samples."""

    def __init__(self, mesh_root: str, device: str = "cuda"):
        self.mesh_root = mesh_root
        self.device = device
        meta_dir = os.path.join(mesh_root, "metaV2")
        with open(os.path.join(meta_dir, "object_id.json")) as f:
            self.real_meta = json.load(f)
        with open(os.path.join(meta_dir, "virtual_object_id.json")) as f:
            self.virtual_meta = json.load(f)

    def _mesh_path(self, oid: str, key: str = "align") -> str:
        is_real = oid in self.real_meta
        meta = self.real_meta if is_real else self.virtual_meta
        subdir = "OakInkObjectsV2" if is_real else "OakInkVirtualObjectsV2"
        obj_dir = os.path.join(self.mesh_root, subdir, meta[oid]["name"], "align_ds")
        paths = glob.glob(os.path.join(obj_dir, "*.obj")) + glob.glob(
            os.path.join(obj_dir, "*.ply")
        )
        if len(paths) > 1:
            paths = [p for p in paths if key in os.path.basename(p)]
        assert len(paths) == 1, (len(paths), oid)
        return paths[0]

    def initialize(self, oid: str):
        """Load the mesh for `oid` (recentered to its bbox), ready for cal_distance/penetration."""
        mesh = trimesh.load(self._mesh_path(oid), process=False, force="mesh", skip_materials=True)
        mesh.vertices = mesh.vertices - (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
        self.face_verts = index_vertices_by_faces(
            torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device),
            torch.tensor(mesh.faces, dtype=torch.long, device=self.device),
        )
        self.surface_points = torch.tensor(
            mesh.sample(4096), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def cal_distance(self, x):
        """For points x (1, n, 3): signed distance (inside +), surface normals, closest points."""
        dis, normals, signs, _, _ = compute_sdf(x[0], self.face_verts)
        closest = x[0] - dis.sqrt().unsqueeze(1) * normals
        signed_dis = torch.sqrt(dis + 1e-8) * (-signs)
        signed_normals = normals * signs.unsqueeze(1)
        return signed_dis.unsqueeze(0), signed_normals.unsqueeze(0), closest.unsqueeze(0)


def cal_q1(hand_model, object_model, hand_pose) -> float:
    """Force-closure Q1 quality for one hand pose against the initialized object."""
    hand_pose = hand_pose.unsqueeze(0)
    global_translation = hand_pose[:, 0:3]
    global_rotation = axis_angle_to_matrix(hand_pose[:, 3:6])
    current_status = hand_model.chain.forward_kinematics(hand_pose[:, 6:])

    contact_points, contact_normals = [], []
    for link_name, link in hand_model.mesh.items():
        if len(link["surface_points"]) == 0:
            continue
        points = current_status[link_name].transform_points(link["surface_points"])
        points = points @ global_rotation.transpose(1, 2) + global_translation.unsqueeze(1)
        distances, normals, closest = object_model.cal_distance(points)
        nearest = distances.argmax()
        if -distances[0, nearest] < CFG["thres_contact"]:
            contact_points.append(closest[0, nearest])
            contact_normals.append(normals[0, nearest])

    if not contact_points:
        contact_points.append(torch.tensor([0.0, 0.0, 0.0], device=hand_pose.device))
        contact_normals.append(torch.tensor([1.0, 0.0, 0.0], device=hand_pose.device))
    contact_points = torch.stack(contact_points).cpu().numpy()
    contact_normals = torch.stack(contact_normals).cpu().numpy()
    if np.isnan(contact_points).any() or np.isnan(contact_normals).any():
        return 0.0

    # Friction-cone basis (u, v) perpendicular to each contact normal.
    n = len(contact_points)
    u1 = np.stack([-contact_normals[:, 1], contact_normals[:, 0], np.zeros(n, np.float32)], axis=1)
    u2 = np.tile(np.array([1, 0, 0], np.float32), (n, 1))
    u = np.where(np.linalg.norm(u1, axis=1, keepdims=True) > 1e-8, u1, u2)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(u, contact_normals)

    # Discretize each cone into m forces, turn (force, torque) into 6-D wrenches.
    theta = np.linspace(0, 2 * np.pi, CFG["m"], endpoint=False).reshape(-1, 1, 1)
    forces = (contact_normals + CFG["mu"] * (np.cos(theta) * u + np.sin(theta) * v)).reshape(-1, 3)
    torques = CFG["lambda_torque"] * np.cross(np.tile(contact_points, (CFG["m"], 1)), forces)
    wrenches = np.concatenate(
        [np.concatenate([forces, torques], axis=1), np.zeros((1, 6), np.float32)], axis=0
    )

    # Q1 = distance from the origin to the convex-hull boundary of the wrench set.
    try:
        hull = scipy.spatial.ConvexHull(wrenches)
    except scipy.spatial.QhullError:
        return 0.0
    q1 = 1.0
    for eq in hull.equations:
        q1 = min(q1, abs(eq[6]) / np.linalg.norm(eq[:6]))
    return float(q1)


def cal_pen(hand_model, object_model, hand_pose) -> float:
    """Max penetration depth (m) of the object surface into the hand."""
    hand_pose = hand_pose.unsqueeze(0)
    global_translation = hand_pose[:, 0:3]
    global_rotation = axis_angle_to_matrix(hand_pose[:, 3:6])
    current_status = hand_model.chain.forward_kinematics(hand_pose[:, 6:])

    x = (object_model.surface_points - global_translation.unsqueeze(1)) @ global_rotation
    depths = []
    for link_name, link in hand_model.mesh.items():
        if link_name in PEN_SKIP_LINKS:
            continue
        matrix = current_status[link_name].get_matrix()
        x_local = ((x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]).reshape(-1, 3)
        if "geom_param" not in link:
            dis, _, signs, _, _ = compute_sdf(x_local, link["face_verts"])
            depth = torch.sqrt(dis + 1e-8) * (-signs)
        else:
            radius = link["geom_param"][0]
            height = link["geom_param"][1] * 2
            nearest = x_local.clone()
            nearest[:, :2] = 0
            nearest[:, 2] = nearest[:, 2].clamp(0, height)
            depth = radius - (x_local - nearest).norm(dim=1)
        depths.append(depth.reshape(x.shape[0], x.shape[1]))
    penetration = torch.stack(depths).max(dim=0).values
    return max(penetration.max().item(), 0.0)


# ============================================================================
# Worker (one slice; loads models once, streams per-object results on stdout)
# ============================================================================
def run_worker(mesh_root: str, assets_dir: str):
    """Evaluate the stdin slice, streaming one @@RESULT@@ record per object.

    GPU selection and single-thread caps come from the environment set by the runner.
    """
    _heavy_import()
    torch.set_num_threads(1)

    hand_model = ShadowHandModel(base_dir=assets_dir, device="cuda")
    object_model = ObjectModel(mesh_root, device="cuda")

    for sample in read_slice():
        hand_pose = torch.tensor(sample["predictions"], device="cuda")
        if hand_pose.dim() == 3:
            hand_pose = hand_pose.squeeze(1)
        elif hand_pose.dim() == 1:
            hand_pose = hand_pose.unsqueeze(0)

        object_model.initialize(sample["obj_id"])
        pen, valid_q1 = [], []
        for pose in hand_pose:
            depth = cal_pen(hand_model, object_model, pose)
            q1 = cal_q1(hand_model, object_model, pose)
            pen.append(depth)
            valid_q1.append(q1 if depth < CFG["thres_pen"] else 0.0)
        emit({"obj_id": sample["obj_id"], "pen": pen, "valid_q1": valid_q1})


def main(
    pred_path: str,
    data_path: str = "/root/data/dexgys",
    assets_dir: str = "./assets/shadowhand",
    gpu: int = 0,
    parallel: int = 1,
    worker: bool = False,
):
    """Compute the Q1 grasp-quality metric for a predictions.json.

    Default (orchestrator): fan Q1 evaluation across --parallel GPU-sharing workers,
    write per-grasp results to q1.csv, and upsert the headline means (`valid_q1`,
    `mean_pen`) into metrics.json next to pred_path. `valid_q1` is the Q1 value zeroed
    for any grasp whose penetration exceeds `thres_pen`.

    --worker is the internal per-slice mode; it reads its samples from stdin and is
    set by the orchestrator, not for direct use.
    """
    if worker:
        run_worker(os.path.join(data_path, "meshes"), assets_dir)
        return

    with open(pred_path) as f:
        samples = json.load(f)
    records = run_parallel(
        __file__,
        samples,
        worker_args=["--pred_path", pred_path, "--data_path", data_path, "--assets_dir", assets_dir],
        gpu=gpu,
        parallel=parallel,
        label="Q1",
        fields={
            "pen": lambda recs: mean([p for r in recs for p in r["pen"]] or [0.0]),
            "valid_q1": lambda recs: mean([v for r in recs for v in r["valid_q1"]] or [0.0]),
        },
    )

    pens = [p for r in records for p in r["pen"]]
    valid = [v for r in records for v in r["valid_q1"]]
    if not pens:
        sys.exit("No results produced.")

    out_dir = osp.dirname(osp.abspath(pred_path))
    with open(osp.join(out_dir, "q1.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["obj_id", "pen", "valid_q1"])
        for r in records:
            for p, v in zip(r["pen"], r["valid_q1"]):
                writer.writerow([r["obj_id"], p, v])

    report(out_dir, "Q1", {"valid_q1": mean(valid), "mean_pen": mean(pens)}, n=len(pens))


if __name__ == "__main__":
    import fire

    fire.Fire(main)

"""
Validate grasps on the Isaac Gym simulator using the DexGYS dataset.

This single script both orchestrates a full benchmark run and acts as the per-object
worker it spawns. There is no longer an intermediate run.sh / per-object _poses.json /
per-shard result files (the old sim_batch.py + merge_shards.py flow) — the orchestrator
groups predictions in memory, fans objects out across worker subprocesses, and writes
the canonical success_rate file directly.

Usage (full benchmark, the common case):
    python benchmark/dexgys/success_rate.py \
        --data-path /path/to/datasets/dexgys \
        --pred-path /path/to/experiment/predictions.json \
        --no_force --parallel 8

    -> writes <pred dir>/success_rate[_raw].json

Single object (debug / rendering):
    python benchmark/dexgys/success_rate.py --data-path ... --pred-path ... --object_code A01001
    python benchmark/dexgys/success_rate.py --data-path ... --pred-path ... --object_code A01001 --index 0

Why one process per object: each object creates and destroys exactly one Isaac Gym sim;
doing more than one create/destroy in a single process accumulates toward Isaac Gym's
repeated-sim CUDA crash. The orchestrator therefore runs each object in its own
short-lived worker subprocess (concurrency = --parallel), which also overlaps the
~5-10s isaacgym/torch import across workers. The orchestrator process itself never
imports torch/isaacgym, so it stays light and GPU-free.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

NUM_SIM_TESTS = 6  # number of stability tests per grasp (lift, shake, etc.)
RESULT_MARKER = "@@RESULT@@"  # worker -> orchestrator result line prefix on stdout
DEFAULT_HAND_ASSET_ROOT = str(REPO_ROOT / "assets" / "shadowhand_openai")


@contextlib.contextmanager
def suppress_low_level_output():
    """Mute Isaac Gym / PhysX banners printed from C at the file-descriptor level.

    Covers "Importing module 'gym_38'", "+++ Using GPU PhysX", "JointSpec type free
    not yet supported!", etc. These come from C, so Python logging can't filter them;
    we redirect fd 1/2 to /dev/null around the noisy calls and restore afterwards.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        for fd in (devnull, saved_out, saved_err):
            os.close(fd)


# --------------------------------------------------------------------------------------
# Result I/O (orchestrator side — no torch needed)
# --------------------------------------------------------------------------------------
def load_existing_results(results_file):
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                return json.load(f)
        except Exception:
            return {"results": [], "summary": {}}
    return {"results": [], "summary": {}}


def get_result_path(output_dir, no_force):
    suffix = "_raw" if no_force else ""
    return os.path.join(output_dir, f"success_rate{suffix}.json")


def write_canonical(result_path, results_by_id):
    """Write the canonical success_rate file from an {object_id: record} mapping.

    Results are sorted by object_id (deterministic, matches a serial run). Called after
    every completed object so an interrupted run leaves a valid, resumable file.
    """
    results = [results_by_id[k] for k in sorted(results_by_id)]
    total_grasps = sum(r["total_grasps"] for r in results)
    total_successful = sum(r["successful_grasps"] for r in results)
    data = {
        "results": results,
        "summary": {
            "total_objects": len(results),
            "total_grasps": total_grasps,
            "total_successful": total_successful,
            "overall_success_rate": (total_successful / total_grasps * 100)
            if total_grasps
            else 0.0,
        },
    }
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)
    return data["summary"]


def is_grasp_successful(sim_results, grasp_idx):
    """A grasp passes if at least 1 of NUM_SIM_TESTS stability tests succeeds."""
    start = grasp_idx * NUM_SIM_TESTS
    return sum(sim_results[start : start + NUM_SIM_TESTS]) >= 1


# --------------------------------------------------------------------------------------
# Per-object simulation (worker side — uses torch/isaacgym bound by _heavy_import)
# --------------------------------------------------------------------------------------
def _heavy_import():
    """Import isaacgym/torch/dexter once and bind them to module globals.

    Run only inside a worker process (or the in-process single-object debug path), never
    in the orchestrator. isaacgym (pulled in by validator) MUST be imported before torch.
    """
    global torch, axis_angle_to_matrix, matrix_to_quaternion
    global IsaacValidator, ObjectModel, ShadowHandModel
    with suppress_low_level_output():
        # isort: off
        from validator import IsaacValidator  # noqa: I001,F401
        import torch  # noqa: F401
        from dexter.utils.rot import axis_angle_to_matrix, matrix_to_quaternion  # noqa: F401
        from dexter.utils.object_model import ObjectModel  # noqa: F401
        from dexter.utils.shadowhand import ShadowHandModel  # noqa: F401
        # isort: on


def optimize_joint_angles(
    data_dict, hand_model, object_model, device, thres_cont, dis_move, grad_move
):
    """1-step gradient optimization to push fingers into object surface."""
    hand_state = torch.stack(
        [torch.tensor(d["predictions"], dtype=torch.float, device=device) for d in data_dict]
    ).requires_grad_()

    hand_model(hand_state, with_surface_points=True)

    batch_size = len(data_dict)
    contact_points = torch.zeros((batch_size, len(hand_model.mesh), 3), device=device)
    contact_normals = torch.zeros((batch_size, len(hand_model.mesh), 3), device=device)

    global_translation = hand_state[:, 0:3]
    global_rotation = axis_angle_to_matrix(hand_state[:, 3:6])
    current_status = hand_model.chain.forward_kinematics(hand_state[:, 6:])

    for i, link_name in enumerate(hand_model.mesh):
        if len(hand_model.mesh[link_name]["surface_points"]) == 0:
            continue
        surface_pts = current_status[link_name].transform_points(
            hand_model.mesh[link_name]["surface_points"]
        )
        surface_pts = surface_pts @ global_rotation.transpose(1, 2) + global_translation.unsqueeze(
            1
        )

        distances, normals = object_model.cal_distance(surface_pts)
        idx = distances.argmax(dim=1)
        nearest_dist = torch.gather(distances, 1, idx.unsqueeze(1))
        nearest_pts = torch.gather(surface_pts, 1, idx.reshape(-1, 1, 1).expand(-1, 1, 3))
        nearest_nrm = torch.gather(normals, 1, idx.reshape(-1, 1, 1).expand(-1, 1, 3))

        admitted = (-nearest_dist < thres_cont).reshape(-1, 1, 1).expand(-1, 1, 3)
        contact_points[:, i : i + 1, :] = torch.where(
            admitted, nearest_pts, contact_points[:, i : i + 1, :]
        )
        contact_normals[:, i : i + 1, :] = torch.where(
            admitted, nearest_nrm, contact_normals[:, i : i + 1, :]
        )

    target = contact_points + contact_normals * dis_move
    loss = (target.detach() - contact_points).square().sum()
    loss.backward()
    with torch.no_grad():
        hand_state[:, 6:] += hand_state.grad[:, 6:] * grad_move

    return hand_state


def compute_penetration_energy(data_dict, hand_model, object_surface_points, device, chunk_size=64):
    """Compute per-grasp penetration energy (E_pen), batched over grasps.

    The hand model forward is fully batched over the grasp dimension and broadcasts
    object_surface_points (shape (1, num_samples, 3)), so we run grasps in chunks of
    chunk_size instead of one-at-a-time. chunk_size bounds peak memory of the SDF.
    """
    preds = torch.tensor(
        np.asarray([d["predictions"] for d in data_dict]), dtype=torch.float, device=device
    )
    e_pen = []
    for start in range(0, len(preds), chunk_size):
        hand_output = hand_model(preds[start : start + chunk_size], object_pc=object_surface_points)
        dist = hand_output["distances"].clamp(min=0)  # (chunk, num_samples)
        e_pen.append(dist.sum(-1))
    return torch.cat(e_pen).cpu().numpy()


def parse_grasp_data(data_dict):
    """Parse prediction dicts into simulation-ready arrays."""
    rotations, translations, hand_poses = [], [], []
    for d in data_dict:
        pred = d["predictions"]
        aa = torch.tensor(pred[3:6]).unsqueeze(0)
        quat = matrix_to_quaternion(axis_angle_to_matrix(aa)).squeeze().numpy().tolist()
        rotations.append(quat)
        translations.append(pred[0:3])
        hand_poses.append(pred[6:])
    return rotations, translations, hand_poses


def run_simulation(sim, rotations, translations, hand_poses, asset_args, obj_urdf_root):
    """Run Isaac Gym simulation over all of an object's grasps; return per-grasp
    success boolean array.

    Every grasp goes into a single sim run (one set_asset + one run_sim), so the sim
    is created and destroyed exactly once per object — avoiding the repeated-sim CUDA
    error entirely. Asset loading and the run are wrapped in suppress_low_level_output()
    to mute Isaac Gym's PhysX / JointSpec C-level banners.
    """
    batch_size = len(rotations)
    with suppress_low_level_output():
        sim.set_asset(asset_args["root"], asset_args["file"], obj_urdf_root, "coacd.urdf")
        for i in range(batch_size):
            sim.add_env(rotations[i], translations[i], hand_poses[i], 1)
        all_results = sim.run_sim()

    return np.array([is_grasp_successful(all_results, i) for i in range(batch_size)])


def _prepare_poses(object_code, data_dict, data_path, device, no_force, cfg):
    """Build (hand_model, obj_urdf_root, rotations, translations, hand_poses) for one
    object, applying pose optimization unless no_force."""
    obj_urdf_root = os.path.join(data_path, "data", object_code, "urdf")
    shadow_hand_dir = str(REPO_ROOT / "assets" / "shadowhand")
    hand_model = ShadowHandModel(base_dir=shadow_hand_dir, device=str(device))

    hand_state = None
    if not no_force:
        object_model = ObjectModel(data_root_path=data_path, num_samples=0, device=device)
        object_model.batch_size_each = len(data_dict)
        object_model.initialize([object_code])
        hand_state = optimize_joint_angles(
            data_dict,
            hand_model,
            object_model,
            device,
            cfg["thres_cont"],
            cfg["dis_move"],
            cfg["grad_move"],
        )

    rotations, translations, hand_poses = parse_grasp_data(data_dict)
    if not no_force:
        hand_poses = hand_state[:, 6:]
    return hand_model, obj_urdf_root, rotations, translations, hand_poses


def simulate_one(object_code, data_dict, data_path, gpu, no_force, cfg):
    """Validate a single object's grasps; return a result dict (no file I/O)."""
    _heavy_import()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu)
    batch_size = len(data_dict)

    hand_model, obj_urdf_root, rotations, translations, hand_poses = _prepare_poses(
        object_code, data_dict, data_path, device, no_force, cfg
    )

    object_model_pen = ObjectModel(data_root_path=data_path, num_samples=2000, device=device)
    object_model_pen.batch_size_each = 1
    object_model_pen.initialize([object_code])
    e_pen = compute_penetration_energy(
        data_dict, hand_model, object_model_pen.surface_points_tensor, device
    )

    with suppress_low_level_output():
        sim = IsaacValidator(gpu=gpu)
    asset_args = {"root": cfg["hand_asset_root"], "file": cfg["hand_asset_file"]}
    simulated = run_simulation(sim, rotations, translations, hand_poses, asset_args, obj_urdf_root)
    sim.destroy()

    estimated = e_pen < cfg["penetration_threshold"]
    valid = simulated & estimated
    n_valid = int(valid.sum())
    return {
        "object_id": object_code,
        "total_grasps": int(batch_size),
        "successful_grasps": n_valid,
        "success_rate": float(n_valid / batch_size * 100) if batch_size else 0.0,
        "n_sim": int(simulated.sum()),
        "n_pen": int(estimated.sum()),
    }


def render_one(object_code, data_dict, data_path, gpu, no_force, cfg, index):
    """Render a single grasp (debug visualization), mirroring the old --index path."""
    _heavy_import()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu)
    _, obj_urdf_root, rotations, translations, hand_poses = _prepare_poses(
        object_code, data_dict, data_path, device, no_force, cfg
    )
    with suppress_low_level_output():
        sim = IsaacValidator(gpu=gpu)
    sim.save_render = True
    sim.set_asset(cfg["hand_asset_root"], cfg["hand_asset_file"], obj_urdf_root, "coacd.urdf")
    sim.add_env_single(rotations[index], translations[index], hand_poses[index], 1, 0)
    sim.run_sim()
    sim.destroy()


# --------------------------------------------------------------------------------------
# Orchestrator (parent side — no torch/isaacgym import here)
# --------------------------------------------------------------------------------------
def _group_by_object(pred_path):
    with open(pred_path) as f:
        preds = json.load(f)
    grouped: dict[str, list] = {}
    for r in preds:
        grouped.setdefault(r["obj_id"], []).append(r)
    return grouped


def orchestrate(data_path, pred_path, output_dir, gpu, no_force, parallel, skip_objects, cfg):
    """Group predictions, fan objects out across worker subprocesses, render progress
    with rich, and write the canonical success_rate file incrementally."""
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ImportError:
        sys.exit("This benchmark needs 'rich' (pip install rich).")

    grouped = _group_by_object(pred_path)
    object_ids = sorted(grouped)

    os.makedirs(output_dir, exist_ok=True)
    result_path = get_result_path(output_dir, no_force)
    results = {r["object_id"]: r for r in load_existing_results(result_path)["results"]}
    skip = set(skip_objects) | set(results)
    todo = [o for o in object_ids if o not in skip]

    done_before = len(object_ids) - len(todo)
    if not todo:
        print(f"All {len(object_ids)} objects already done in {result_path}")
        return
    print(
        f"Simulating {len(todo)} objects "
        f"({done_before} already done) across {parallel} workers on GPU {gpu}"
    )

    # Base worker command; each worker gets --object_code and its grasps via stdin.
    worker_base = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--data_path",
        data_path,
        "--gpu",
        str(gpu),
        "--thres_cont",
        str(cfg["thres_cont"]),
        "--dis_move",
        str(cfg["dis_move"]),
        "--grad_move",
        str(cfg["grad_move"]),
        "--penetration_threshold",
        str(cfg["penetration_threshold"]),
        "--hand_asset_root",
        cfg["hand_asset_root"],
        "--hand_asset_file",
        cfg["hand_asset_file"],
    ]
    if no_force:
        worker_base.append("--no_force")

    lock = threading.Lock()
    failed: list[str] = []

    def run_one(object_code):
        try:
            proc = subprocess.Popen(
                worker_base + ["--object_code", object_code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, err = proc.communicate(input=json.dumps(grouped[object_code]))
            if proc.returncode != 0:
                return object_code, None, err
            for line in out.splitlines():
                if line.startswith(RESULT_MARKER):
                    return object_code, json.loads(line[len(RESULT_MARKER) :]), None
            return object_code, None, err  # no result marker emitted
        except Exception as e:  # noqa: BLE001
            return object_code, None, str(e)

    # A single global progress bar; per-object results are printed above it as workers
    # finish (progress.console.print keeps the bar intact at the bottom).
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("elapsed •"),
        TimeRemainingColumn(),
        TextColumn("left"),
    )
    with progress:
        overall_task = progress.add_task("Simulating", total=len(todo))
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            for fut in as_completed([ex.submit(run_one, o) for o in todo]):
                object_code, res, err = fut.result()
                with lock:
                    if res is None:
                        failed.append(object_code)
                        snippet = (err or "").strip().splitlines()[-1:] or [""]
                        progress.console.print(f"[red]FAILED[/] {object_code}  {snippet[0]}")
                    else:
                        results[object_code] = {
                            k: res[k]
                            for k in (
                                "object_id",
                                "total_grasps",
                                "successful_grasps",
                                "success_rate",
                            )
                        }
                        write_canonical(result_path, results)
                        bs = res["total_grasps"]
                        progress.console.print(
                            f"{object_code:<8} sim {res['n_sim']:>3}/{bs:<3}"
                            f"  pen {res['n_pen']:>3}/{bs:<3}"
                            f"  valid {res['successful_grasps']:>3}/{bs:<3}"
                            f" ({res['success_rate']:5.1f}%)"
                        )
                    progress.advance(overall_task)

    summary = write_canonical(result_path, results)
    msg = (
        f"\nDone: {summary['total_objects']} objects  "
        f"{summary['total_successful']}/{summary['total_grasps']} = "
        f"{summary['overall_success_rate']:.1f}%  -> {result_path}"
    )
    if failed:
        msg += f"\n{len(failed)} failed (re-run to retry): {sorted(failed)}"
    print(msg)


def main(
    *,
    data_path: str,
    pred_path: str | None = None,
    output_dir: str | None = None,
    gpu: int = 0,
    no_force: bool = False,
    parallel: int = 1,
    skip_objects: tuple = (),
    object_code: str | None = None,
    index: int | None = None,
    worker: bool = False,
    thres_cont: float = 0.001,
    dis_move: float = 0.001,
    grad_move: float = 500,
    penetration_threshold: float = 0.001,
    hand_asset_root: str = DEFAULT_HAND_ASSET_ROOT,
    hand_asset_file: str = "hand/shadow_hand.xml",
):
    """Run the DexGYS Isaac Gym grasp-success benchmark.

    Modes (auto-selected):
      * Orchestrator (default): needs --data-path + --pred-path. Groups predictions,
        runs every object in its own worker subprocess (--parallel workers), and writes
        <pred dir>/success_rate[_raw].json (override dir with --output-dir).
      * Single object (debug): pass --object_code [--index N for a rendered grasp].
      * Worker (internal, --worker): runs one object, reading its grasps from stdin and
        printing a result line to stdout. Spawned by the orchestrator; not for direct use.

    Args:
        data_path: dataset root (contains data/<obj>/urdf).
        pred_path: predictions.json (flat list of {obj_id, predictions, ...}).
        output_dir: output dir for success_rate[_raw].json (default: <pred dir>).
        gpu: CUDA device id.
        no_force: skip pose optimization (use raw predictions).
        parallel: number of concurrent worker subprocesses sharing the GPU.
        skip_objects: object IDs to skip (e.g. --skip_objects o42131,o23104).
        object_code: run just this object (debug / rendering) instead of the full set.
        index: with object_code, render a single grasp instead of scoring the batch.
        thres_cont / dis_move / grad_move: pose-optimization knobs (ignored if no_force).
        penetration_threshold: E_pen cutoff for the (always-applied) penetration filter.
        hand_asset_root / hand_asset_file: Shadow Hand asset location.
    """
    cfg = {
        "thres_cont": thres_cont,
        "dis_move": dis_move,
        "grad_move": grad_move,
        "penetration_threshold": penetration_threshold,
        "hand_asset_root": hand_asset_root,
        "hand_asset_file": hand_asset_file,
    }

    # --- Worker mode: one object, grasps from stdin, result to stdout. ---
    if worker:
        if object_code is None:
            sys.exit("--worker requires --object_code")
        data_dict = json.loads(sys.stdin.read())
        res = simulate_one(object_code, data_dict, data_path, gpu, no_force, cfg)
        sys.stdout.write(RESULT_MARKER + json.dumps(res) + "\n")
        sys.stdout.flush()
        return

    if pred_path is None:
        sys.exit("--pred-path is required (predictions.json)")

    # --- Single-object debug / render mode. ---
    if object_code is not None:
        data_dict = _group_by_object(pred_path).get(object_code)
        if not data_dict:
            sys.exit(f"{object_code} not found in {pred_path}")
        if index is not None:
            render_one(object_code, data_dict, data_path, gpu, no_force, cfg, index)
        else:
            res = simulate_one(object_code, data_dict, data_path, gpu, no_force, cfg)
            bs = res["total_grasps"]
            print(
                f"{object_code:<8} sim {res['n_sim']:>3}/{bs:<3}  pen {res['n_pen']:>3}/{bs:<3}"
                f"  valid {res['successful_grasps']:>3}/{bs:<3} ({res['success_rate']:5.1f}%)"
            )
        return

    # --- Orchestrator (full benchmark). ---
    if output_dir is None:
        output_dir = os.path.dirname(pred_path)
    orchestrate(
        data_path, pred_path, output_dir, gpu, no_force, parallel, tuple(skip_objects), cfg
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)

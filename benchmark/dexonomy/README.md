# Dexonomy Benchmark

Scoring a `predictions.json` on Dexonomy. Generate one first with `scripts/test.py` — see
[Evaluation](../../README.md#evaluation) in the top-level README — and point `$PRED` at its
directory:

```bash
PRED=test_output/dexonomy_dexter_unseen_obj
```

Dexonomy measures **generalization**, so repeat everything below once per split (`seen_val` /
`unseen_grasp` / `unseen_obj` / `unseen_both`). Two kinds of metric, both reading that same file:

| Metric | Measures | Direction |
|--------|----------|-----------|
| FID | realism of the generated pose distribution | lower = better |
| Chamfer | agreement with the ground-truth grasp | lower = better |
| Success rate | MuJoCo force-closure shake test (DexGraspBench) | higher = better |

---

## 1. Environment setup

```bash
uv sync
```

That covers everything on this page, MuJoCo included. See the
[top-level README](../../README.md) for details.

---

## 2. Computational metrics

FID and Chamfer are computed together (they share the hand-posing pipeline). The run prints both
headline numbers to the console, writes per-grasp scores to `chamfer.csv`, and upserts the numbers
into a shared `$PRED/metrics.json`:

```bash
python -m benchmark.dexonomy.fid --pred-path $PRED/predictions.json --data-path /datasets/dexonomy
```

`metrics.json` holds the reported numbers in one place. Read it with `jq . $PRED/metrics.json`:

```json
{ "chamfer": ..., "fid": ... }
```

---

## 3. Simulation-based metric

MuJoCo rollout with analytic force-closure and penetration/contact metrics.

```bash
python -m benchmark.dexonomy.success_rate \
    task.pred_path=$PRED/predictions.json \
    task.data_path=/datasets/dexonomy
```

| Override | Description |
|----------|-------------|
| `task.pred_path` | Path to `predictions.json` (required) |
| `task.data_path` | Path to Dexonomy dataset root (required) |
| `n_worker=48` | Number of parallel workers (default: 48) |
| `task.skip_existing=False` | Re-evaluate all grasps (default: True, skips existing) |
| `task.filter_ids_path=/path/to/ids.txt` | Only evaluate specific grasps |

### Output & parse

```
$PRED/
├── eval/<obj_id>/<index>.npy    # per-grasp evaluation results
├── succ/<obj_id>/<index>.npy    # successful grasps only
└── log/eval.log                 # evaluation log
```

Success rate = successful grasps / evaluated grasps:

```bash
python -c "import glob; e=len(glob.glob('$PRED/eval/**/*.npy',recursive=True)); \
s=len(glob.glob('$PRED/succ/**/*.npy',recursive=True)); print(f'success {100*s/e:.2f}%  ({s}/{e})')"
# or: grep succeeded $PRED/log/eval.log
```

### Visualization

Render rollouts as GIFs with per-grasp debug images:

```bash
MUJOCO_GL=egl python -m benchmark.dexonomy.success_rate \
    task.pred_path=$PRED/predictions.json \
    task.data_path=/datasets/dexonomy \
    task.debug_render=True
#  → $PRED/debug/{success,failure}/<obj_id>/<index>.gif
```

---

## 4. What to report

- **Success rate** — `succ / eval` count (primary Dexonomy metric).
- **FID** and **Chamfer** — from `$PRED/metrics.json` (`fid`, `chamfer`).
- Report all four splits to show generalization.

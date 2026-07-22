# DexGYS Benchmark

Scoring a `predictions.json` on DexGYS. Generate one first with `scripts/test.py` (see
[Evaluation](../../README.md#evaluation)) and point `$PRED` at its directory:

```bash
# e.g.
PRED=test_output/dexgys_dexter_test
```

Two kinds of metric, **computational** (§1) and **simulation-based** (§2), both reading that same file and writing their results next to it. The scripts output **raw** values; **Unit** is the factor multiplies each raw value by to report it, e.g. `mean_pen` ×100 (meters → cm).

| Kind | Metric (Table 1) | `metrics.json` | Measures | Better | Unit |
|------|------------------|----------------|----------|--------|-------------|
| Computational | Chamfer (CD)           | `chamfer`      | agreement with the ground-truth grasp        | ↓ | raw        |
| Computational | Contact map (Con.)     | `cmap`         | contact-map agreement with the GT grasp      | ↓ | ×10⁻¹        |
| Computational | Q1 (Q₁)                | `valid_q1`     | force-closure quality, zeroed on penetration | ↑ | ×10⁻¹      |
| Computational | Penetration (Pen.)     | `mean_pen`     | hand→object penetration depth                | ↓ | ×100 (cm)  |
| Computational | FID (P-FID)            | `fid`          | realism of the generated pose distribution   | ↓ | raw        |
| Simulation    | Success rate (Success) | —¹             | Isaac Gym 6-direction shake test             | ↑ | %          |

- `metrics.json` keeps the raw value; multiply by **Unit** to reproduce the paper number (e.g.
  `mean_pen` 0.008 m × 100 = 0.8 cm).
- `valid_q1` is Q₁ zeroed for grasps penetrating more than 5 mm.

¹ Success rate is written to `success_rate.json` by the simulation metric (§2), not `metrics.json`.

---

## 1. Computational metrics

```bash
source .venv/bin/activate # main env (.venv)
```

Run the commands below to compute the computational metrics. Each is a standalone script that prints its headline numbers, writes raw per-grasp scores to a CSV, and **upserts** its numbers into a shared `$PRED/metrics.json`:

```bash
python -m benchmark.dexgys.chamfer \
    --pred-path $PRED/predictions.json \
    --data-path $DATA_ROOT/dexgys_final \
    --gpu 0 --parallel 16

python -m benchmark.dexgys.q1 \
    --pred-path $PRED/predictions.json \
    --data-path $DATA_ROOT/dexgys_final \
    --gpu 0 --parallel 16
    
python -m benchmark.dexgys.fid \
    --pred-path $PRED/predictions.json \
    --data-path $DATA_ROOT/dexgys_final
```

Adjust `--gpu` to run on a different GPU, and `--parallel` to run more (or fewer) GPU-sharing workers.

The result is one file (`$PRED/metrics.json`) with every reported number, keyed by metric:

```json
{
  "hand_chamfer": ..., "cmap": ...,   // chamfer
  "valid_q1": ..., "mean_pen": ...,   // q1
  "fid": ...                          // fid
}
```

---

## 2. Simulation-based metric

The **success rate** replays each grasp in Isaac Gym and checks whether it survives a shake test.
Because Isaac Gym requires Python ≤3.8, it runs in its own environment. `setup_isaacgym.sh` downloads Preview 4, vendors it into the repo, and builds `.venv-isaacgym` from [`isaacgym-env/pyproject.toml`](isaacgym-env/pyproject.toml); it is safe to re-run.

```bash
bash benchmark/dexgys/setup_isaacgym.sh     # builds .venv-isaacgym
source .venv-isaacgym/bin/activate

python benchmark/dexgys/success_rate.py \
    --pred-path $PRED/predictions.json \
    --data-path $DATA_ROOT/dexgys_final \
    --no_force --gpu 0 --parallel 8
```

`success_rate.py` groups the predictions by object in memory and runs each object in its own short-lived worker subprocess (concurrency = `--parallel`), writing `$PRED/success_rate[_raw].json` directly.

- `--no_force` evaluates raw predictions (skip pose optimization); penetration filtering is
  always applied. A grasp passes if it survives **≥1 of 6 shake directions**.
- Single GPU → `--gpu 0`. `--parallel N` runs N worker subprocesses concurrently on that GPU;
  raise N for throughput, lower it (or `--parallel 1`) under memory pressure.

### Output & parse

Written to `$PRED/success_rate.json` (or `success_rate_raw.json` with `--no_force`):

```json
{
  "results": [ {"object_id": "...", "total_grasps": N, "successful_grasps": M, "success_rate": ...}, ... ],
  "summary": { "total_objects": ..., "total_grasps": ..., "total_successful": ..., "overall_success_rate": <pct> }
}
```

```bash
python -c "import json; s=json.load(open('$PRED/success_rate_raw.json'))['summary']; \
print(f\"success {s['overall_success_rate']:.2f}%  ({s['total_successful']}/{s['total_grasps']}, {s['total_objects']} objs)\")"
# or: jq '.summary' $PRED/success_rate_raw.json
```

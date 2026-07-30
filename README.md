# salad-kelpie-docking

A generic harness for running containerised docking and molecular dynamics jobs
on [SaladCloud](https://salad.com) (interruptible consumer GPU/CPU cloud) via
[Kelpie](https://github.com/SaladTechnologies/kelpie), with Cloudflare R2 as
the sole data channel.

Ships two Docker images and a worked example using the public PDB entry 1HSG
(HIV-1 protease + indinavir).

---

## Operational findings up front

These were learned by running this pattern in production.  Read them before
touching the code.

**1. The queue's job status is a proxy — completion is the result artifact.**
Kelpie can report `completed` before the upload finishes, and `failed` for a job
that already wrote its result on a previous attempt.  Use `poll.py`, which checks
R2 directly, not the queue API.

**2. Resource requests are a filter, not a budget.**
Asking for more RAM or CPU threads than a job needs reduces the set of matching
nodes and causes allocation stalls.  Defaults in this repo are intentionally
modest.  Increase them only if you observe OOM failures, not pre-emptively.

**3. A stall in allocation is a capacity mismatch, not a failure.**
If jobs sit in `pending` for more than 10 minutes, the fix is to broaden
hardware classes in the Salad Portal or lower the resource request — never to
retry identically.  See `docs/RUNBOOK.md`.

**4. CPU prep, GPU run, CPU scoring.**
Parametrisation, solvation, and MM-GBSA scoring are CPU-bound.  Running them on
a GPU node wastes accelerator time and increases cost.  This repo encodes that
split: `Dockerfile.cpu` for docking prep, `Dockerfile.gpu` for MD trajectories,
and `mmgbsa.py` can run on any CPU.

**5. Cache expensive per-item work.**
The MD engine caches `system.xml` and `solvated.pdb` in the checkpoint directory
after the first build.  Resumed jobs skip solvation and AM1-BCC charge
calculation entirely.

**6. The run prefix is provenance.**
`--run-prefix` is required and has no default.  Reusing a prefix deliberately
extends a run.  Change it whenever the image, inputs, or protocol change.

---

## Repository layout

```
submit.py              build job definitions and submit them to Kelpie
poll.py                poll R2 for completion (not the queue API)
Dockerfile.cpu         lightweight smina docking image
Dockerfile.gpu         OpenMM + GAFF-2 + MM-GBSA image (CUDA)
worker/
  cpu_worker.py        smina docking worker (invoked by Kelpie)
  gpu_worker.py        MD + scoring orchestrator (invoked by Kelpie)
  md_engine.py         OpenMM engine with checkpoint/resume
  mmgbsa.py            PBC-corrected MM-GBSA scorer
config/
  .env.example         local submitter credentials (copy to config/.env)
  portal.cpu.env.example  CPU container group Portal variables
  portal.gpu.env.example  GPU container group Portal variables
examples/docking/
  prepare_example.sh   download and prep public 1HSG data
  jobs.json            CPU docking manifest for the example
  matrix.json          GPU MD manifest for the example
  data/                prepared PDBQT and SDF files (after prepare_example.sh)
docs/RUNBOOK.md        what to do when things stall or behave oddly
tests/                 unit tests (no network)
```

---

## Quick start — CPU docking

### 1. Prerequisites

- Docker with ≥ 4 GB RAM and x86_64 build support
- A Salad account with a CPU container group running this image
- A Cloudflare R2 bucket
- Open Babel (`brew install open-babel` or `apt install openbabel`)
- Python 3.11+ with `boto3` (`pip install boto3`)

### 2. Prepare example data

```bash
bash examples/docking/prepare_example.sh
```

This downloads PDB 1HSG, extracts the receptor and co-crystal ligand
(indinavir), and prepares three additional HIV protease inhibitors as test
molecules.  All data comes from public sources.

### 3. Upload inputs to R2

```bash
source config/.env   # fill in config/.env from config/.env.example first

aws s3 cp examples/docking/data/1hsg__indinavir/ \
  s3://${R2_BUCKET}/${RUN_PREFIX}/inputs/1hsg__indinavir/ \
  --endpoint-url ${AWS_ENDPOINT_URL} --recursive
# repeat for other job IDs
```

### 4. Submit

```bash
python submit.py --mode cpu \
                 --manifest examples/docking/jobs.json \
                 --run-prefix 2024-01-example-v1
```

### 5. Poll for results

```bash
python poll.py --mode cpu \
               --manifest examples/docking/jobs.json \
               --run-prefix 2024-01-example-v1 \
               --watch
```

---

## Quick start — GPU MD + MM-GBSA

### Prerequisites (additional)

- A Salad GPU container group running the `Dockerfile.gpu` image
- Input data at `/app/data/<target>__<ligand>/{receptor.pdb,ligand.sdf}` baked
  into the image (see `examples/docking/prepare_example.sh` for the 1HSG example)

### Submit

```bash
python submit.py --mode gpu \
                 --manifest examples/docking/matrix.json \
                 --run-prefix 2024-01-md-v1 \
                 --prod-ns 10 \
                 --n-reps 3
```

### Poll

```bash
python poll.py --mode gpu \
               --manifest examples/docking/matrix.json \
               --run-prefix 2024-01-md-v1 \
               --n-reps 3 \
               --watch
```

---

## Architecture

```
 Operator workstation
   submit.py  ──── POST /jobs ────► Kelpie API
   poll.py    ──── head_object ───► R2 bucket

 Salad CPU node                   Salad GPU node
   Kelpie                           Kelpie
     sync.before: R2 → /app/input/    sync.before: R2 → checkpoints/
     run: cpu_worker.py               sync.during: checkpoints/ → R2
     sync.after: /app/outputs/ → R2   run: gpu_worker.py
                                      sync.after: outputs/ → R2

 R2 bucket
   <prefix>/inputs/<id>/         ← inputs (CPU mode)
   <prefix>/outputs/<id>/        ← result.json + poses (CPU mode)
   <prefix>/<name>/rep<N>/checkpoints/  ← trajectory + checkpoint (GPU mode)
   <prefix>/<name>/rep<N>/outputs/      ← mmgbsa.json (GPU mode)
```

---

## Environment variables

See `config/.env.example` for the full list with descriptions.

Variables set in the Salad Portal (worker side) are documented in
`config/portal.cpu.env.example` and `config/portal.gpu.env.example`.

---

## Tests

```bash
pip install pytest boto3
python -m pytest tests/ -v
```

Tests cover job definition construction, polling logic, and worker argument
validation.  No network access is required.

---

## Costs

Costs scale with trajectory length (ns), system size (number of atoms), and
Salad node pricing, which varies by GPU model.  No specific figures are given
here because they must be re-derived for your system size and hardware class.
Order of magnitude: consumer GPU nodes on Salad are typically 1–2 orders of
magnitude cheaper than on-demand cloud GPU instances.

---

## Contributing

PRs welcome.  Before submitting, run `python -m pytest tests/ -v` and verify
no strings from `SCRUB.local.md` appear in your changes.

---

## Licence

MIT

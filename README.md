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

**4. CPU docking, GPU MD, in-container CPU scoring.**
`Dockerfile.cpu` runs smina docking.  `Dockerfile.gpu` runs OpenMM MD and then
uses the node's CPU for the GBn2/GAFF endpoint estimate.  The GPU remains
allocated during scoring, so keep `N_FRAMES` modest and verify the cost on a
one-job smoke test before scaling a campaign.

**5. Cache expensive per-item work.**
The MD engine caches `system.xml`, `solvated.pdb`, and
`system_metadata.json` in the checkpoint directory after the first build.
Resumed jobs skip solvation and AM1-BCC charge calculation entirely.

**6. The run prefix is provenance.**
`--run-prefix` is required and has no default.  Reusing a prefix deliberately
extends a compatible GPU run.  Change it whenever the image, inputs, or
protocol change.  The submitter also keeps a local `submitted_jobs.jsonl`
ledger and refuses duplicate job keys unless `--allow-duplicate` is explicit.

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
  prepare_example.sh   build 1HSG example from RCSB + PubChem (InChIKey-verified)
  jobs.json            CPU docking manifest for the example
  matrix.json          GPU MD manifest for the example
  data/1hsg__<ligand>/ one self-contained input set per job id:
                         receptor.pdbqt, ligand.pdbqt, box.txt  (CPU docking)
                         receptor.pdb,   ligand.sdf             (GPU MD)
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

This downloads PDB 1HSG from RCSB for the receptor, and four FDA-approved
HIV-1 protease inhibitors (indinavir, ritonavir, nelfinavir, lopinavir) from
PubChem by CID.  Every ligand is checked against its published InChIKey before
use, so the run fails loudly rather than silently docking a wrong structure.
All data comes from public sources.

All four ligands are generated as fresh 3D conformers the same way, so their
scores are directly comparable.  The co-crystallised indinavir pose is kept
separately as `1hsg__indinavir/ligand_crystal.pdb` — it is not a docking input,
it is the reference for measuring redocking RMSD.

### 3. Upload inputs to R2

The CPU worker needs only `receptor.pdbqt`, `ligand.pdbqt`, and `box.txt`; the
`.pdb`/`.sdf` files in the same directory are GPU MD inputs and are baked into
the GPU image instead.

```bash
source config/.env   # fill in config/.env from config/.env.example first
export RUN_PREFIX=2026-07-example-v1

for d in examples/docking/data/*/; do
  job_id="$(basename "$d")"
  aws s3 cp "$d" "s3://${R2_BUCKET}/${RUN_PREFIX}/inputs/${job_id}/" \
    --endpoint-url "${AWS_ENDPOINT_URL}" --recursive \
    --exclude "*" --include "receptor.pdbqt" --include "ligand.pdbqt" --include "box.txt"
done
```

### 4. Submit

```bash
python submit.py --mode cpu \
                 --manifest examples/docking/jobs.json \
                 --run-prefix 2026-07-example-v1
```

### 5. Poll for results

```bash
python poll.py --mode cpu \
               --manifest examples/docking/jobs.json \
               --run-prefix 2026-07-example-v1 \
               --watch
```

---

## Quick start — GPU MD + MM-GBSA

### Prerequisites (additional)

- A Salad GPU container group running the `Dockerfile.gpu` image
- Input data at `/app/data/<target>__<ligand>/{receptor.pdb,ligand.sdf}` baked
  into the image (see `examples/docking/prepare_example.sh` for the 1HSG example)

### Submit

Start with one 3 ns job.  The default `SKIP_NS=2` and `CHECKPOINT_PS=500`
remain valid for this smoke test.

```bash
python submit.py --mode gpu \
                 --manifest examples/docking/matrix.json \
                 --run-prefix 2026-07-md-smoke-v1 \
                 --prod-ns 3 \
                 --n-reps 1 \
                 --max 1
```

Confirm the trajectory, checkpoint, and MM-GBSA result in R2 before submitting
the larger matrix.  Use a new prefix for the corrected engine; checkpoints
created by versions without `system_metadata.json` are intentionally rejected.

### Poll

```bash
python poll.py --mode gpu \
               --manifest examples/docking/matrix.json \
               --run-prefix 2026-07-md-smoke-v1 \
               --n-reps 1 \
               --watch
```

### Configure scale-to-zero

Create one Kelpie scaling rule per container group.  Start with a maximum of 2
replicas and increase only after clean smoke runs:

```bash
curl --fail-with-body -X POST "${KELPIE_API_URL}/scaling-rules" \
  -H "Salad-Api-Key: ${SALAD_API_KEY}" \
  -H "Salad-Organization: ${SALAD_ORGANIZATION}" \
  -H "Salad-Project: ${SALAD_PROJECT}" \
  -H "Content-Type: application/json" \
  -d "{\"container_group_id\":\"${GPU_CONTAINER_GROUP_ID}\",\"min_replicas\":0,\"max_replicas\":2,\"idle_threshold_seconds\":0}"
```

Until this rule is verified, stop the container group manually whenever the
queue is empty; an idle running GPU worker is still billable.

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
     sync.after: /app/outputs/ → R2   run: gpu_worker.py (GPU MD, CPU scoring)
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
Treat those tracked files as references only: never put real credentials in
them, and enter credential-bearing values as Salad Portal secrets rather than
ordinary environment variables.

---

## Tests

```bash
pip install pytest boto3
python -m pytest tests/ -v
```

Tests cover job definition construction, duplicate protection, polling,
worker failures, MD unit conversion, progress metadata, and scorer atom
partitioning.  No network access is required.

---

## Costs

Costs scale with trajectory length (ns), system size (number of atoms), and
Salad node pricing, which varies by GPU model.  No specific figures are given
here because they must be re-derived for your system size and hardware class.
Running replicas are billable while idle; configure and verify Kelpie
scale-to-zero with `min_replicas=0`.
Benchmark the current Salad price and measured ns/day for the selected hardware
against alternatives before scaling; neither pricing nor achieved throughput is
stable enough for a generic savings multiplier.

---

## Contributing

PRs welcome.  Before submitting, run `python -m pytest tests/ -v` and verify
no strings from `SCRUB.local.md` appear in your changes.

---

## Licence

MIT

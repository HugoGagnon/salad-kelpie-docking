# RUNBOOK — what to do when things stall or behave oddly

## Quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Jobs stay `pending` for >10 min | Capacity mismatch | Broaden hardware classes or lower resource request |
| Jobs claimed but no heartbeat | Node died before first checkpoint | Kelpie retries automatically |
| GPU job completes on queue but no result.json | `complete=false` in progress.json | Normal — Kelpie retries; wait for next allocation |
| All replicas fail on first attempt | Bad input data or environment variable | Check error log from one job |
| poll.py shows everything pending long after submission | Wrong `--run-prefix` | Verify the prefix matches what was submitted |
| boto3 `NoCredentialsError` | `.env` not sourced | `source config/.env` before running poll.py |

---

## Stalled allocations

**Stalls are a capacity mismatch, not a failure.** Retrying identically does nothing.

The resource request in a job definition is a filter over the pool of available
Salad nodes.  If too few nodes match (wrong GPU model, VRAM threshold too high,
too many CPU threads requested), the job sits in `pending` indefinitely.

Fix: **broaden first, lower second.**
1. In the Salad Portal, add more hardware classes to the container group.
2. Reduce the requested CPU threads or VRAM if you're specifying them.
3. Do not retry the same job definition identically — change the container group
   configuration instead.

---

## Checking whether a job actually completed

Do not rely on the Kelpie job status alone.  The queue status can show
`completed` for a job that never wrote its result, and `failed` for a job whose
result is already in R2 from a previous attempt.

**The canonical check:**

```bash
source config/.env
python poll.py --mode cpu --manifest examples/docking/jobs.json --run-prefix <your-prefix>
```

A job is done only when `result.json` (CPU) or `rep<N>_mmgbsa.json` (GPU)
exists in R2.

---

## GPU jobs: checkpoints and resume

GPU MD jobs are interruptible by design.  If a Salad node is preempted
(evicted, rebooted, or allocated to another user), Kelpie will:
1. Pick up the `sync.during` checkpoint that was pushed to R2 during the run.
2. Allocate the job to a new node.
3. Run `sync.before` to restore the checkpoint directory.
4. Re-launch the worker, which resumes from the binary checkpoint.

This is automatic.  You do not need to resubmit.

If a job has been pending for a new allocation for more than 30 minutes, see
**Stalled allocations** above.

---

## Resuming a campaign with the same run prefix

Using the same `--run-prefix` deliberately extends an existing campaign:
- **GPU jobs resume. CPU jobs do not.**
- GPU jobs: `sync.before` restores the checkpoint; the engine picks up from
  the last saved state.  Re-submitting a partially finished replica is cheap.
- CPU jobs: **not idempotent.**  `cpu_worker.py` does not check for an existing
  `result.json`, and it cannot — `sync.before` gives it only the *inputs*
  prefix, so prior outputs are not on the node.  It then wipes `/app/outputs/`
  at startup.  Re-submitting a manifest under the same prefix re-docks every
  job in it and overwrites the previous `result.json` and `poses.pdbqt`.
  To extend a CPU campaign, submit a manifest containing only the jobs that
  have not finished yet — use `poll.py` to see which those are.

Change the run prefix when you change the image, inputs, or protocol.  Mixing
outputs from different images or force fields under the same prefix will
silently corrupt results.

---

## Monitoring a running campaign

```bash
# Watch until all jobs complete (polls every 60 s):
python poll.py --mode cpu --manifest examples/docking/jobs.json \
               --run-prefix <your-prefix> --watch --interval 60

# GPU campaign with 3 replicas:
python poll.py --mode gpu --manifest examples/docking/matrix.json \
               --run-prefix <your-prefix> --n-reps 3 --watch
```

---

## Docker image builds

Build the CPU image on any machine (cross-compilation works fine):

```bash
docker build --platform linux/amd64 -f Dockerfile.cpu \
  -t <registry>/docking-cpu:001 .
```

Build the GPU image — requires ≥ 8 GB RAM in Docker Desktop:

```bash
# In Docker Desktop: Settings → Resources → Memory ≥ 8 GB
docker build --platform linux/amd64 -f Dockerfile.gpu \
  --build-arg KELPIE_SHA256=<digest> \
  -t <registry>/docking-gpu:001 .
```

The GPU build performs two separate conda transactions to keep peak solver
RAM under the Docker VM limit.  Do not merge them into one.

---

## Validating example data

Before uploading inputs, run the prepare script and verify locally:

```bash
bash examples/docking/prepare_example.sh
# Check that all four PDBQT files were created:
ls examples/docking/data/1hsg__indinavir/
```

Run a local dry-run to verify the job definitions:

```bash
source config/.env
python submit.py --mode cpu --manifest examples/docking/jobs.json \
                 --run-prefix test-v1 --dry-run
```

---

## Pre-publish checklist (before adding a remote and pushing)

Run each item and report the result — do not assert they passed.

```bash
# 1. Grep for private strings across the working tree:
while IFS= read -r line; do
  [[ "$line" =~ ^# ]] || [[ -z "$line" ]] && continue
  hits=$(git grep -rn --fixed-strings "$line" -- . 2>/dev/null | grep -v '^Binary')
  [ -n "$hits" ] && echo "HIT [$line]:" && echo "$hits"
done < SCRUB.local.md

# 2. Grep across git history:
git log --all --format="%H %s %b" | \
  grep -F -f <(grep -v '^#' SCRUB.local.md | grep -v '^$') && echo "HISTORY HITS FOUND"

# 3. Confirm .gitignore covers SCRUB.local.md and .env files:
git check-ignore -v SCRUB.local.md config/.env

# 4. Run the test suite:
python -m pytest tests/ -v

# 5. Run a dry-run of the example to verify job definitions build correctly:
python submit.py --mode cpu --manifest examples/docking/jobs.json \
                 --run-prefix verify-v1 --dry-run
```

# Project rules

This repository is public. It generalises a private pipeline. These rules exist
so that nothing from the private side reaches the public side.

## Provenance rule

Code here is written fresh. Nothing is copied from the private reference repos,
including fragments, config values, fixtures, comments, or commit messages.

If a block of logic needs to exist here, retype it in a generic form. Do not
paste and edit. Pasting carries context you stop noticing after the third read,
and that is how identifiers leak.

## What never appears here

By category, since this file is public and cannot list the values themselves.
The literal strings live in `SCRUB.local.md`, which is gitignored.

- Organisation and project names on the compute platform
- Object storage bucket names
- Container group UUIDs, or any live infrastructure identifier
- Container image paths under a private registry account
- Any programme or compound prefix
- Protein target names, in code, docs, tests, or fixtures
- Compound identifiers, SMILES, or structure files from private work
- Search box coordinates from private work, which disclose a binding site
- Cost figures derived from a specific private system

Costs may be discussed as orders of magnitude, with an explicit note that they
scale with system size and must be re-derived.

## Example and test data

Public sources only. The worked example uses a single public PDB entry with a
co-crystallised ligand, so receptor, ligand, and box all come from one citable
place.

Never adapt a fixture from the private repos, even one that looks harmless. Box
coordinates alone are disclosive.

## Design principles this repo encodes

These come from operating the pipeline and should survive refactoring:

1. Completion is the existence of the result artifact in object storage. The
   queue's job status is a proxy and can be wrong in both directions.
2. Resource requests filter the pool of eligible nodes. Over-asking costs
   availability, not money. Default modest.
3. A long stall in allocation is a capacity mismatch. The response is to broaden
   hardware classes or lower the ask, never to retry identically.
4. CPU-bound preparation and scoring do not belong on a GPU node.
5. Work that varies on only one axis of a campaign matrix gets hoisted and
   cached.
6. The run prefix is the provenance key. It is required, never defaulted.
   Reusing one means deliberately extending the same run.

## Before publishing or pushing

Run every check and report each result rather than asserting success:

- Grep all strings in `SCRUB.local.md` across the working tree. Zero hits.
- Grep the same strings across the full git history and all commit messages.
- Confirm `.gitignore` covers `SCRUB.local.md` and any `.env`.
- Test suite passes.
- The example runs from a clean checkout with fresh credentials.

A secret removed in a later commit is still in the history. If one is ever
committed, rotate it and start a fresh repository rather than rewriting history.

## Environment notes

- Never commit a real credential, including a partially redacted one.
  `config/.env.example` carries obviously fake placeholders.
- Tests do not touch the network.
- Image builds are a local or CI step, not something this repo automates against
  a private registry.

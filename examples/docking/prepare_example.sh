#!/usr/bin/env bash
# prepare_example.sh — build the public 1HSG worked example from primary sources.
#
# Receptor: PDB entry 1HSG (HIV-1 protease with co-crystallised indinavir),
# downloaded from RCSB.
# Ligands:  four FDA-approved HIV-1 protease inhibitors, downloaded from
# PubChem by CID.  Each structure is verified against its published InChIKey
# before use, so a silent upstream change cannot corrupt the example.
#
# Search box: the published AutoDock Vina tutorial box for 1HSG
# (https://vina.scripps.edu/tutorial/).
#
# Requirements: curl (or wget), python3, Open Babel (obabel).
#   macOS:  pip install openbabel-wheel
#   Debian: apt install openbabel
#
# Produces one self-contained input directory per job id:
#   data/1hsg__<ligand>/
#     receptor.pdbqt  ligand.pdbqt  box.txt     <- CPU docking (smina)
#     receptor.pdb    ligand.sdf                <- GPU MD (OpenMM)
#
# The indinavir directory additionally keeps ligand_crystal.pdb, the
# co-crystallised pose extracted from 1HSG.  It is NOT a docking input; it is
# the reference for measuring redocking pose RMSD.
#
# Run from the repo root:
#   bash examples/docking/prepare_example.sh

set -euo pipefail
cd "$(dirname "$0")"

PDB_ID="1hsg"
PDB_ID_UPPER="$(echo "$PDB_ID" | tr '[:lower:]' '[:upper:]')"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# name : PubChem CID : expected InChIKey (connectivity + stereochemistry)
LIGANDS="indinavir:5362440:CBVCZFGXHXORBI-PXQQMZJSSA-N
ritonavir:392622:NCDNCNXCDXHOMX-XGKFQTDJSA-N
nelfinavir:64143:QAGYKUNXZHXKMR-HKWSIXNMSA-N
lopinavir:92727:KJHKTHWMRKYKJE-SUGCFTRWSA-N"

fetch() {  # fetch <url> <outfile>
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --max-time 120 "$1" -o "$2"
    else
        wget -q --tries=3 --timeout=120 "$1" -O "$2"
    fi
}

# ── receptor ──────────────────────────────────────────────────────────────────
echo "==> Downloading ${PDB_ID_UPPER}.pdb from RCSB"
fetch "https://files.rcsb.org/download/${PDB_ID_UPPER}.pdb" "${WORK}/${PDB_ID}.pdb"

echo "==> Extracting receptor (protein ATOM records only; no waters, no HETATM)"
grep '^ATOM' "${WORK}/${PDB_ID}.pdb" > "${WORK}/receptor.pdb"
test -s "${WORK}/receptor.pdb" || { echo "ERROR: no ATOM records found"; exit 1; }

echo "==> Converting receptor to PDBQT"
obabel "${WORK}/receptor.pdb" -O "${WORK}/receptor.pdbqt" -xr -h 2>/dev/null
test -s "${WORK}/receptor.pdbqt" || { echo "ERROR: receptor PDBQT conversion failed"; exit 1; }

echo "==> Extracting co-crystal ligand (MK1 = indinavir) as redocking reference"
grep '^HETATM' "${WORK}/${PDB_ID}.pdb" | grep ' MK1 ' > "${WORK}/ligand_crystal.pdb"
echo 'END' >> "${WORK}/ligand_crystal.pdb"

# ── box (published AutoDock Vina tutorial box for 1HSG) ───────────────────────
cat > "${WORK}/box.txt" <<'BOX'
center_x = 16.0
center_y = 26.0
center_z = 4.0
size_x = 20
size_y = 20
size_z = 20
BOX

# ── ligands ───────────────────────────────────────────────────────────────────
echo "$LIGANDS" | while IFS=: read -r name cid want_key; do
    [ -n "$name" ] || continue
    echo "==> ${name} (PubChem CID ${cid})"

    smiles_file="${WORK}/${name}.smi"
    fetch "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${cid}/property/IsomericSMILES/TXT" \
          "$smiles_file"
    smiles="$(tr -d '\r\n' < "$smiles_file")"
    test -n "$smiles" || { echo "ERROR: empty SMILES for ${name}"; exit 1; }

    got_key="$(obabel -:"$smiles" -oinchikey 2>/dev/null | tr -d '[:space:]')"
    if [ "$got_key" != "$want_key" ]; then
        echo "ERROR: InChIKey mismatch for ${name}" >&2
        echo "       expected ${want_key}" >&2
        echo "       got      ${got_key}" >&2
        echo "       PubChem CID ${cid} may have changed; verify before using." >&2
        exit 1
    fi
    echo "    InChIKey verified: ${got_key}"

    out="data/${PDB_ID}__${name}"
    mkdir -p "$out"

    # 3D conformer for docking.  All four ligands are generated the same way so
    # their scores are directly comparable; none starts from a crystal pose.
    obabel -:"$smiles" -O "${out}/ligand.pdbqt" --gen3d -h 2>/dev/null
    obabel -:"$smiles" -O "${out}/ligand.sdf"   --gen3d -h 2>/dev/null
    test -s "${out}/ligand.pdbqt" || { echo "ERROR: ${name} PDBQT generation failed"; exit 1; }
    test -s "${out}/ligand.sdf"   || { echo "ERROR: ${name} SDF generation failed"; exit 1; }

    cp "${WORK}/receptor.pdbqt" "${out}/receptor.pdbqt"
    cp "${WORK}/receptor.pdb"   "${out}/receptor.pdb"
    cp "${WORK}/box.txt"        "${out}/box.txt"

    if [ "$name" = "indinavir" ]; then
        cp "${WORK}/ligand_crystal.pdb" "${out}/ligand_crystal.pdb"
        echo "    kept ligand_crystal.pdb (redocking RMSD reference)"
    fi
    echo "    wrote ${out}/"
done

echo ""
echo "==> Example data ready."
ls -d data/${PDB_ID}__*/ | sed 's/^/    /'
echo ""
echo "    Upload inputs to R2, then submit:"
echo "      python submit.py --mode cpu --manifest examples/docking/jobs.json \\"
echo "                       --run-prefix <your-prefix>"

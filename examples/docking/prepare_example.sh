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
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PDB_ID="1hsg"
PDB_ID_UPPER="$(echo "$PDB_ID" | tr '[:lower:]' '[:upper:]')"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

LIGAND_SPEC="${WORK}/ligands.json"
cat > "$LIGAND_SPEC" <<'JSON'
[
  {"name": "indinavir",  "cid": 5362440, "inchikey": "CBVCZFGXHXORBI-PXQQMZJSSA-N"},
  {"name": "ritonavir",  "cid": 392622,  "inchikey": "NCDNCNXCDXHOMX-XGKFQTDJSA-N"},
  {"name": "nelfinavir", "cid": 64143,   "inchikey": "QAGYKUNXZHXKMR-HKWSIXNMSA-N"},
  {"name": "lopinavir",  "cid": 92727,   "inchikey": "KJHKTHWMRKYKJE-SUGCFTRWSA-N"}
]
JSON

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
# 3D generation and its verification live in prepare_ligands.py.  Every ligand
# is checked twice: the SMILES from PubChem against its published InChIKey, and
# then the generated conformer's own geometry against the same key, so a
# structure that lost stereochemistry cannot reach disk.  All four are built the
# same way, so their scores stay comparable; none starts from a crystal pose.
python3 "${SCRIPT_DIR}/prepare_ligands.py" \
    --ligands "$LIGAND_SPEC" --prefix "$PDB_ID" --data-dir data

# ── assemble one self-contained input set per job id ──────────────────────────
echo "==> Assembling input directories"
for out in data/${PDB_ID}__*/; do
    name="$(basename "$out")"
    test -s "${out}/ligand.sdf" || { echo "ERROR: ${name} has no ligand.sdf"; exit 1; }

    # Docking input is converted from the SAME verified conformer rather than
    # generated independently, so the two files cannot disagree.
    obabel "${out}/ligand.sdf" -O "${out}/ligand.pdbqt" 2>/dev/null
    test -s "${out}/ligand.pdbqt" || { echo "ERROR: ${name} PDBQT conversion failed"; exit 1; }

    cp "${WORK}/receptor.pdbqt" "${out}/receptor.pdbqt"
    cp "${WORK}/receptor.pdb"   "${out}/receptor.pdb"
    cp "${WORK}/box.txt"        "${out}/box.txt"

    if [ "$name" = "${PDB_ID}__indinavir" ]; then
        cp "${WORK}/ligand_crystal.pdb" "${out}/ligand_crystal.pdb"
    fi
    echo "    ${out}"
done

echo ""
echo "==> Example data ready."
ls -d data/${PDB_ID}__*/ | sed 's/^/    /'
echo ""
echo "    Upload inputs to R2, then submit:"
echo "      python submit.py --mode cpu --manifest examples/docking/jobs.json \\"
echo "                       --run-prefix <your-prefix>"

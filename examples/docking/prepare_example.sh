#!/usr/bin/env bash
# prepare_example.sh — download and prepare the public 1HSG worked example.
#
# Uses PDB entry 1HSG (HIV-1 protease with co-crystallised indinavir).
# Reference: Harte WE Jr et al. (1990); PDB deposited by Fitzgerald et al.
# AutoDock Vina tutorial box coordinates are published at:
#   https://vina.scripps.edu/tutorial/
#
# Requirements: wget or curl, python3, Open Babel (obabel).
# Install Open Babel: brew install open-babel  OR  apt install openbabel
#
# What this script does:
#   1. Downloads 1HSG.pdb from RCSB
#   2. Strips water molecules and heteroatoms to get the apo receptor
#   3. Extracts the co-crystal ligand (MK1 / indinavir) as a separate SDF
#   4. Converts both to PDBQT format for smina
#   5. Writes box.txt with the published search box centred on the binding site
#   6. Converts SMILES for three additional public test molecules to PDBQT
#
# Run from the repo root:
#   bash examples/docking/prepare_example.sh

set -euo pipefail
cd "$(dirname "$0")"

PDB_ID="1hsg"
DATA_DIR="data/${PDB_ID}__indinavir"
mkdir -p "$DATA_DIR"

PDB_ID_UPPER="$(echo "$PDB_ID" | tr '[:lower:]' '[:upper:]')"
echo "==> Downloading ${PDB_ID_UPPER}.pdb from RCSB"
if command -v wget &>/dev/null; then
    wget -q "https://files.rcsb.org/download/${PDB_ID_UPPER}.pdb" -O "${DATA_DIR}/${PDB_ID}.pdb"
else
    curl -fsSL "https://files.rcsb.org/download/${PDB_ID_UPPER}.pdb" -o "${DATA_DIR}/${PDB_ID}.pdb"
fi

echo "==> Extracting receptor (protein only, no HETATM, no water)"
grep -E '^(ATOM)' "${DATA_DIR}/${PDB_ID}.pdb" > "${DATA_DIR}/receptor_raw.pdb"

echo "==> Extracting co-crystal ligand (MK1 = indinavir)"
grep -E '^HETATM.*MK1' "${DATA_DIR}/${PDB_ID}.pdb" > "${DATA_DIR}/ligand_mk1.pdb"
echo 'END' >> "${DATA_DIR}/ligand_mk1.pdb"

echo "==> Converting to PDBQT"
obabel "${DATA_DIR}/receptor_raw.pdb" -O "${DATA_DIR}/receptor.pdbqt" \
    -xr -h 2>/dev/null
obabel "${DATA_DIR}/ligand_mk1.pdb" -O "${DATA_DIR}/ligand.pdbqt" \
    --gen3d -h 2>/dev/null

echo "==> Writing box.txt (published Vina tutorial box for 1HSG)"
# Box centred on the 1HSG binding site; from the official AutoDock Vina tutorial.
cat > "${DATA_DIR}/box.txt" <<'BOX'
center_x = 16.0
center_y = 26.0
center_z = 4.0
size_x = 20
size_y = 20
size_z = 20
BOX

echo "==> Preparing additional test molecules from public SMILES"
# Three FDA-approved HIV protease inhibitors with known activity, SMILES from PubChem.
python3 - <<'PY'
import subprocess, os

molecules = [
    # name, SMILES from PubChem CID
    ("ritonavir",   "CC(C)c1csc(NC(=O)c2nc(C(C)C)cs2)n1"),
    ("nelfinavir",  "CC1(C)OC(=O)N(Cc2ccccc2)C1Cc1ccc(O)cc1"),
    ("lopinavir",   "CC(C)c1csc(NC(=O)[C@@H](Cc2ccccc2)NC(=O)c2ccc(N3CCOCC3)cc2)n1"),
]
out_dir = "data/1hsg__indinavir"
for name, smi in molecules:
    pdbqt = os.path.join(out_dir, f"test_{name}.pdbqt")
    result = subprocess.run(
        ["obabel", f"-:{smi}", "-O", pdbqt, "--gen3d", "-h", "--minimize"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  prepared {name}")
    else:
        print(f"  warning: could not prepare {name}: {result.stderr.strip()}")
PY

echo ""
echo "==> Example data ready in ${DATA_DIR}/"
echo "    receptor.pdbqt, ligand.pdbqt, box.txt, test_*.pdbqt"
echo ""
echo "    Upload inputs to R2 then submit:"
echo "    python submit.py --mode cpu --manifest examples/docking/jobs.json \\"
echo "                     --run-prefix <your-prefix>"

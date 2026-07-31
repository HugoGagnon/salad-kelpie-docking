#!/usr/bin/env python3
"""
prepare_ligands.py — fetch and build verified 3D ligand structures.

Called by prepare_example.sh.  Kept in Python rather than shell because every
step here needs to be checked, and a silent failure produces a file that looks
fine and docks to a meaningless number.

For each ligand:
  1. Fetch the isomeric SMILES from PubChem by CID.
  2. Check it against the published InChIKey (connectivity + stereochemistry).
  3. Generate a 3D conformer with RDKit ETKDG, then MMFF-optimise it.
  4. Re-derive the InChIKey *from the generated 3D coordinates* and check it
     again, so a conformer that lost stereochemistry cannot reach disk.

Step 4 is the one that matters.  Open Babel's --gen3d silently emits flat 2D
coordinates for flexible molecules such as lopinavir while still exiting 0;
checking only the input SMILES does not catch it.  PubChem's own 3D records
are used when available but do not exist for the most flexible ligands, which
is precisely the case where local generation is hardest.

Embedding uses a fixed random seed so the committed files are reproducible.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
EMBED_SEED = 42


def fetch(url: str, timeout: int = 120) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def inchikey_from_mol(mol) -> str:
    """InChIKey derived from 3D geometry, not from the source SMILES."""
    from rdkit import Chem
    probe = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(probe)
    return Chem.inchi.MolToInchiKey(probe)


def conformer_from_pubchem(cid: str, want_key: str):
    """PubChem's precomputed 3D record, if it has one and it verifies."""
    from rdkit import Chem
    sdf = fetch(f"{PUBCHEM}/{cid}/SDF?record_type=3d")
    if not sdf:
        return None
    mol = Chem.MolFromMolBlock(sdf, removeHs=False)
    if mol is None:
        return None
    mol = Chem.AddHs(mol, addCoords=True)
    return mol if inchikey_from_mol(mol) == want_key else None


def conformer_from_smiles(smiles: str, want_key: str):
    """Locally embedded conformer via ETKDG, verified after optimisation."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return mol if inchikey_from_mol(mol) == want_key else None


def build(name: str, cid: str, want_key: str, out_dir: Path) -> None:
    from rdkit import Chem

    smiles = fetch(f"{PUBCHEM}/{cid}/property/IsomericSMILES/TXT")
    if not smiles:
        sys.exit(f"ERROR: could not fetch SMILES for {name} (CID {cid})")
    smiles = smiles.strip()

    key_from_smiles = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))
    if key_from_smiles != want_key:
        sys.exit(
            f"ERROR: PubChem SMILES for {name} does not match the published "
            f"structure\n       expected {want_key}\n       got      {key_from_smiles}\n"
            f"       CID {cid} may have changed; verify before using."
        )
    print(f"    SMILES verified: {key_from_smiles}")

    mol, source = conformer_from_pubchem(cid, want_key), "PubChem 3D record"
    if mol is None:
        mol, source = conformer_from_smiles(smiles, want_key), "RDKit ETKDG"
    if mol is None:
        sys.exit(
            f"ERROR: no 3D conformer of {name} reproduced {want_key}. "
            f"Refusing to write a structure whose stereochemistry is wrong."
        )

    conf = mol.GetConformer()
    zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    if max(zs) - min(zs) < 1.0:
        sys.exit(f"ERROR: {name} conformer is flat ({max(zs) - min(zs):.2f} A in z)")

    out_dir.mkdir(parents=True, exist_ok=True)
    Chem.MolToMolFile(mol, str(out_dir / "ligand.sdf"))
    print(f"    3D conformer verified via {source} "
          f"({mol.GetNumAtoms()} atoms, {max(zs) - min(zs):.1f} A in z)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build verified 3D ligand structures")
    ap.add_argument("--ligands", required=True,
                    help="JSON list of {name, cid, inchikey}")
    ap.add_argument("--prefix", required=True, help="target name prefix, e.g. 1hsg")
    ap.add_argument("--data-dir", required=True, help="output root")
    args = ap.parse_args()

    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        sys.exit("ERROR: RDKit is required.  Install it with:  pip install rdkit")

    for spec in json.loads(Path(args.ligands).read_text()):
        print(f"==> {spec['name']} (PubChem CID {spec['cid']})")
        build(spec["name"], str(spec["cid"]), spec["inchikey"],
              Path(args.data_dir) / f"{args.prefix}__{spec['name']}")


if __name__ == "__main__":
    main()

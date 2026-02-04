#!/usr/bin/env python3
"""
Generate reference data for PDB loading parity testing.
Uses la-proteina's PDB loading to compare with Julia implementation.
"""

import sys
la_proteina = os.environ.get('LA_PROTEINA_PATH', '')
if la_proteina: sys.path.insert(0, la_proteina)

import numpy as np
from pathlib import Path

# Import from openfold's protein utilities
from openfold.np import protein as protein_np
from openfold.np import residue_constants

# Use a sample PDB file
pdb_path = Path(__file__).parent / "samples_gpu" / "gpu_sde_1.pdb"
output_dir = Path(__file__).parent / "data"

print(f"Loading PDB from {pdb_path}")

# Read PDB file
with open(pdb_path, 'r') as f:
    pdb_str = f.read()

# Parse using OpenFold's protein utilities
protein = protein_np.from_pdb_string(pdb_str)

print(f"\nProtein structure loaded:")
print(f"  aatype shape: {protein.aatype.shape}")  # [L]
print(f"  atom_positions shape: {protein.atom_positions.shape}")  # [L, 37, 3]
print(f"  atom_mask shape: {protein.atom_mask.shape}")  # [L, 37]

# Get coordinates in nanometers
coords_angstrom = protein.atom_positions  # [L, 37, 3]
coords_nm = coords_angstrom / 10.0  # Convert to nm

# Get amino acid types (0-19 for standard amino acids)
aatype = protein.aatype  # [L]

# Get atom mask
atom_mask = protein.atom_mask  # [L, 37]

# Get residue mask (all valid for now)
residue_mask = np.ones(len(aatype), dtype=np.float32)

# Get sequence string
sequence = ''.join([residue_constants.restypes[aa] for aa in aatype])

print(f"\nData:")
print(f"  coords_nm shape: {coords_nm.shape}")
print(f"  aatype: {aatype}")
print(f"  sequence: {sequence}")
print(f"  Length: {len(aatype)}")

# Save reference data
np.save(output_dir / "pdb_coords_nm.npy", coords_nm.astype(np.float32))
np.save(output_dir / "pdb_aatype.npy", aatype)
np.save(output_dir / "pdb_atom_mask.npy", atom_mask.astype(np.float32))
np.save(output_dir / "pdb_residue_mask.npy", residue_mask)

# Save CA coordinates separately for easier comparison
ca_idx = 1  # CA is index 1 in atom37
ca_coords = coords_nm[:, ca_idx, :]  # [L, 3]
np.save(output_dir / "pdb_ca_coords_nm.npy", ca_coords.astype(np.float32))

print(f"\nSaved reference data to {output_dir}")
print(f"  pdb_coords_nm.npy: {coords_nm.shape}")
print(f"  pdb_aatype.npy: {aatype.shape}")
print(f"  pdb_atom_mask.npy: {atom_mask.shape}")
print(f"  pdb_ca_coords_nm.npy: {ca_coords.shape}")

# Print sample values for verification
print(f"\nSample CA coordinates (first 3 residues):")
for i in range(min(3, len(aatype))):
    print(f"  res {i}: {ca_coords[i]}")

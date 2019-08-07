# MOF_scripts

#### Gen_top.py

Generte itp file with parameters from UFF

#### trim_pdb.py

Remove atoms outside of defined box from input pdb.

#### sort_by_resid.py

Assign new residue id to molecules according to the connection in the structure.

## An example of using the scripts

### Get MOF structures from Chimera using the cif file of a unit cell

cif file: 668974_no_H2O_with_H_OpenMetalSite.cif

Make copies to generate 1x1x1 unit cell.

### save the MOF structure into a pdb file 

Write PDB to chimera_111.pdb

### put atoms together and get rid of CONECT information
awk '{if ($1=="HETATM") {print}}' chimera_111.pdb > chimera_111.merge.pdb

### add missing atoms to make the ligand whole molecules
python3 add_missing_atoms.py --pdbin chimera_111.merge.pdb --pdbout chimera_111.full.pdb

### sort the atoms in pdb file by residue number
python3 sort_by_resid.py --pdbin chimera_111.full.pdb --pdbout chimera_111.full.new_resid.pdb --idmap chimera_111.full.idmap

### generate topology file for the pdb
python3 Gen_top.py --pdbin chimera_111.full.new_resid.pdb --out_path . --itp chimera_111.full.new_resid.itp --top chimera_111.full.new_resid.top --params ../UFF_params.0806.rename.txt

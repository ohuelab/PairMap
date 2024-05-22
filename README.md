# PairMap

PairMap is a comprehensive tool for the calculation of relative binding free energies in complex compound transformations. It involves the exhaustive generation of intermediates and the construction of intermediate-induced perturbation maps.

## Installation


## Dependencies

PairMap requires the following dependencies:

- lomap2
- networkx
- numpy
- rdkit>=2021.03.1
- tqdm

These dependencies will be automatically installed when you install PairMap using pip.

## Usage

```python
from pairmap import IntermediateGenerator
source_ligand = Chem.MolFromMolFile("/path/to/source_ligand.mol")
target_ligand = Chem.MolFromMolFile("/path/to/target_ligand.mol")

# Generate intermediates
searchIntm = SearchIntermediates(source_ligand, target_ligand)
intermediates = searchIntm.generate_intermediates()

# ligand preparation with openbabel and extract same formal charge
obabel_path = "/path/to/obabel"
intermediates_avail = execute_ligand_preparation(intermediates, obabel_path = obabel_path, extract_same_formal_charge=True)

# Generate map
mapGen = MapGenerator(intermediates_avail, maxPathLength=4, cycleLength=3, maxOptimalPathLength=3, jobs=-1)

```

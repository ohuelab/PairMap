import copy

from rdkit import Chem
from rdkit.Chem import MolStandardize

import logging
logging.disable(logging.INFO)

HYDROGEN_ATOM = Chem.Atom(1)
CARBON_ATOM = Chem.Atom(6)

class IntermediateGenerator:
    def __init__(self, is_atom_modfication_enabled = True, cap_ring_with_carbon = True, cap_ring_with_hydrogen = True, verbose = False):
        '''
        :param is_atom_modfication_enabled: Whether to enable atom modification.
        :param cap_ring_with_carbon: Whether to cap rings with carbon atoms.
        :param cap_ring_with_hydrogen: Whether to cap rings with hydrogen atoms.
        :param verbose: Whether to print verbose output.
        '''
        if not cap_ring_with_carbon and not cap_ring_with_hydrogen:
            raise ValueError('At least one of the options cap_ring_with_carbon and cap_ring_with_hydrogen must be True.')

        self.is_atom_modfication_enabled = is_atom_modfication_enabled
        self.cap_ring_with_carbon = cap_ring_with_carbon
        self.cap_ring_with_hydrogen = cap_ring_with_hydrogen
        self.verbose = verbose
        # TODO: is this necessary?
        self.lfc = MolStandardize.rdMolStandardize.LargestFragmentChooser()

    @staticmethod
    def remove_props(mol):
        '''Remove all properties from a molecule'''
        for prop in mol.GetPropsAsDict():
            mol.ClearProp(prop)

    @staticmethod
    def remove_atom_map(mol):
        '''Remove all atom map from a molecule'''
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return mol

    def postprocess_ligand(self, ligand):
        '''Postprocess a molecule'''
        try:
            self.remove_props(ligand)
            # ligand = self.lfc.choose(ligand)
            Chem.SanitizeMol(ligand)
            ligand = self.remove_atom_map(ligand)
            ligand = Chem.RemoveHs(ligand)
            return ligand
        except:
            return None

    @staticmethod
    def get_atom_idx_by_map_num(mol, mapnum):
        '''Get atom index from atom map number'''
        indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == mapnum]
        return indices[0] if len(indices)==1 else None

    def get_terminal_rings(self, rings):
        '''Identify rings with exactly one neighboring atom outside the ring.

        # :param rings: List of rings, where each ring is a list of atom indices.
        # :return: List of rings each having exactly one neighbor outside the ring.
        '''

        ligand = self.source_ligand
        terminal_rings = []
        for ring in rings:
            external_neighbor_indices = []
            for idx in ring:
                atom = ligand.GetAtomWithIdx(idx)
                for neighbor_atom in atom.GetNeighbors():
                    neighbor_idx = neighbor_atom.GetIdx()
                    if (neighbor_idx not in ring) and neighbor_idx not in external_neighbor_indices:
                        external_neighbor_indices.append(neighbor_idx)
            # Add the ring to the result if it has exactly one external neighbor
            if len(external_neighbor_indices) == 1:
                terminal_rings.append(ring)
        return terminal_rings

    def get_fused_rings(self, rings):
        """
        Get the fused rings from a ligand and a list of rings.

        :param rings: List of rings, where each ring is a list of atom indices.
        :return: List of unique rings that are edge-fused with other rings.
        """

        ligand = self.source_ligand
        fused_rings = []
        for i in range(len(rings)):
            for j in range(0, len(rings)):
                if i == j:
                    continue
                ring1 = set(rings[i])
                ring2 = set(rings[j])
                if len(ring1.intersection(ring2)) > 0:
                    ring_diff = ring1.difference(ring2)
                    fused_ring = set()
                    for idx in ring_diff:
                        atom = ligand.GetAtomWithIdx(idx)
                        for neighbor_atom in atom.GetNeighbors():
                            neighbor_idx = neighbor_atom.GetIdx()
                            if (neighbor_idx not in ring1):
                                fused_ring.add(idx)
                    fused_rings+=[list(ring_diff)]
        return fused_rings

    def handle_deletable_entities(self, atom, terminal_rings, fused_rings):
        '''Identify deletable atoms and rings.

        :param atom: Current atom being analyzed.
        :param terminal_rings: List of terminal rings in the ligand.
        :param fused_rings: List of fused rings in the ligand.
        :return: Tuple containing lists of deletable atoms, rings, and fused rings.
        '''
        ligand = self.source_ligand
        deletable_atoms = []
        deletable_rings = []
        deletable_fused_rings = []
        atom_idx = atom.GetIdx()

        if not atom.IsInRing():
            if atom.GetDegree() == 1:
                deletable_atoms.append(atom)
        else:
            for ring in terminal_rings:
                if atom_idx in ring:
                    atom_ring = [ligand.GetAtomWithIdx(idx) for idx in ring]
                    deletable_rings.append(atom_ring)
                    terminal_rings.remove(ring)
            for ring in fused_rings:
                if atom_idx in ring:
                    atom_ring = [ligand.GetAtomWithIdx(idx) for idx in ring]
                    deletable_fused_rings.append(atom_ring)
                    fused_rings.remove(ring)

        return deletable_atoms, deletable_rings, deletable_fused_rings

    def extract_atoms_for_modification_and_deletion(self):
        '''Extract atoms from the source ligand that can be modified or deleted to transform into the target ligand.

        :return: Tuple containing lists of modifiable atoms:
            - atoms_for_modification: Atoms that can be modified (changed to another atom).
            - atoms_for_deletion: Atoms that can be deleted.
            - rings_for_deletion: Rings that can be deleted.
            - fused_rings_for_deletion: Fused rings that can be deleted.
        '''

        source_ligand,target_ligand, mcs_map = self.source_ligand, self.target_ligand, self.mcs_map

        source_rings = list(source_ligand.GetRingInfo().AtomRings())
        terminal_rings = self.get_terminal_rings(source_rings)
        fused_rings = self.get_fused_rings(source_rings)

        atoms_for_modification = []
        atoms_for_deletion = []
        rings_for_deletion = []
        fused_rings_for_deletion = []

        for atom in source_ligand.GetAtoms():
            source_atom_idx = atom.GetIdx()
            if source_atom_idx in mcs_map:
                target_atom_idx = mcs_map[source_atom_idx]
                target_atom = target_ligand.GetAtomWithIdx(target_atom_idx)
                # If atom symbol is different, it can be modified
                if atom.GetSymbol() != target_atom.GetSymbol():
                    atoms_for_modification.append(atom)
            else:
                # Handle deletable atoms and rings
                deletable_atoms, deletable_rings, deletable_fused_rings = self.handle_deletable_entities(
                    atom, terminal_rings, fused_rings)
                atoms_for_deletion.extend(deletable_atoms)
                rings_for_deletion.extend(deletable_rings)
                fused_rings_for_deletion.extend(deletable_fused_rings)
        return (
            atoms_for_modification,
            atoms_for_deletion,
            rings_for_deletion,
            fused_rings_for_deletion
        )

    def generate_atom_modification_intermediate(self, atom_map_num):
        '''Generate an intermediate ligand by modifying a specific atom in the source ligand to match the target ligand.

        :param atom_map_num: The mapping number of the atom to be modified.
        :return: RDKit molecule object of the intermediate ligand.
        '''

        source_ligand, target_ligand, mcs_map = self.source_ligand, self.target_ligand, self.mcs_map

        intermediate_ligand = copy.deepcopy(source_ligand)
        source_atom_idx = self.get_atom_idx_by_map_num(intermediate_ligand, atom_map_num)
        target_atom_idx = mcs_map[source_atom_idx]
        target_atomic_num = target_ligand.GetAtomWithIdx(target_atom_idx).GetAtomicNum()

        intermediate_ligand.ReplaceAtom(source_atom_idx, Chem.Atom(target_atomic_num))
        return intermediate_ligand

    def generate_atom_deletion_intermediate(self, atom_map_num):
        '''Generate an intermediate ligand by deleting a specific atom from the source ligand.

        :param atom_map_num: The mapping number of the atom to be deleted.
        :return: RDKit molecule object of the intermediate ligand.
        '''

        source_ligand = self.source_ligand

        intermediate_ligand = copy.deepcopy(source_ligand)
        atom_idx = self.get_atom_idx_by_map_num(intermediate_ligand, atom_map_num)
        atom_to_delete = intermediate_ligand.GetAtomWithIdx(atom_idx)
        assert len(atom_to_delete.GetBonds()) == 1
        bond_to_remove = atom_to_delete.GetBonds()[0]

        if bond_to_remove.GetBondType() == Chem.rdchem.BondType.SINGLE:
            intermediate_ligand.ReplaceAtom(atom_idx, HYDROGEN_ATOM)
        else:
            intermediate_ligand.RemoveAtom(atom_idx)

        return intermediate_ligand

    def generate_ring_deletion_intermediate(self, ring_atom_map_nums):
        '''Generate an intermediate ligand by deleting a specific ring structure from the source ligand.

        :param ring_atom_map_nums: List of mapping numbers for the atoms in the ring to be deleted.
        :return: RDKit molecule object of the intermediate ligand.
        '''

        source_ligand = self.source_ligand
        intermediate_ligand = copy.deepcopy(source_ligand)
        atoms_to_cap = []

        for map_num in ring_atom_map_nums:
            atom_idx = self.get_atom_idx_by_map_num(intermediate_ligand, map_num)
            neighbor_indices_outside_ring = self.get_neighbor_indices_outside_ring(atom_idx, ring_atom_map_nums, intermediate_ligand)

            if neighbor_indices_outside_ring:
                atoms_to_cap.append(map_num)
            else:
                intermediate_ligand.RemoveAtom(atom_idx)

        # DEBUG: Ensure that all atoms in the ring are to be capped
        assert len(set(atoms_to_cap)) == 1
        map_num = atoms_to_cap[0]
        atom_idx = self.get_atom_idx_by_map_num(intermediate_ligand, map_num)
        intermediate_ligands = []

        # Cap the ring with a carbon atom
        if self.cap_ring_with_carbon:
            new_intermediate_ligand = copy.deepcopy(intermediate_ligand)
            new_intermediate_ligand.ReplaceAtom(atom_idx, CARBON_ATOM)
            intermediate_ligands.append(new_intermediate_ligand)

        # Cap the ring with a hydrogen atom
        if self.cap_ring_with_hydrogen:
            new_intermediate_ligand = copy.deepcopy(intermediate_ligand)
            new_intermediate_ligand.ReplaceAtom(atom_idx, HYDROGEN_ATOM)
            intermediate_ligands.append(new_intermediate_ligand)

        return intermediate_ligands

    def generate_fused_ring_deletion_intermediate(self, ring_atom_map_nums):
        '''Generate multiple intermediate ligands by deleting a fused ring structure from the source ligand and
        ensuring correct bonding in the remaining structure.

        :param ring_atom_map_nums: List of mapping numbers for the atoms in the fused ring to be deleted.
        :return: List of RDKit molecule objects for each possible intermediate ligand.
        '''

        source_ligand = self.source_ligand
        intermediate_ligand = copy.deepcopy(source_ligand)
        neighbor_map_nums = []

        for map_num in ring_atom_map_nums:
            atom_idx = self.get_atom_idx_by_map_num(intermediate_ligand, map_num)
            neighbor_indices_outside_ring = self.get_neighbor_indices_outside_ring(atom_idx, ring_atom_map_nums, intermediate_ligand)

            if neighbor_indices_outside_ring:
                neighbor_map_nums.extend([intermediate_ligand.GetAtomWithIdx(idx).GetAtomMapNum() for idx in neighbor_indices_outside_ring])
            intermediate_ligand.RemoveAtom(atom_idx)

        neighbor_map_nums = list(set(neighbor_map_nums))
        intermediate_ligands = [intermediate_ligand]

        # TODO: check aromaticity of another ring
        for i, map_num_i in enumerate(neighbor_map_nums):
            for map_num_j in neighbor_map_nums[i+1:]:
                idx_i = self.get_atom_idx_by_map_num(intermediate_ligand, map_num_i)
                idx_j = self.get_atom_idx_by_map_num(intermediate_ligand, map_num_j)
                if intermediate_ligand.GetBondBetweenAtoms(idx_i, idx_j):
                    new_intermdiate_ligands = []
                    for intermdiate_ligand in intermediate_ligands:
                        new_intermdiate_ligand = copy.deepcopy(intermdiate_ligand)
                        # convert bond to single bond
                        bond = new_intermdiate_ligand.GetBondBetweenAtoms(idx_i, idx_j)
                        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                        bond.SetIsAromatic(False)
                        new_intermdiate_ligand.GetAtomWithIdx(idx_i).SetIsAromatic(False)
                        new_intermdiate_ligand.GetAtomWithIdx(idx_j).SetIsAromatic(False)
                        new_intermdiate_ligands.append(new_intermdiate_ligand)
                    intermediate_ligands.extend(new_intermdiate_ligands)
        return intermediate_ligands

    @staticmethod
    def get_neighbor_indices_outside_ring(atom_idx, ring_atom_map_nums, ligand):
        '''Get indices of neighboring atoms that are not part of the specified ring.

        :param
        :param ring_atom_map_nums: List of atom mapping numbers in the ring.
        :param ligand: RDKit mol object.
        :return: List of indices for neighboring atoms outside the ring.
        '''

        atom_neighbors = ligand.GetAtomWithIdx(atom_idx).GetNeighbors()
        neighbor_indices_outside_ring = []
        for neighbor_atom in atom_neighbors:
            neighbor_idx = neighbor_atom.GetIdx()
            neighbor_map_num = ligand.GetAtomWithIdx(neighbor_idx).GetAtomMapNum()
            if neighbor_map_num not in ring_atom_map_nums:
                neighbor_indices_outside_ring.append(neighbor_idx)

        return neighbor_indices_outside_ring


    def generate_intermediates(self, source_ligand, target_ligand, mcs_map):
        '''Generate intermediates for transforming the source ligand into the target ligand.
        :param source_ligand: RDKit molecule object of the source ligand.
        :param target_ligand: RDKit molecule object of the target ligand.
        :param mcs_map: Dictionary mapping atom indices in the source ligand to the target ligand.
        :return: List of RDKit molecule objects generated as intermediates.
        '''

        # preprocess ligands
        self.source_ligand = source_ligand
        self.target_ligand = target_ligand
        for i, atom in enumerate(source_ligand.GetAtoms(), start=1):
            atom.SetAtomMapNum(i)
        self.mcs_map = mcs_map

        # Extract atoms and rings eligible for modification or deletion
        atoms_for_modification, atoms_for_deletion, rings_for_deletion, fused_rings_for_deletion = self.extract_atoms_for_modification_and_deletion()
        if self.verbose:
            print('atoms_for_modification:', len(atoms_for_modification))
            print('atoms_for_deletion:', len(atoms_for_deletion))
            print('rings_for_deletion:', len(rings_for_deletion))
            print('fused_rings_for_deletion:', len(fused_rings_for_deletion))
        # Convert atoms and rings to their map numbers
        atom_map_nums_for_modification = [atom.GetAtomMapNum() for atom in atoms_for_modification]
        atom_map_nums_for_deletion = [atom.GetAtomMapNum() for atom in atoms_for_deletion]
        ring_atom_map_nums_list_for_deletion = [[atom.GetAtomMapNum() for atom in ring] for ring in rings_for_deletion]
        fused_ring_atom_map_nums_list_for_deletion = [[atom.GetAtomMapNum() for atom in ring] for ring in fused_rings_for_deletion]

        # Generate intermediates
        intermediates_list = []
        if self.is_atom_modfication_enabled:
            # Generate intermediates for atom modification
            intermediates_list.extend([[self.generate_atom_modification_intermediate(atom_map_num)] for atom_map_num in atom_map_nums_for_modification])

        # Generate intermediates for atom deletion
        intermediates_list.extend([[self.generate_atom_deletion_intermediate(atom_map_num)] for atom_map_num in atom_map_nums_for_deletion])

        # Generate intermediates for ring deletion
        intermediates_list.extend([self.generate_ring_deletion_intermediate(ring_atom_map_nums) for ring_atom_map_nums in ring_atom_map_nums_list_for_deletion])

        # Generate intermediates for fused ring deletion
        intermediates_list.extend([self.generate_fused_ring_deletion_intermediate(fused_ring_atom_map_nums) for fused_ring_atom_map_nums in fused_ring_atom_map_nums_list_for_deletion])

        # Postprocess intermediates and remove None values
        intermediates = [self.postprocess_ligand(intermediate) for intermediates in intermediates_list for intermediate in intermediates]
        intermediates = [intermediate for intermediate in intermediates if intermediate is not None]

        return intermediates

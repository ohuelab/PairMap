# openbabel ligand preparation
import os
from rdkit import Chem
import subprocess
from .mcs import formal_charge

def execute_ligand_preparation(mols, input_file='ligand.sdf', output_file='ligand_prepared.sdf', pH = 7.4, remove_files=True, override=False, obabel_path='obabel', extract_same_formal_charge = False, charge_indices = [0, 1]):
    """Execute ligand preparation using obabel.

    :param mols: A list of molecules.
    :param input_file: The file name to write the input molecules.
    :param output_file: The file name to write the ligand prepared molecules.
    :param pH: The pH value for the ligand preparation.
    :param remove_files: Whether to remove the input and output files after the preparation.
    :param override: Whether to override the input and output files if they already exist.
    :param obabel_path: The path to the obabel executable.
    :param extract_same_formal_charge: Whether to extract the molecules with the same formal charge to the molecules in charge_indices.
    :param charge_indices: charges must be the same as the first molecule.
    :return: A list of prepared molecules.
    """
    if not override and os.path.exists(input_file):
        raise FileExistsError('Input file already exists. Set override=True to overwrite.')
    if not override and os.path.exists(output_file):
        raise FileExistsError('Output file already exists. Set override=True to overwrite.')
    try:
        with Chem.SDWriter(input_file) as writer:
            for mol in mols:
                writer.write(mol)
        try:
            subprocess.run([obabel_path, input_file, '-O', output_file, '-p', str(pH)], check=True)
        except:
            raise RuntimeError('obabel failed. Please check if obabel is installed and in your PATH.')
        prepared_mols = Chem.SDMolSupplier(output_file)
        if remove_files and os.path.exists(input_file):
            os.remove(input_file)
        if remove_files and os.path.exists(output_file):
            os.remove(output_file)
    except:
        if remove_files and os.path.exists(input_file):
            os.remove(input_file)
        if remove_files and os.path.exists(output_file):
            os.remove(output_file)
        raise RuntimeError('RDKit failed to write the input file.')
    if extract_same_formal_charge:
        formal_charges = [formal_charge(mol) for mol in prepared_mols]
        if isinstance(charge_indices, int):
            charge_indices = [charge_indices]
        if not isinstance(charge_indices, list):
            raise ValueError('charge_indices must be a list of integers or a single integer.')
        if len(charge_indices) == 0:
            raise ValueError('charge_indices must not be empty.')
        if not all(isinstance(i, int) for i in charge_indices):
            raise ValueError('charge_indices must be a list of integers or a single integer.')

        base_charge = formal_charges[0]
        if not all(formal_charges[i] == base_charge for i in charge_indices):
            raise ValueError('Formal charges of the molecules are not the same.')
        prepared_mols = [mol for mol, charge in zip(prepared_mols, formal_charges) if charge == base_charge]
        print('Extracted {} molecules with the same formal charge from {} molecules.'.format(len(prepared_mols), len(mols)))
    return list(prepared_mols)

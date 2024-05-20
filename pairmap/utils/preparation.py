# openbabel ligand preparation
import os
from rdkit import Chem
import subprocess

def execute_ligand_preparation(mols, input_file='ligand.sdf', output_file='ligand_prepared.sdf', remove_files=True, override=False, obabel_path='obabel'):
    """Execute ligand preparation using obabel.

    :param mols: A list of molecules.
    :return: A list of prepared molecules.
    """
    if not override and os.path.exists(input_file):
        raise FileExistsError('Input file already exists. Set override=True to overwrite.')
    if not override and os.path.exists(output_file):
        raise FileExistsError('Output file already exists. Set override=True to overwrite.')
    with Chem.SDWriter(input_file) as writer:
        for mol in mols:
            writer.write(mol)
    try:
        subprocess.run([obabel_path, input_file, '-O', output_file, '-p', '7.4'], check=True)
    except:
        raise RuntimeError('obabel failed. Please check if obabel is installed and in your PATH.')
    prepared_mols = Chem.SDMolSupplier(output_file)
    if remove_files:
        os.remove(input_file)
        os.remove(output_file)
    return list(prepared_mols)

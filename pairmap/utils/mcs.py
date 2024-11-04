import numpy as np
import itertools
from lomap.mcs import MCS
from rdkit.Chem import rdFMCS
import copy

from tqdm import tqdm

from multiprocessing import Pool



def formal_charge(mol):
    total_charge_mol = 0.0

    try:
        # Assume mol2
        total_charge_mol=sum([float(a.GetProp('_TriposPartialCharge')) for a in mol.GetAtoms()])
    except:
        # wasn't mol2, so assume SDF with correct formal charge props for mols
        total_charge_mol=sum([a.GetFormalCharge() for a in mol.GetAtoms()])

    return total_charge_mol

def ecr(mol_i, mol_j):

    total_charge_mol_i = formal_charge(mol_i)
    total_charge_mol_j = formal_charge(mol_j)

    if abs(total_charge_mol_j - total_charge_mol_i) < 1e-3:
        scr_ecr = 1.0
    else:
        scr_ecr = 0.0

    return scr_ecr

def score_function(mola, molb, options = None):
    """Calculate the score of two molecules based on various rules.

    Args:
        mola (Molecule): The first molecule.
        molb (Molecule): The second molecule.
        options (dict, optional): Additional options for the scoring. Defaults to None.

    Raises:
        ValueError: If the molecules are not valid.

    Returns:
        tuple: A tuple containing the MCS object and the calculated score.
    """
    mola = copy.deepcopy(mola)
    molb = copy.deepcopy(molb)
    ecr_score = ecr(mola,molb)
    if options is None:
        options = {'time':20, 'verbose':'info', 'max3d':0, 'threed':False}
    MC = MCS(mola, molb, **options)
    tmp_scr = ecr_score * MC.mncar() * MC.mcsr() * MC.atomic_number_rule() * MC.hybridization_rule()
    tmp_scr *= MC.sulfonamides_rule() * MC.heterocycles_rule()
    tmp_scr *= MC.transmuting_methyl_into_ring_rule()
    tmp_scr *= MC.transmuting_ring_sizes_rule()
    return MC, tmp_scr


def compute_score(pair):
    i, j, mols, options = pair
    try:
        _, score = score_function(mols[i], mols[j], options)
    except:
        _, score = score_function(mols[i], mols[j], options)
    return i, j, score


def get_score_matrix(mols, options = None, use_seed = True, jobs = 1):
    if options is None:
        options = {}
    if use_seed:
        mcs = calc_mcs(mols)
        seedSmarts = mcs.smartsString
    else:
        seedSmarts = ""

    N=len(mols)
    score_matrix = np.zeros((N, N))
    pairs = [(i, j, mols, {**options, "seed": seedSmarts}) for i, j in itertools.combinations(range(N), 2)]

    if jobs == 1 or jobs == 0:
        for i, j, score in tqdm(map(compute_score, pairs), total=len(pairs)):
            score_matrix[i][j]=score
            score_matrix[j][i]=score
    else:
        if jobs < 0:
            jobs = None
        with Pool(jobs) as pool:
            for i, j, score in tqdm(pool.imap_unordered(compute_score, pairs), total=len(pairs)):
                score_matrix[i][j]=score
                score_matrix[j][i]=score

    # with Pool() as pool:
    #     for i, j, score in tqdm(pool.imap_unordered(compute_score, pairs), total=len(pairs)):
    #         score_matrix[i][j]=score
    #         score_matrix[j][i]=score
    return score_matrix


def calc_mcs(mols):
    mcs = rdFMCS.FindMCS(mols,
                    timeout=1200,
                    atomCompare=rdFMCS.AtomCompare.CompareAny,
                    bondCompare=rdFMCS.BondCompare.CompareAny,
                    matchValences=False,
                    ringMatchesRingOnly=True,
                    completeRingsOnly=True,
                    matchChiralTag=False)
    return mcs

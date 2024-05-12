from utils import score_function

import itertools
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

from multiprocessing import Pool
def compute_score(pair):
    i, j, mols, options, seedSmarts = pair
    try:
        _, score = score_function(mols[i], mols[j], **options, seed = seedSmarts)
    except:
        _, score = score_function(mols[i], mols[j], **options, seed = '')
    return i, j, score

def get_scoremap(intermediate_list, options):
    mols = [intermediate['ligand'] for intermediate in intermediate_list]

    mcs = calc_mcs(mols)
    seedSmarts = mcs.smartsString
    N=len(mols)
    scoremap = np.zeros((N, N))
    pairs = [(i, j, mols, options, seedSmarts) for i, j in itertools.combinations(range(N), 2)]

    with Pool() as pool:
        for i, j, score in tqdm(pool.imap_unordered(compute_score, pairs), total=len(pairs)):
            scoremap[i][j]=score
            scoremap[j][i]=score
    return scoremap


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

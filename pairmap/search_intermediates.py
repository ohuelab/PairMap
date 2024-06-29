from .intermediate_generator import IntermediateGenerator
from .utils.preparation import execute_ligand_preparation
from .utils.mcs import formal_charge

from lomap.mcs import MCS

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import RWMol
from collections import deque
import copy

class SearchIntermediates:
    def __init__(self, source_ligand, target_ligand, verbose=False, cap_ring_with_carbon=True, cap_ring_with_hydrogen=True, no_backward_search=False, intermediate_name_prefix='intermediate', use_seed = True, score_config=None, ionize=False, obabel_path='obabel'):
        self.source_ligand = RWMol(AllChem.RemoveHs(source_ligand))
        self.target_ligand = RWMol(AllChem.RemoveHs(target_ligand))

        self.verbose = verbose
        self.no_backward_search = no_backward_search
        self.use_seed = use_seed
        self.intermediate_name_prefix = intermediate_name_prefix
        self.score_config = score_config if score_config is not None else {}

        self.ionize = ionize
        self.obabel_path = obabel_path
        if self.ionize and formal_charge(self.source_ligand) != formal_charge(self.target_ligand):
            raise ValueError("Formal charges of the two ligands must be the same")
        self.formal_charge = formal_charge(self.source_ligand)


        self.generator = IntermediateGenerator(cap_ring_with_carbon=cap_ring_with_carbon, cap_ring_with_hydrogen=cap_ring_with_hydrogen, verbose=verbose)

        self.baseMC = self.MCS(source_ligand, target_ligand)
        mcs = self.baseMC.mcs_mol
        self.seedSmarts = Chem.MolToSmarts(mcs) if self.use_seed else ''

    def MCS(self, source_ligand, target_ligand):
        try:
            MC = MCS(source_ligand, target_ligand, **self.score_config, seed = self.seedSmarts)
        except:
            MC = MCS(source_ligand, target_ligand, **self.score_config, seed = '')
        return MC


    def simplex_search(self, direction='forward'):
        if direction=='forward':
            source_ligand = copy.deepcopy(self.source_ligand)
            target_ligand = copy.deepcopy(self.target_ligand)
        else:
            source_ligand = copy.deepcopy(self.target_ligand)
            target_ligand = copy.deepcopy(self.source_ligand)

        q = deque()
        q.append(source_ligand)
        intermediate_info_list = [self.get_intermediate_info(source_ligand)]
        intermediate_info_list += [self.get_intermediate_info(target_ligand)]
        source_index = 0
        target_index = 1
        smiles_list = [intermediate_info_list[source_index]['smiles'], intermediate_info_list[target_index]['smiles']]
        traces=[[source_index, target_index]]
        while len(q)>0:
            ligand = q.popleft()
            ligand = RWMol(ligand)
            smiles = Chem.MolToSmiles(ligand)
            ligand_index = smiles_list.index(smiles)
            MC = self.MCS(ligand, target_ligand)
            mcs_map = {a1:a2 for a1,a2 in MC.heavy_atom_mcs_map()}
            intermediates = self.generator.generate_intermediates(ligand, target_ligand, mcs_map)
            for intermediate in intermediates:
                info = self.get_intermediate_info(intermediate)
                if info['smiles'] not in smiles_list:
                    if self.verbose:
                        print('intermediate: {}'.format(info['smiles']))
                    smiles_list.append(info['smiles'])
                    intermediate_info_list.append(info)
                    q.append(intermediate)
                intermediate_index = smiles_list.index(info['smiles'])
                traces.append([ligand_index, intermediate_index])
                traces.append([intermediate_index, target_index])
        return intermediate_info_list, traces

    def get_intermediate_info(self, ligand):
        # DEBUG: smiles must be canonical
        if Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(ligand))) != Chem.MolToSmiles(ligand):
            raise ValueError('smiles must be canonical, please report if this error occurs')
        smiles = Chem.MolToSmiles(ligand)

        return {
            'ligand': ligand,
            'smiles': smiles,
        }

    def merge_intermediate_info_list(self, prefix='intermediate'):
        forward_intermediate_info_list = self.forward_intermediate_info_list
        forward_traces = self.forward_traces
        backward_intermediate_info_list = self.backward_intermediate_info_list
        backward_traces =self.backward_traces

        forward_intermediate_smiles = [info['smiles'] for info in forward_intermediate_info_list]
        backward_intermediate_info_list_uniq = [info for info in backward_intermediate_info_list if info['smiles'] not in forward_intermediate_smiles]
        intermediate_info_list = forward_intermediate_info_list + backward_intermediate_info_list_uniq
        intermediate_smiles = [info['smiles'] for info in intermediate_info_list]

        for i, info in enumerate(intermediate_info_list):
            info['name'] = f'{prefix}-{i:04d}'
            info['ligand'].SetProp('_Name', info['name'])
            info['ligand'].SetProp('NAME', info['name'])
            info['ligand'].SetProp('smiles', info['smiles'])

        backward_traces_reindex = []
        reindexmap = {i:intermediate_smiles.index(info['smiles']) for i, info in enumerate(backward_intermediate_info_list)}
        for source_idx, target_idx in backward_traces:
            backward_traces_reindex.append([reindexmap[target_idx], reindexmap[source_idx]])
        intermediate_traces = forward_traces + backward_traces_reindex
        self.intermediate_info_list = intermediate_info_list
        self.intermediate_smiles = intermediate_smiles
        self.intermediate_traces = intermediate_traces
        return self.intermediate_info_list

    def show_result(self):
        print('Number of total intermediates: {}'.format(len(self.intermediate_info_list)))
        if self.ionize:
            print('Number of intermediates with the same formal charge: {}'.format(len(self.intermediates)))


    def search(self):
        # forward search
        self.forward_intermediate_info_list, self.forward_traces  = self.simplex_search('forward')
        # backward search
        if self.no_backward_search:
            self.backward_intermediate_info_list, self.backward_traces = [], []
        else:
            self.backward_intermediate_info_list, self.backward_traces = self.simplex_search('backward')

        # merge forward and backward and return
        intermediate_info_list = self.merge_intermediate_info_list(self.intermediate_name_prefix)
        self.intermediates_all = [info['ligand'] for info in intermediate_info_list]
        if self.ionize:
            self.intermediates_ionized = execute_ligand_preparation(self.intermediates_all, obabel_path=self.obabel_path)
            self.intermediates = [mol for mol in self.intermediates_ionized if formal_charge(mol) == self.formal_charge]
        else:
            self.intermediates = self.intermediates_all
        if self.verbose:
            self.show_result()
        return self.intermediates

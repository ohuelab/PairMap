from intermediate_generator import IntermediateGenerator
from lomap.mcs import MCS

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import RWMol
from collections import deque

class SearchIntermediates:
    def __init__(self, source_ligand, target_ligand, options):
        self.source_ligand = RWMol(AllChem.RemoveHs(source_ligand))
        self.target_ligand = RWMol(AllChem.RemoveHs(target_ligand))

        self.options = options
        self.verbose = options.verbose
        self.score_options = options.score_options

        self.generator = IntermediateGenerator(options.is_atom_modfication_enabled, options.cap_ring_with_carbon, options.cap_ring_with_hydrogen, options.verbose)

        self.baseMC = self.MCS(source_ligand, target_ligand)
        mcs = self.baseMC.mcs_mol
        self.seedSmarts = Chem.MolToSmarts(mcs)

    def MCS(self, source_ligand, target_ligand, seedSmarts=''):
        try:
            MC = MCS(source_ligand, target_ligand, **self.score_options, seed = seedSmarts)
        except:
            MC = MCS(source_ligand, target_ligand, **self.score_options, seed = '')
        return MC

    def search(self):
        # forward search
        self.forward_intermediate_list, self.forward_traces  = self.simplex_search('forward')
        # backward search
        self.backward_intermediate_list, self.backward_traces = self.simplex_search('backward')

        # merge forward and backward and return
        return self.get_intermediate_list()

    def simplex_search(self, direction='forward'):
        if direction=='forward':
            source_ligand = self.source_ligand
            target_ligand = self.target_ligand
        else:
            source_ligand = self.target_ligand
            target_ligand = self.source_ligand

        q = deque()
        q.append(source_ligand)
        intermediate_list = [self.get_intermediate_info(source_ligand)]
        intermediate_list += [self.get_intermediate_info(target_ligand)]
        source_index = 0
        target_index = 1
        smiles_list = [intermediate_list[source_index]['smiles'], intermediate_list[target_index]['smiles']]
        traces=[[source_index, target_index]]
        while len(q)>0:
            ligand = q.popleft()
            ligand = RWMol(ligand)
            # DEBUG: smiles must be canonical
            assert Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(ligand))) == Chem.MolToSmiles(ligand)
            smiles = Chem.MolToSmiles(ligand)
            ligand_index = smiles_list.index(smiles)
            MC = self.MCS(ligand, target_ligand, seedSmarts = self.seedSmarts)
            mcs_map = {a1:a2 for a1,a2 in MC.heavy_atom_mcs_map()}
            intermediates = self.generator.generate_intermediates(ligand, target_ligand, mcs_map)
            for intermediate in intermediates:
                info = self.get_intermediate_info(intermediate)
                if info['smiles'] not in smiles_list:
                    if self.verbose:
                        print('intermediate: {}'.format(info['smiles']))
                    smiles_list.append(info['smiles'])
                    intermediate_list.append(info)
                    q.append(intermediate)
                intermediate_index = smiles_list.index(info['smiles'])
                traces.append([ligand_index, intermediate_index])
                traces.append([intermediate_index, target_index])
        return intermediate_list, traces

    def get_intermediate_info(self, ligand):
        # DEBUG: smiles must be canonical
        assert Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(ligand))) == Chem.MolToSmiles(ligand)
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(ligand)))

        return {
            'ligand': ligand,
            'smiles': smiles,
        }

    def get_intermediate_list(self, prefix='mol'):
        forward_intermediate_list = self.forward_intermediate_list
        forward_traces = self.forward_traces
        backward_intermediate_list = self.backward_intermediate_list
        backward_traces =self.backward_traces

        forward_intermediate_smiles = [info['smiles'] for info in forward_intermediate_list]
        backward_intermediate_list_uniq = [info for info in backward_intermediate_list if info['smiles'] not in forward_intermediate_smiles]
        intermediate_list = forward_intermediate_list + backward_intermediate_list_uniq
        intermediate_smiles = [info['smiles'] for info in intermediate_list]

        for i, info in enumerate(intermediate_list):
            info['name'] = f'{prefix}-{i:04d}'
            info['ligand'].SetProp('_Name', info['name'])

        backward_traces_reindex = []
        reindexmap = {i:intermediate_smiles.index(info['smiles']) for i, info in enumerate(backward_intermediate_list)}
        for source_idx, target_idx in backward_traces:
            backward_traces_reindex.append([reindexmap[target_idx], reindexmap[source_idx]])
        intermediate_traces = forward_traces + backward_traces_reindex

        self.intermediate_list = intermediate_list
        self.intermediate_smiles = intermediate_smiles
        self.intermediate_traces = intermediate_traces
        return self.intermediate_list


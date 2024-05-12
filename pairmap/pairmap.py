from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
import argparse
import networkx as nx
import os
from search_intermediates import SearchIntermediates
from utils.scoremap import get_scoremap
from graphgen import IntermediateMapper


class Mapper:
    def __init__(self, intermediates, source_ligand, target_ligand, options=None):
        self.intemediates = intermediates
        self.source_ligand = source_ligand
        self.target_ligand = target_ligand

        self.options = options
        self.intermediate_options = options.intermediate_options
        self.score_options = options.intermediate_options.score_options
        self.mapper_options = options.mapper_options


    def search(self):
        self.intermediater = SearchIntermediates(self.source_ligand, self.target_ligand, self.intermediate_options)
        self.intermediate_list = self.searcher.intermediater()

    def calculate_score_matrix(self):
        self.scoremap = get_scoremap(self.intermediate_list, self.score_options)

    def generate_mapping(self):
        self.mapper = IntermediateMapper(self.intermediate_list, self.scoremap, self.mapper_options)
        self.final_graph = self.mapper.build_map()

    def save_molimages(self, force=False):
        intermediate_dict = self.intermediate_list
        image_dir = self.options.image_dir
        if force and os.path.exists(image_dir):
            for f in os.listdir(image_dir):
                os.remove(os.path.join(image_dir, f))
        elif not force and os.path.exists(image_dir):
            raise ValueError(f'{image_dir} already exists')
        os.makedirs(image_dir, exist_ok=True)
        molsdict={}
        for k in intermediate_dict:
            molsdict[k]=Chem.MolFromSmiles(intermediate_dict[k]['smiles'])

        mols = list(molsdict.values())
        # get mcs
        mcs_res = rdFMCS.FindMCS(mols, atomCompare=rdFMCS.AtomCompare.CompareAny)
        mcs = mcs_res.queryMol
        AllChem.Compute2DCoords(mcs)

        for fname in molsdict:
            imagefile=os.path.join(image_dir,fname+'.png')
            if not force and os.path.exists(imagefile):
                continue
            mol = molsdict[fname]
            AllChem.Compute2DCoords(mol)
            AllChem.GenerateDepictionMatching2DStructure(mol,mcs)
            img = Draw.MolToImage(mol, size=(200, 200))
            img.save(imagefile, bbox_inches='tight')

    def draw_graph(self, fontsize=20, imagefile='graph.png', pos = None):
        graph = self.final_graph
        if pos is None:
            # TODO: use a better layout
            pos = nx.spring_layout(graph)

        for n in graph:
            graph.nodes[n]['labelloc'] = 't'
            graph.nodes[n]['penwidth'] = 2.5

        for u, v, d in graph.edges(data=True):
            score=d.get('score')
            if graph[u][v].get('color') is not None:
                continue
            if score>0.4:
                graph[u][v]['color'] = 'blue'
                graph[u][v]['penwidth'] = 2.5
            else:
                graph[u][v]['color'] = 'red'
                graph[u][v]['penwidth'] = 2.5
            graph[u][v]['fontsize']=fontsize
            graph[u][v]['label'] = '{:.3f}'.format(score)

        a_graph = nx.nx_agraph.to_agraph(graph)
        a_graph.graph_attr.update(splines='true', overlap='false')
        a_graph.node_attr.update(shape='box', fontcolor='blue', fontname='Arial')
        a_graph.edge_attr.update(fontname='Arial')

        pos_dict = {}

        for node, coords in pos.items():
            pos_dict[node] = f"{coords[0]*1500},{coords[1]*1500}"

        # Apply the positions to the agraph layout
        a_graph.layout(prog="neato", args="-n")
        for i, node in enumerate(a_graph.nodes()):
            node.attr["pos"] = pos_dict[int(node)]

        # Draw the graph
        a_graph.draw(imagefile)


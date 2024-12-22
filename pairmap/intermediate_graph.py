import os
import yaml
import pickle
import copy
import logging
import networkx as nx
import numpy as np
import pandas as pd
from operator import itemgetter
import itertools

from rdkit import Chem
from rdkit.Chem import AllChem

# LOMAP imports
import lomap
from lomap import DBMolecules
from lomap.dbmol import ecr
from lomap import mcs

# Local imports
from .utils import find_bridges
from .search_intermediates import SearchIntermediates
from .map_generator import MapGenerator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SOURCE_INDEX = 0
TARGET_INDEX = 1

def get_similarity(moli, molj, options=None):
    if options is None:
        options = {
            'time': 20,
            'verbose': 'info',
            'max3d': 1000.0,
            'threed': False,
            'element_change': True,
            'seed': '',
            'shift': True
        }

    moli, molj = copy.deepcopy(moli), copy.deepcopy(molj)
    ecr_score = ecr(moli, molj)

    MC = mcs.MCS(
        moli, molj,
        time=options.get('time', 20),
        verbose=options.get('verbose', 'info'),
        threed=options.get('threed', False),
        max3d=options.get('max3d', 1000.0),
        element_change=options.get('element_change', True),
        seed=options.get('seed', ''),
        shift=options.get('shift', True)
    )
    MC.all_atom_match_list()
    tmp_scr = ecr_score * MC.mncar() * MC.mcsr() * MC.atomic_number_rule() * MC.hybridization_rule()
    tmp_scr *= MC.sulfonamides_rule() * MC.heterocycles_rule() * MC.transmuting_methyl_into_ring_rule()
    tmp_scr *= MC.transmuting_ring_sizes_rule()
    strict_scr = tmp_scr * 1  # MC.tmcsr(strict_flag=True)
    return strict_scr


class IntermediateGraphManager:

    def __init__(self, config_file=None, custom_execute_ligand_preparation=None, custom_get_similarity=None, custom_get_score_matrix = None, **kwargs):
        """
        Initialize settings from a YAML config file, or use defaults.
        """
        self.config = {}
        if config_file:
            self.load_config(config_file)

        # Default options
        self.config.update(kwargs)
        self.config.setdefault('input_dir', './input')
        self.config.setdefault('output_dir', './output')
        self.config.setdefault('save_output', True)
        self.config.setdefault('similarity_threshold', 0.6)
        self.config.setdefault('max_intermediate', -1)

        self.config.setdefault('jobs', -1)
        self.config.setdefault('cutoff', 0.0)
        self.config.setdefault('chunk_mode', True)
        self.config.setdefault('chunk_scale', 10)
        self.config.setdefault('chunk_terminate_factor', 2)
        self.config.setdefault('node_mode', False)
        self.config.setdefault('max', 6)
        self.config.setdefault('max_dist_from_actives', 6)
        self.config.setdefault('allow_tree', False)
        self.config.setdefault('radial', False)
        self.config.setdefault('max_path_length', 4)

        # MapGenerator options
        self.config.setdefault("maxOptimalPathLength", 4)
        self.config.setdefault("roughScoreThreshold", 0.5)
        self.config.setdefault("optimal_path_mode", True)
        self.config.setdefault("minScoreThreshold", 0.2)
        self.config.setdefault("verbose", True)

        # Custom functions
        self.get_score_matrix = custom_get_score_matrix
        self.execute_ligand_preparation = custom_execute_ligand_preparation
        self.get_similarity = custom_get_similarity or get_similarity

    def load_config(self, config_file):
        """
        Load YAML configuration from a file into self.config.
        """
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
            if cfg:
                self.config.update(cfg)

    def generate_intermediate_path(self, node_mols, new_graph, intermediates_avail, options):
        """
        Insert a chain of intermediate molecules into 'new_graph' if they are not already present.
        Returns updated structures or None if no additional intermediates can be added.
        """

        custom_score_matrix = self.get_score_matrix(list(intermediates_avail), options, jobs = self.config["jobs"]) if self.get_score_matrix is not None else None
        mapGen = MapGenerator(
            intermediates_avail,
            maxOptimalPathLength=self.config["maxOptimalPathLength"],
            roughScoreThreshold=self.config["roughScoreThreshold"],
            optimal_path_mode=self.config["optimal_path_mode"],
            minScoreThreshold=self.config["minScoreThreshold"],
            custom_score_matrix=custom_score_matrix,
            verbose=self.config["verbose"],
        )
        pairgraph = mapGen.build_map()

        existing_smiles = [Chem.MolToSmiles(node_mols[n]) for n in new_graph.nodes]
        generated_intermediate_names = {
            i: data["label"] for i, data in pairgraph.nodes(data=True) if data.get("label")
        }
        generated_intermediates = {
            i: mapGen.intermediate_list[mapGen.intermediate_names.index(name)]
            for i, name in generated_intermediate_names.items()
        }

        if len(generated_intermediate_names) == 2:
            logger.info("No additional intermediates to add.")
            return None

        node_idx = max(new_graph.nodes) + 1
        node_remapping = {}
        additional_intermediates = {}

        for i, mol in generated_intermediates.items():
            smiles = mol.GetProp("smiles")
            if smiles not in existing_smiles:
                new_graph.add_node(node_idx,
                                   active=False, intermediate=True)
                node_remapping[i] = node_idx
                mol.SetProp("_Name", f"Intermediate-{node_idx:03d}")
                additional_intermediates[i] = mol
                node_mols[node_idx] = mol
                node_idx += 1
            else:
                found_node = False
                for node in new_graph.nodes:
                    if Chem.MolToSmiles(node_mols[node]) == smiles:
                        node_remapping[i] = node
                        found_node = True
                        break
                if not found_node:
                    raise ValueError("Intermediate node with existing SMILES not found in graph.")
        return new_graph, node_mols, node_remapping, additional_intermediates

    def run_from_moldf(self, mols, df):
        """
        Build or update a graph from an existing list of RDKit molecules 'mols'
        and a DataFrame 'df' containing edges. Then prune edges below a similarity
        threshold by inserting intermediates.

        The DataFrame 'df' should contain columns:
          - 'Node1': str  (the RDKit Mol's _Name)
          - 'Node2': str  (the RDKit Mol's _Name)
          - 'score': float
          - 'BadEdge': bool
        """

        new_graph = nx.Graph()
        name2id = {m.GetProp("_Name"): i for i, m in enumerate(mols)}
        node_mols = {i: m for i, m in enumerate(mols)}

        # Create nodes
        for name, i in name2id.items():
            new_graph.add_node(i, fname=name, active=True, intermediate=False)

        # Add edges
        for _, row in df.iterrows():
            n1, n2 = name2id[row["Node1"]], name2id[row["Node2"]]
            new_graph.add_edge(n1, n2, similarity=row["score"], bad_edge=row["BadEdge"])

        # Prune the initial graph
        GGen = IntermediateGraphGen(new_graph, self.config)
        new_graph = GGen.get_graph()

        initial_num = len(new_graph.nodes)
        new_graphs = [new_graph.copy()]
        added_edges = []

        while True:
            intermediate_num = len(new_graph.nodes) - initial_num
            if (self.config['max_intermediate'] > 0) and (intermediate_num > self.config['max_intermediate']):
                logger.info("Max number of intermediates reached.")
                break

            bad_edges = []
            bad_sims = []
            for u, v, d in new_graph.edges(data=True):
                if d.get("bad_edge", False):
                    bad_edges.append((u, v))
                    bad_sims.append(d["similarity"])

            if not bad_edges:
                logger.info("No more bad edges.")
                break

            sort_indices = np.argsort(bad_sims)
            bad_edges = [bad_edges[i] for i in sort_indices]

            found = False
            for edge in bad_edges:
                if (edge not in added_edges) and ((edge[1], edge[0]) not in added_edges):
                    added_edges.append(edge)
                    source_node, target_node = edge
                    found = True
                    break
            if not found:
                logger.info("No more edges to fix.")
                break

            logger.info(f"Target edge to fix: {source_node} -- {target_node}")

            # Generate potential intermediates
            source_ligand = node_mols[source_node]
            target_ligand = node_mols[target_node]

            search_intm = SearchIntermediates(
                source_ligand, target_ligand,
                is_atom_modfication_enabled=self.config.get("is_atom_modfication_enabled", True), cap_ring_with_carbon=self.config.get("cap_ring_with_carbon", True),
                cap_ring_with_hydrogen=self.config.get("cap_ring_with_hydrogen", True), no_backward_search=self.config.get("no_backward_search", False),
                intermediate_name_prefix=self.config.get("intermediate_name_prefix", "Intermediate"), use_seed=self.config.get("use_seed", True),
                max_intermediate=self.config['max_intermediate'],
            )
            intermediates_avail = search_intm.search()
            if self.execute_ligand_preparation is not None:
                intermediates_avail = self.execute_ligand_preparation(intermediates_avail)

            for m in intermediates_avail:
                Chem.AssignStereochemistryFrom3D(m)

            if Chem.MolToSmiles(source_ligand) != Chem.MolToSmiles(intermediates_avail[SOURCE_INDEX]):
                raise ValueError("Source ligand changed after search.")
            if Chem.MolToSmiles(target_ligand) != Chem.MolToSmiles(intermediates_avail[TARGET_INDEX]):
                raise ValueError("Target ligand changed after search.")

            out = self.generate_intermediate_path(node_mols, new_graph, intermediates_avail, self.config)
            if out is None:
                logger.info("No new intermediates added. Skipping.")
                continue

            new_graph, node_mols, node_remapping, additional_intermediates = out
            if not additional_intermediates:
                logger.info("Retrying with novel intermediates only.")
                existing_smiles = [Chem.MolToSmiles(node_mols[n]) for n in new_graph.nodes]
                existing_smiles2node = {Chem.MolToSmiles(node_mols[n]): n for n in new_graph.nodes}
                intermediates_novel = []

                for i, mol in enumerate(intermediates_avail):
                    if i in [SOURCE_INDEX, TARGET_INDEX]:
                        intermediates_novel.append(mol)
                    elif Chem.MolToSmiles(mol) not in existing_smiles:
                        intermediates_novel.append(mol)
                    else:
                        intm_node = existing_smiles2node.get(Chem.MolToSmiles(mol))
                        if (new_graph.get_edge_data(source_node, intm_node) is None and
                                new_graph.get_edge_data(target_node, intm_node) is None):
                            intermediates_novel.append(mol)

                intermediates_avail = intermediates_novel
                out = self.generate_intermediate_path(node_mols, new_graph, intermediates_avail, self.config)
                if out is not None:
                    new_graph, node_mols, node_remapping, additional_intermediates = out

            update_nodes = list(node_remapping.values())
            for u in new_graph:
                for v in update_nodes:
                    if (u != v) and not new_graph.get_edge_data(u, v):
                        sim_ = self.get_similarity(node_mols[u], node_mols[v], self.config)
                        if sim_ > self.config["minScoreThreshold"]:
                            new_graph.add_edge(u, v, similarity=sim_, strict_flag=True)

            # Prune again
            GGen = IntermediateGraphGen(new_graph, self.config)
            new_graph = GGen.get_graph()

            new_graphs.append(new_graph.copy())
            logger.info("=====================================")
            logger.info(f"Target edge: {source_node} -- {target_node}")
            logger.info(f"Remapping: {node_remapping}")
            logger.info(f"Number of nodes: {len(new_graph.nodes)}")
            logger.info(f"Number of edges: {len(new_graph.edges)}")
            logger.info(f"Number of additional intermediates: {len(additional_intermediates)}")
            logger.info("=====================================")

        # Save
        return new_graphs, node_mols

    def generate_moldf(self, input_dir):

        db_mol = DBMolecules(input_dir, output=False, output_no_images=True, output_no_graph=True)
        _, _ = db_mol.build_matrices()
        nx_graph = db_mol.build_graph()

        mols = []
        for i in nx_graph.nodes():
            m = db_mol[i].getMolecule()
            if not m.HasProp("_Name"):
                m.SetProp("_Name", f"Molecule_{i}")
            m = AllChem.RemoveHs(m)
            mols.append(m)

        data = []
        for u, v in nx_graph.edges():
            lomap_sim = nx_graph[u][v]["similarity"]

            bad_edge = (lomap_sim < self.config["similarity_threshold"])

            name_u = mols[u].GetProp("_Name")
            name_v = mols[v].GetProp("_Name")

            data.append((name_u, name_v, lomap_sim, bad_edge))

        df = pd.DataFrame(data, columns=["Node1", "Node2", "score", "BadEdge"])
        return mols, df

    def save_output(self, new_graphs, node_mols):
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])

        # Save the final graph
        with open(os.path.join(self.config["output_dir"], "intermediate_graph.pkl"), "wb") as f:
            pickle.dump({"graph": new_graphs, "mols": node_mols}, f)

        # Save Intermediate Molecules
        new_graph = new_graphs[-1]
        output_mols = []
        for n in new_graph.nodes:
            if n in new_graph.nodes:
                m=node_mols[n]
                name = m.GetProp("_Name")
                new_graph.nodes[n]["NAME"]=name
                m.ClearProp("NAME")
                m = AllChem.AddHs(m)
                output_mols.append(m)
        with Chem.SDWriter(os.path.join(self.config["output_dir"], "intermediate_mols.sdf")) as w:
            for m in output_mols:
                w.write(m)

        # Save the intermediate links
        output_links = ""
        for u,v in new_graph.edges():
            namei=new_graph.nodes[u]["NAME"]
            namej=new_graph.nodes[v]["NAME"]
            score=new_graph.get_edge_data(u,v)["similarity"]
            output_links+=f"{namei} {namej} {score:.3f}\n"
        with open(os.path.join(self.config["output_dir"], "intermediate_links.txt"), "w") as f:
            f.write(output_links)
        logger.info(f"Output saved to {self.config['output_dir']}")
        return

    def run(self):
        logger.info("Generating intermediate graph...")
        mols, df = self.generate_moldf(self.config["input_dir"])

        logger.info("Running intermediate graph generation...")
        new_graphs, node_mols = self.run_from_moldf(mols, df)
        logger.info("Intermediate graph generation complete.")

        if self.config.get("save_output", False):
            self.save_output(new_graphs, node_mols)
        return new_graphs, node_mols

class IntermediateGraphGen(object):
    """
    This class is used to set and generate the graph used to plan
    binding free energy calculation.
    """

    def __init__(self, subgraph, options, ignore_intermediates=True):

        self.subgraph = subgraph
        self.maxPathLength = options.get('max', 4)
        self.maxDistFromActive = options.get('max_dist_from_actives', self.maxPathLength)
        self.similarityScoresLimit = options.get('cutoff', 0.0)
        self.requireCycleCovering = not options.get('allow_tree', False)

        if ignore_intermediates:
            self.intermediate_nodes = [node for node in subgraph.nodes if subgraph.nodes[node].get("intermediate")]
            self.essential_nodes = [node for node in subgraph.nodes if not subgraph.nodes[node].get("intermediate")]
        else:
            self.intermediate_nodes = []
            self.essential_nodes = [node for node in subgraph.nodes]

        if options.get('radial', False):
            self.lead_index = self.pick_lead()
        else:
            self.lead_index = None

        self.nonCycleNodesSet = None
        self.nonCycleEdgesSet = None
        self.distanceToActiveFailures = 0

        self.chunk_mode = options.get('chunk_mode', True)
        self.node_mode = options.get('node_mode', False)

        # Sort edges by similarity
        self.weightsList = sorted([(i, j, d['similarity']) for i, j, d in subgraph.edges(data=True)],
                                  key=itemgetter(2))

        if options.get("chunk_scale", 0) > 1:
            self.chunk_scale = options.get('chunk_scale', 10)
        else:
            self.chunk_scale = 10

        self.chunk_terminate_factor = options.get('chunk_terminate_factor', 2)

        self.minimize_edges()

    def pick_lead(self):
        raise NotImplementedError("Radial mode not implemented yet.")

    def minimize_edges(self):
        """
        Minimize edges in each subgraph while ensuring constraints are met
        """
        subgraph = self.subgraph.copy()

        def check_chunk(edge_chunk, data_chunk):
            # Remove edges in the chunk and check constraints
            # If constraints fail, revert
            similarities = [d['similarity'] < 1.0 for d in data_chunk]
            if not all(similarities):
                # If we have any edges with similarity=1.0, skip removing them
                if not any(similarities):
                    logger.info(f"Skip chunk with all edges similarity=1.0, chunk size={len(edge_chunk)}")
                    return True
                return False
            else:
                subgraph.remove_edges_from(edge_chunk)
                if self.check_constraints(subgraph):
                    logger.info(f"Removed chunk size={len(edge_chunk)}")
                    return True
                # revert
                for (i, j), d in zip(edge_chunk, data_chunk):
                    subgraph.add_edge(i, j, **d)
                return False

        def chunk_process(edge_chunk, data_chunk, chunk_size, idx):
            if check_chunk(edge_chunk, data_chunk):
                # Edges can be removed
                return True
            elif chunk_size == 1:
                # The edge cannot be removed
                return False
            else:
                # Attempt smaller chunk splits
                logger.info(f"Split chunk range=({idx}, {idx+chunk_size}), size={chunk_size}")
                chunk_size = max(chunk_size // self.chunk_scale, 1)
                for i in range(0, len(edge_chunk), chunk_size):
                    logger.info(f"Process chunk range=({idx + i}, {idx + i + chunk_size}), size={chunk_size}")
                    ret = chunk_process(edge_chunk[i:i + chunk_size],
                                        data_chunk[i:i + chunk_size],
                                        chunk_size,
                                        idx + i)
                    if not ret:
                        # Try removing the rest
                        if check_chunk(edge_chunk[i + chunk_size:], data_chunk[i + chunk_size:]):
                            break

        self.nonCycleNodesSet = self.find_non_cyclic_nodes(subgraph) if not self.nonCycleNodesSet else self.nonCycleNodesSet
        self.nonCycleEdgesSet = self.find_non_cyclic_edges(subgraph) if not self.nonCycleEdgesSet else self.nonCycleEdgesSet
        self.distanceToActiveFailures = self.count_distance_to_active_failures(subgraph)

        # If chunk_mode is False, do a simpler iterative removal
        if len(subgraph.edges()) > 2 and not self.chunk_mode:
            for edge in self.weightsList:
                if self.lead_index is not None:
                    if self.lead_index not in [edge[0], edge[1]]:
                        edge_data = subgraph.get_edge_data(edge[0], edge[1])
                        subgraph.remove_edge(edge[0], edge[1])
                        if not self.check_constraints(subgraph):
                            subgraph.add_edge(edge[0], edge[1], **edge_data)
                elif edge[2] < 1.0:
                    logger.info(f"Attempt removing edge {edge[0]}--{edge[1]} sim={edge[2]:.2f}")
                    edge_data = subgraph.get_edge_data(edge[0], edge[1])
                    subgraph.remove_edge(edge[0], edge[1])
                    if not self.check_constraints(subgraph):
                        subgraph.add_edge(edge[0], edge[1], **edge_data)
                    else:
                        logger.info(f"Removed edge {edge[0]}--{edge[1]}")
                else:
                    logger.info(f"Skipping edge {edge[0]}--{edge[1]} with similarity=1")
        elif len(subgraph.edges()) > 2:
            # radial not supported in fast chunk mode
            N = len(subgraph)
            M = len(subgraph.edges())
            edges = [(i, j) for i, j, d in self.weightsList]
            data = [subgraph.get_edge_data(i, j) for i, j, d in self.weightsList]

            chunk_size = self.chunk_scale ** int(np.log(len(self.weightsList)) / np.log(self.chunk_scale))
            terminate_n = int(self.chunk_terminate_factor * N)
            chunk_list = list(range(0, M - terminate_n, chunk_size)) + list(range(M - terminate_n, M))

            for i, idx_i in enumerate(chunk_list):
                idx_j = chunk_list[i+1] if i < len(chunk_list)-1 else M
                chunk_size_l = idx_j - idx_i
                edge_chunk = edges[idx_i:idx_j]
                data_chunk = data[idx_i:idx_j]
                if len(edge_chunk) > 1:
                    logger.info(f"Process chunk=({idx_i}, {idx_j}), size={chunk_size_l}")
                    chunk_process(edge_chunk, data_chunk, chunk_size_l, idx_i)
                else:
                    logger.info(f"Process chunk=({idx_i}, {idx_j}), size={chunk_size_l}")
                    check_chunk(edge_chunk, data_chunk)

        # Ensure essential nodes remain in a single connected component
        subgraphs = list(nx.connected_components(subgraph))
        for sub in subgraphs:
            if len(set(sub).intersection(self.essential_nodes)) == len(self.essential_nodes):
                subgraph = subgraph.subgraph(sub).copy()
                break

        self.resultGraph = subgraph

    def find_non_cyclic_nodes(self, subgraph):
        cycleList = nx.cycle_basis(subgraph)
        cycleNodes = set(itertools.chain.from_iterable(cycleList))
        missingNodesSet = {node for node in self.essential_nodes if node not in cycleNodes}
        return missingNodesSet

    def find_non_cyclic_edges(self, subgraph):
        missingEdgesSet = find_bridges(subgraph)
        removeEdges = []
        for node in self.intermediate_nodes:
            for edge in missingEdgesSet:
                if node in edge:
                    removeEdges.append(edge)
        for edge in removeEdges:
            if edge in missingEdgesSet:
                missingEdgesSet.remove(edge)
        return missingEdgesSet

    def check_constraints(self, subgraph):
        if not self.remains_connected(subgraph):
            logger.info("Rejecting edge deletion on connectedness.")
            return False

        if self.requireCycleCovering:
            if not self.check_cycle_covering(subgraph):
                logger.info("Rejecting edge deletion on cycle covering.")
                return False

        if not self.check_max_distance(subgraph):
            logger.info("Rejecting edge deletion on max distance.")
            return False

        if not self.check_distance_to_active(subgraph):
            logger.info("Rejecting edge deletion on distance-to-actives.")
            return False

        return True

    def remains_connected(self, subgraph):
        subgraphs = list(nx.connected_components(subgraph))
        if len(subgraphs) == 1:
            return True
        # Or if essential nodes remain in the same component
        for sub in subgraphs:
            if len(set(sub).intersection(self.essential_nodes)) == len(self.essential_nodes):
                return True
        return False

    def check_cycle_covering(self, subgraph):
        if self.find_non_cyclic_nodes(subgraph).difference(self.nonCycleNodesSet):
            logger.info("Rejecting edge deletion on cycle covering (nodes).")
            return False
        if self.find_non_cyclic_edges(subgraph).difference(self.nonCycleEdgesSet):
            logger.info("Rejecting edge deletion on cycle covering (edges).")
            return False
        return True

    def check_max_distance(self, subgraph):
        for node1 in self.essential_nodes:
            for node2 in self.essential_nodes:
                if node1 != node2:
                    if not nx.has_path(subgraph, node1, node2):
                        return False
                    if nx.shortest_path_length(subgraph, node1, node2) > self.maxPathLength:
                        return False
        return True

    def count_distance_to_active_failures(self, subgraph):
        # If no nodes are active, we skip
        if not any(subgraph.nodes[node].get("active", False) for node in self.essential_nodes):
            return 0

        failures = 0
        paths = dict(nx.shortest_path(subgraph))
        for node in self.essential_nodes:
            if subgraph.nodes[node].get("active", False):
                continue
            ok = False
            for node2 in self.essential_nodes:
                if subgraph.nodes[node2].get("active", False):
                    pathlen = len(paths[node][node2]) - 1
                    if pathlen <= self.maxDistFromActive:
                        ok = True
                        break
            if not ok:
                failures += 1
        return failures

    def check_distance_to_active(self, subgraph):
        count = self.count_distance_to_active_failures(subgraph)
        failed = (count > self.distanceToActiveFailures)
        if failed:
            logger.info(f"Rejecting edge deletion on distance-to-actives. {count} vs {self.distanceToActiveFailures}")
        else:
            logger.info(f"Distance-to-active check: {count} vs {self.distanceToActiveFailures}")
        return not failed

    def get_graph(self):
        return self.resultGraph

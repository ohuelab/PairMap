import networkx as nx
import numpy as np
from operator import itemgetter
import logging
import itertools
import copy
from rdkit import Chem
from rdkit.Chem import AllChem


import lomap
from lomap.dbmol import ecr
from lomap import mcs

from .utils import execute_ligand_preparation

from .search_intermediates import SearchIntermediates
from .map_generator import MapGenerator


from .utils import find_bridges


class IntermediateGraphGen(object):
    """
    This class is used to set and generate the graph used to plan
    binding free energy calculation
    """

    def __init__(self, subgraph, options, ignore_intermediates=True):

        """
        Inizialization function

        Parameters
        ----------

        dbase : dbase object
            the molecule container

        """

        self.subgraph = subgraph

        self.maxPathLength = nx.diameter(subgraph)

        self.maxDistFromActive = options['max_dist_from_actives']

        self.similarityScoresLimit = options['cutoff']

        self.requireCycleCovering = not options['allow_tree']

        if ignore_intermediates:
            self.intermediate_nodes = [node for node in subgraph.nodes if subgraph.nodes[node].get("intermediate")]
            self.essential_nodes = [node for node in subgraph.nodes if not subgraph.nodes[node].get("intermediate")]
        else:
            self.intermediate_nodes = []
            self.essential_nodes = [node for node in subgraph.nodes]

        if options['radial']:
            self.lead_index = self.pick_lead()
        else:
            self.lead_index = None

        # A set of nodes that will be used to save nodes that are not a cycle cover for a given subgraph
        self.nonCycleNodesSet = set()

        # A set of edges that will be used to save edges that are acyclic for given subgraph
        self.nonCycleEdgesSet = set()

        # A count of the number of nodes that are not within self.maxDistFromActive edges
        # of an active
        self.distanceToActiveFailures = 0

        # Draw Parameters

        # THIS PART MUST BE CHANGED

        self.chunk_mode = options.get('chunk_mode', True)
        self.node_mode = options.get('node_mode', False)

        self.weightsList = sorted([(i, j, d['similarity']) for i, j, d in subgraph.edges(data=True)], key=itemgetter(2))

        if options.get("chunk_scale", 0) > 1:
            self.chunk_scale = options.get('chunk_scale', 10)
        else:
            self.chunk_scale = 10

        self.chunk_terminate_factor = options.get('chunk_terminate_factor', 2)

        self.minimize_edges()

    def minimize_edges(self):
        """
        Minimize edges in each subgraph while ensuring constraints are met
        """

        # Remove edges in chunks and check constraints
        def chunk_process(edge_chunk, data_chunk, chunk_size, idx):
            if check_chunk(edge_chunk, data_chunk):
                # Edges can be removed
                return True
            elif chunk_size == 1:
                # The edge cannot be removed
                return False
            else:
                # Re-split when there are multiple chunks
                logging.info('Split: #E={}, {} {}'.format(len(subgraph.edges()), idx, idx + chunk_size))
                # Run chunk_process recursively with smaller chunks
                chunk_size = max(chunk_size // self.chunk_scale, 1)
                for i in range(0, len(edge_chunk), chunk_size):
                    logging.info('Process: #E={}, {} {}'.format(len(subgraph.edges()), idx + i, idx + i + chunk_size))
                    ret = chunk_process(edge_chunk[i:i + chunk_size], data_chunk[i:i + chunk_size], chunk_size, idx + i)
                    # If unremovable edges are found, try to check the rest of the chunk
                    if not ret:
                        if check_chunk(edge_chunk[i + chunk_size:len(edge_chunk)], data_chunk[i + chunk_size:len(data_chunk)]):
                            # Remain edges can be removed
                            break

        # Check constraints by chunk
        def check_chunk(edge_chunk, data_chunk):
            similarities = [d['similarity'] < 1.0 for d in data_chunk]
            if not all(similarities):
                if not any(similarities):
                    logging.info('Skip (similarity=1.0): {}'.format(len(edge_chunk)))
                    return True
                # Restore edges because they contain edges that cannot be removed
                return False
            else:
                # Remove edges in the chunk and check constraints
                subgraph.remove_edges_from(edge_chunk)
                if self.check_constraints(subgraph):
                    logging.info('Removed: {}'.format(len(edge_chunk)))
                    return True
                # Restore edges because constraints are not satisfied
                for (i, j), d in zip(edge_chunk, data_chunk):
                    subgraph.add_edge(i, j, **d)
                return False

        subgraph = self.subgraph.copy()
        weightsList = self.weightsList

        # weightsList = sorted(weightsList, key = itemgetter(1))

        # This part has been copied from the original code
        self.nonCycleNodesSet = self.find_non_cyclic_nodes(subgraph)
        self.nonCycleEdgesSet = self.find_non_cyclic_edges(subgraph)
        self.distanceToActiveFailures = self.count_distance_to_active_failures(subgraph)

        if len(subgraph.edges()) > 2 and not self.chunk_mode:  # Graphs must have at least 3 edges to be minimzed
            for edge in weightsList:
                if self.lead_index is not None:
                    # Here the radial option is appplied, will check if the remove_edge is connect to
                    # the hub(lead) compound, if the edge is connected to the lead compound,
                    # then add it back into the graph.
                    if self.lead_index not in [edge[0], edge[1]]:
                        subgraph.remove_edge(edge[0], edge[1])
                        if self.check_constraints(subgraph) == False:
                            subgraph.add_edge(edge[0], edge[1], similarity=edge[2], strict_flag=True)
                elif edge[2] < 1.0:  # Don't remove edges with similarity 1
                    logging.info("Trying to remove edge %d-%d with similarity %f" % (edge[0],edge[1],edge[2]))
                    subgraph.remove_edge(edge[0], edge[1])
                    if self.check_constraints(subgraph) == False:
                        subgraph.add_edge(edge[0], edge[1], similarity=edge[2], strict_flag=True)
                    else:
                        logging.info("Removed edge %d-%d" % (edge[0],edge[1]))
                else:
                    logging.info("Skipping edge %d-%d as it has similarity 1" % (edge[0],edge[1]))
        elif len(subgraph.edges()) > 2:
            # radial option is not supported in fast mode
            N = len(subgraph)
            M = len(subgraph.edges())
            edges = [(i, j) for i, j, d in weightsList]
            data = [{'similarity': d, 'strict_flag': True} for i, j, d in weightsList]
            chunk_size = self.chunk_scale **int(np.log(len(weightsList))/np.log(self.chunk_scale))
            terminate_n = int(self.chunk_terminate_factor * N)
            chunk_list = list(range(0, M-terminate_n, chunk_size))+list(range(M-terminate_n, M))
            # Process edges in chunks
            for i, idx_i in enumerate(chunk_list):
                idx_j = chunk_list[i+1] if i<len(chunk_list)-1 else M
                chunk_size_l = idx_j - idx_i
                edge_chunk = edges[idx_i:idx_j]
                data_chunk = data[idx_i:idx_j]
                if len(edge_chunk) >1:
                    logging.info('Process: #E={}, {} {}'.format(len(subgraph.edges()), idx_i, idx_j))
                    chunk_process(edge_chunk, data_chunk, chunk_size_l, idx_i)
                else:
                    logging.info('Process: #E={}, {} {}'.format(len(subgraph.edges()), idx_i, idx_j))
                    check_chunk(edge_chunk, data_chunk)
        subgraphs = list(nx.connected_components(subgraph))
        for sub in subgraphs:
            if len(set(sub).intersection(self.essential_nodes)) == len(self.essential_nodes):
                break
        subgraph = subgraph.subgraph(sub).copy()
        self.resultGraph = subgraph

    def find_non_cyclic_nodes(self, subgraph):
        """
        Generates a list of nodes of the subgraph that are not in a cycle

        Parameters
        ---------
        subgraph : NetworkX subgraph obj
            the subgraph to check for not cycle nodes

        Returns
        -------
        missingNodesSet : set of graph nodes
            the set of graph nodes that are not in a cycle

        """

        cycleList = nx.cycle_basis(subgraph)
        cycleNodes = set(list(itertools.chain.from_iterable(cycleList)))
        missingNodesSet = set([node for node in self.essential_nodes if node not in cycleNodes])
        return missingNodesSet

    def find_non_cyclic_edges(self, subgraph):
        """
        Generates a set of edges of the subgraph that are not in a cycle.

        Parameters
        ---------
        subgraph : NetworkX subgraph obj
            the subgraph to check for not cycle nodes

        Returns
        -------
        missingEdgesSet : set of graph edges
            the set of edges that are not in a cycle

        """

        missingEdgesSet = find_bridges(subgraph)
        # if missingEdgesSet contains an edge that is connected to an intermediate node, remove it
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
        """
        Determine if the given subgraph still meets the constraints


        Parameters
        ----------
        subgraph : NetworkX subgraph obj
             the subgraph to check for the constraints

        numComp : int
            the number of connected componets

        Returns
        -------
        constraintsMet : bool
           True if all the constraints are met, False otherwise
        """

        constraintsMet = True

        if not self.remains_connected(subgraph):
            constraintsMet = False

        # The requirement to keep a cycle covering is now optional
        if constraintsMet and self.requireCycleCovering:
            if not self.check_cycle_covering(subgraph):
                constraintsMet = False

        if constraintsMet:
            if not self.check_max_distance(subgraph):
                constraintsMet = False

        if constraintsMet:
            if not self.check_distance_to_active(subgraph):
                constraintsMet = False

        return constraintsMet

    def remains_connected(self, subgraph):
        """
        Determine if the subgraph remains connected after an edge has been
        removed

        Parameters
        ---------
        subgraph : NetworkX subgraph obj
            the subgraph to check for connection after the edge deletition

        numComp : int
            the number of connected componets

        Returns
        -------
        isConnected : bool
            True if the subgraph is connected, False otherwise
            :param numComponents:

        """

        subgraphs = list(nx.connected_components(subgraph))
        isConnected = False
        if len(subgraphs) == 1:
            isConnected = True
        for sub in subgraphs:
            # sub is subset of essential nodes
            # if len(set(sub.nodes()).intersection(self.essential_nodes)) == len(sub.nodes()):
            if len(set(sub).intersection(self.essential_nodes)) == len(self.essential_nodes):
                isConnected = True
                break
        return isConnected

    def check_cycle_covering(self, subgraph):
        """
        Checks if the subgraph has a cycle covering. Note that this has been extended from
        the original algorithm: we not only care if the number of acyclic nodes has
        increased, but we also care if the number of acyclic edges (bridges) has increased.
        Note that if the number of acyclic edges hasn't increased, then the number of
        acyclic nodes hasn't either, so that test is included in the edges test.

        Parameters
        ---------
        subgraph : NetworkX subgraph obj
            the subgraph to check for connection after the edge deletion

        Returns
        -------
        hasCovering : bool
            True if the subgraph has a cycle covering, False otherwise

        """

        hasCovering = True

        if self.find_non_cyclic_nodes(subgraph).difference(self.nonCycleNodesSet):
            hasCovering = False
            logging.info("Rejecting edge deletion on cycle covering (nodes)")

        # Have we increased the number of non-cyclic edges?
        if self.find_non_cyclic_edges(subgraph).difference(self.nonCycleEdgesSet):
            hasCovering = False
            logging.info("Rejecting edge deletion on cycle covering")

        return hasCovering

    def check_max_distance(self, subgraph):
        """
        Check to see if the graph has paths from all compounds to all other
        compounds within the specified limit

        Parameters
        ---------
        subgraph : NetworkX subgraph obj
            the subgraph to check for the max distance between nodes

        Returns
        -------
        withinMaxDistance : bool
            True if the subgraph has all the nodes within the specified
            max distance
        """

        withinMaxDistance = True
        for node in self.essential_nodes:
            if not withinMaxDistance:
                break
            for node2 in self.essential_nodes:
                if node != node2:
                    if not nx.has_path(subgraph, node, node2):
                        withinMaxDistance = False
                        break
                    if nx.shortest_path_length(subgraph, node, node2) > self.maxPathLength:
                        withinMaxDistance = False
                        break
        return withinMaxDistance

    def count_distance_to_active_failures(self, subgraph):
        """
        Count the number of compounds that don't have a minimum-length path to an active
        within the specified limit

        Parameters
        ---------
        subgraph : NetworkX subgraph obj
            the subgraph to check for the max distance between nodes

        Returns
        -------
        failures : int
            Number of nodes that are not within the max distance to any active node
        """

        failures = 0

        hasActives=False
        for node in self.essential_nodes:
            if (subgraph.nodes[node]["active"]):
                hasActives=True
        if (not hasActives):
            return 0     # No actives, so don't bother checking

        paths = nx.shortest_path(subgraph)
        for node in self.essential_nodes:
            if (subgraph.nodes[node]["active"]):
                continue
            ok = False
            for node2 in self.essential_nodes:
                if (subgraph.nodes[node2]["active"]):
                    pathlen = len(paths[node][node2]) - 1
                    if (pathlen <= self.maxDistFromActive):
                        ok = True
            if (not ok):
                failures = failures + 1

        return failures

    def check_distance_to_active(self, subgraph):
        """
        Check to see if we have increased the number of distance-to-active failures

        Parameters
        ---------
        subgraph : NetworkX subgraph obj
            the subgraph to check for the max distance between nodes

        Returns
        -------
        ok : bool
            True if we have not increased the number of failed nodes
        """

        count = self.count_distance_to_active_failures(subgraph)
        failed =  (count > self.distanceToActiveFailures)
        if (failed): logging.info("Rejecting edge deletion on distance-to-actives %d vs %d" % (count,self.distanceToActiveFailures))
        logging.info("Checking edge deletion on distance-to-actives %d vs %d" % (count,self.distanceToActiveFailures))
        return not failed


    def get_graph(self):
        """

        Returns the final generated NetworkX graph

        """

        return self.resultGraph


def get_similarity(moli, molj, options = None):
    moli, molj = copy.deepcopy(moli), copy.deepcopy(molj)
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
    ecr_score = ecr(moli, molj)
    MC = mcs.MCS(
        moli, molj, time=options['time'],
        verbose=options['verbose'],
        threed=options['threed'],
        max3d=options['max3d'],
        element_change=options['element_change'],
        seed=options['seed'],
        shift=options['shift'],
    )
    MC.all_atom_match_list()
    tmp_scr = ecr_score * MC.mncar() * MC.mcsr() * MC.atomic_number_rule() * MC.hybridization_rule()
    tmp_scr *= MC.sulfonamides_rule() * MC.heterocycles_rule() * MC.transmuting_methyl_into_ring_rule()
    tmp_scr *= MC.transmuting_ring_sizes_rule()
    strict_scr = tmp_scr * 1  # MC.tmcsr(strict_flag=True)
    return strict_scr

def test(args):
    db_mol = lomap.DBMolecules(args.input_dir, output=True)
    similarity_threshold = args.similarity_threshold

    strict, loose = db_mol.build_matrices()
    nx_graph = db_mol.build_graph()
    new_graph = nx_graph.copy()
    initial_num = len(new_graph.nodes)
    for node in new_graph:
        if new_graph.nodes[node].get("intermediate") is None:
            new_graph.nodes[node]["intermediate"] = False
    node_mols = {node:AllChem.RemoveHs(db_mol[node].getMolecule()) for node in nx_graph.nodes}
    options = db_mol.options

    options["chunk_mode"]=True
    options["chunk_scale"]=10
    options["node_mode"]=False
    added_edges = []
    new_graphs = []
    new_graphs.append(new_graph.copy())
    while True:
        intermediate_num = len(new_graph.nodes) - initial_num
        if args.max_intermediate > 0 and intermediate_num > args.max_intermediate:
            print("Max number of intermediates reached")
            break
        bad_edges = []
        bad_sims = []
        for u,v,d in new_graph.edges(data=True):
            similarity = d["similarity"]
            if similarity < similarity_threshold:
                bad_edges.append((u,v))
                bad_sims.append(similarity)
        sort_indices = np.argsort(bad_sims)
        bad_edges = [bad_edges[i] for i in sort_indices]

        # Get the edge with the lowest similarity
        found = False
        for edge in bad_edges:
            if (edge[0], edge[1]) in added_edges or (edge[1], edge[0]) in added_edges:
                continue
            added_edges.append((edge[0], edge[1]))
            source_node, target_node = edge
            found = True
            break
        if not found:
            print("No more edges to add")
            break

        source_ligand = node_mols[source_node]
        target_ligand = node_mols[target_node]
        search_intm = SearchIntermediates(source_ligand, target_ligand, max_intermediate=50)
        intermediates = search_intm.search()
        intermediates_avail = intermediates # execute_ligand_preparation(intermediates, extract_same_formal_charge=True)
        for m in intermediates_avail:
            Chem.AssignStereochemistryFrom3D(m)
        if Chem.MolToSmiles(source_ligand) != Chem.MolToSmiles(intermediates_avail[0]):
            raise ValueError("Source ligand has changed")
        if Chem.MolToSmiles(target_ligand) != Chem.MolToSmiles(intermediates_avail[1]):
            raise ValueError("Target ligand has changed")
        mapGen = MapGenerator(intermediates_avail, jobs=-1, maxOptimalPathLength=4, optimal_path_mode=True, minScoreThreshold=0.2)
        pairgraph = mapGen.build_map()

        exisiting_smiles = [Chem.MolToSmiles(node_mols[node]) for node in new_graph.nodes]
        generated_intermediate_names = {i: data["label"] for i, data in pairgraph.nodes(data=True) if data["label"]}
        generated_intermediates = {i: mapGen.intermediate_list[mapGen.intermediate_names.index(name)] for i,name in generated_intermediate_names.items()}


        node_idx = max(new_graph.nodes)+1
        node_remapping = {}
        additional_intermediates = {}
        for i, mol in generated_intermediates.items():
            smiles = mol.GetProp("smiles")
            if smiles not in exisiting_smiles:
                new_graph.add_node(node_idx, ID=node_idx, fname_comp = generated_intermediate_names[i], active=False, intermediate=True)
                node_remapping[i]=node_idx
                mol.SetProp("_Name", "Intermediate-{:03d}".format(node_idx))
                additional_intermediates[i] = mol
                node_mols[node_idx] = mol
                node_idx+=1
            else:
                found_node = False
                for node in new_graph.nodes:
                    if Chem.MolToSmiles(node_mols[node]) == smiles:
                        node_remapping[i] = node
                        found_node = True
                        break
                if not found_node:
                    raise ValueError("Node not found")

        if not additional_intermediates:
            print("No additional intermediates to add")
            break

        # Add intermiediate edges
        for u,v in pairgraph.edges:
            similarity = get_similarity(intermediates_avail[u], intermediates_avail[v], options)
            new_graph.add_edge(node_remapping[u], node_remapping[v], similarity=similarity, strict_flag=True)

        # Add edges with additional intermediates
        for u in new_graph:
            for j in additional_intermediates:
                v = node_remapping[j]
                if u==v or new_graph.get_edge_data(u,v):
                    continue
                similarity = get_similarity(node_mols[u], node_mols[v])
                if similarity>0.4:
                    new_graph.add_edge(u, v, similarity=similarity, strict_flag=True)

        # Add edges with additional intermediates
        new_graph = IntermediateGraphGen(new_graph, options).get_graph()
        new_graphs.append(new_graph.copy())
        print("=====================================")
        print("Target edge: ", source_node, target_node)
        print("Remapping: ", node_remapping)
        print("Number of nodes: ", len(new_graph.nodes))
        print("Number of edges: ", len(new_graph.edges))
        print("Numeber of additional intermediates: ", len(additional_intermediates))
        print("=====================================")

    # Save
    import os
    import pickle
    with open(os.path.join(args.output_dir, "intermediate_graphs.pkl"), "wb") as f:
        pickle.dump(new_graphs, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--similarity_threshold", type=float, default=0.6)
    parser.add_argument("--max_intermediate", type=int, default=-1, help="Maximum number of intermediates to add. -1 means no limit")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    test(args)

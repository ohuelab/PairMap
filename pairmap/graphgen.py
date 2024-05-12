
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import os
import numpy as np
import networkx as nx
from tqdm import tqdm

from utils import find_bridges

import logging

class IntermediateMapper:
    def __init__(self, intermediate_dict, intermediate_traces, options):
        self.intermediate_dict = intermediate_dict
        self.intermediate_names = list(intermediate_dict.keys())
        self.intermediate_traces = intermediate_traces

        self.source_node = 0
        self.target_node = 1
        self.options = options

        # Optimal path parameters
        self.maxOptimalPathLength = options.maxOptimalPathLength
        self.roughtMaxPathLength = options.roughtMaxPathLength

        # Pairmap parameters
        self.cycleLength = options.cycleLength
        self.maxPathLength = options.maxPathLength
        self.chunkScale = options.chunkScale
        self.minScoreThreshold = options.minScoreThreshold

        self.image_dir = options.image_dir

    def make_graph(self, min_score):
        graph = nx.Graph()
        for i, key in enumerate(self.intermediate_dict):
            imagefile=os.path.join(self.image_dir,key+'.png')
            graph.add_node(i)
            graph.nodes[i]['label'] = key
            graph.nodes[i]['image'] = imagefile
            # set edges
            for u, v, score in self.intermediate_traces:
                round_score = np.round(score, decimals=2)
                if round_score >= min_score:
                    graph.add_edge(u, v, score=round_score)
        return graph

    def find_optimal_path(self):
        graph = self.make_graph(self.roughScoreThreshold)
        source_node, target_node = self.source_node, self.target_node
        has_path = nx.has_path(graph, source_node, target_node)
        if has_path:
            path_length = nx.shortest_path_length(graph, source_node, target_node)
            if path_length > self.roughtMaxPathLength:
                has_path = False
        if not has_path:
            graph = self.make_graph(self.minScoreThreshold)
            has_path = nx.has_path(graph, source_node, target_node)
            if not has_path:
                raise Exception('No path found, please check the input.')

        all_simple_paths = list(nx.all_simple_paths(graph, source_node, target_node, cutoff=self.maxOptimalPathLength))

        path_scores_list = []
        for i, path in enumerate(all_simple_paths):
            path_scores = [graph.get_edge_data(path[i],path[i+1])['score'] for i in range(len(path)-1)]
            path_scores_list.append(sorted(path_scores))

        sorted_indices = sorted(enumerate(path_scores_list), key=lambda x: [min(x[1]), len(x[1])], reverse=True)
        argsort_indices = [index for index, _ in sorted_indices]
        sorted_path_scores_list = [value for _, value in sorted_indices]
        max_score = -np.inf
        max_path_scores = None
        min_dist = np.inf
        for i, path_scores in enumerate(sorted_path_scores_list):
            dist = len(path_scores)
            score=min(path_scores)
            if score>=max_score:
                if dist < min_dist or score>max_score or path_scores>max_path_scores:
                    max_score = score
                    max_path_scores = path_scores
                    min_dist = dist
                    best_idx = argsort_indices[i]
        found_path = all_simple_paths[best_idx]
        self.found_path = found_path
        self.found_links = [(found_path[i], found_path[i+1]) if found_path[i]<found_path[i+1] else (found_path[i+1], found_path[i]) for i in range(len(found_path)-1)]
        return self.found_path

    def get_cycled_nodes(self, graph):
        all_simple_cycles = list(nx.simple_cycles(graph, length_bound=self.cycleLength))
        unique_nodes = set()
        for path in all_simple_cycles:
            if any([node in self.found_path[1:-1] for node in path]):
                unique_nodes.update(path)
        cycled_nodes = set(self.found_path).intersection(unique_nodes)
        return cycled_nodes

    def get_cycled_edges(self, graph):
        bridges = find_bridges(graph)
        cycled_found_links = set(self.found_links).difference(bridges)
        return cycled_found_links

    def check_node_cycle_covering(self, graph):
        return len(self.initialCycledNodesSet.difference(self.get_cycled_nodes(graph)))==0

    def check_edge_cycle_covering(self, graph):
        return len(self.initialCycledEdgesSet.difference(self.get_cycled_edges(graph)))==0

    def check_constraints(self, graph):
        constraintsMet = True
        if constraintsMet:
            constraintsMet = self.check_node_cycle_covering(graph)
        if constraintsMet:
            constraintsMet = self.check_edge_cycle_covering(graph)
        return constraintsMet

    def get_main_subgraph(self, graph):
        subgraphs = list(nx.connected_components(graph))
        subgraph = graph.subgraph([])
        for nodes in subgraphs:
            if all([node in nodes for node in self.found_path]):
                subgraph = graph.subgraph(nodes)
                break
        is_invalid = not all([node in subgraph.nodes for node in self.found_path])

        if is_invalid:
            raise Exception('invalid graph: get_main_subgraph')
        return subgraph

    def get_reachable_subgraph(self, graph):
        all_simple_paths = list(nx.all_simple_paths(graph, self.source_node, self.target_node, cutoff = self.maxPathLength))
        unique_nodes = set()
        unique_nodes.update(self.found_path)
        for path in all_simple_paths:
            unique_nodes.update(path)
        subgraph = graph.subgraph(unique_nodes)

        is_invalid = not all([node in subgraph.nodes for node in self.found_path])
        if is_invalid:
            raise Exception('invalid graph: get_reachable_subgraph')
        return subgraph

    def get_cycle_subgraph(self, graph):
        all_simple_cycles = list(nx.simple_cycles(graph, length_bound=self.cycleLength))
        unique_nodes = set()
        unique_nodes.update(self.found_path)
        for path in all_simple_cycles:
            if any([node in self.found_path for node in path]):
                unique_nodes.update(path)
        subgraph = graph.subgraph(unique_nodes)
        is_invalid = not all([node in subgraph.nodes for node in self.found_path])
        if is_invalid:
            raise Exception('invalid graph: get_cycle_subgraph')
        return subgraph

    def generate_initial_graph(self):
        graph = self.make_graph(self.minScoreThreshold)
        for u,v in graph.edges:
            graph[u][v]['found_path']=False
        for i in range(len(self.found_path)-1):
            u=self.found_path[i]
            v=self.found_path[i+1]
            graph[u][v]['found_path']=True
        return graph

    def chunk_process(self, edge_chunk, data_chunk, chunk_size, idx):
        subgraph = self.tmp_subgraph
        if self.check_chunk(edge_chunk, data_chunk):
            # Edges can be removed
            return True
        elif chunk_size == 1:
            # The edge cannot be removed
            return False
        else:
            # Re-split when there are multiple chunks
            logging.info('Split: #E={}, {} {}'.format(len(subgraph.edges()), idx, idx + chunk_size))
            # Run chunk_process recursively with smaller chunks
            chunk_size = max(chunk_size // self.chunkScale, 1)
            crt=0
            while crt < len(edge_chunk):
                edge_chunk_in = []
                data_chunk_in = []
                while len(edge_chunk_in) < chunk_size and crt < len(edge_chunk):
                    u,v = edge_chunk[crt]
                    if subgraph.get_edge_data(u,v):
                        edge_chunk_in+=[edge_chunk[crt]]
                        data_chunk_in+=[data_chunk[crt]]
                    crt+=1
                ret = self.chunk_process(edge_chunk_in, data_chunk_in, chunk_size, idx+crt)
                # If unremovable edges are found, try to check the rest of the chunk
                if not ret:
                    edge_chunk_x = [(u,v) for u,v in edge_chunk[crt:] if subgraph.get_edge_data(u,v) is not None]
                    data_chunk_x = [d for (u,v), d in zip(edge_chunk[crt:], data_chunk[crt:]) if subgraph.get_edge_data(u,v) is not None]
                    if self.check_chunk(edge_chunk_x, data_chunk_x):
                        # Remain edges can be removed
                        break
            return True

    # Check constraints by chunk
    def check_chunk(self, edge_chunk, data_chunk):
        subgraph = self.tmp_subgraph
        removables = [d['score'] < 1.0 and not d['found_path'] for d in data_chunk]
        if not all(removables):
            if not any(removables):
                logging.info('Skip (score=1.0): {}'.format(len(edge_chunk)))
                return True
            # Restore edges because they contain edges that cannot be removed
            return False
        else:
            # Remove edges in the chunk and check constraints
            subgraph.remove_edges_from(edge_chunk)
            exgraph = self.get_reachable_subgraph(subgraph)
            exgraph = self.get_cycle_subgraph(exgraph)
            exgraph = self.get_main_subgraph(exgraph)
            is_invalid = not all([node in exgraph.nodes for node in self.found_path])
            if is_invalid:
                for (i, j), d in zip(edge_chunk, data_chunk):
                    subgraph.add_edge(i, j, **d)
                return False
            satisfied = self.check_constraints(exgraph)
            if not satisfied:
                for (i, j), d in zip(edge_chunk, data_chunk):
                    subgraph.add_edge(i, j, **d)
                return False
            logging.info('Removed: {}'.format(len(edge_chunk)))
            subgraph = exgraph.copy()
            logging.info('#E={}, #N={}'.format(len(subgraph.edges()), len(subgraph)))
            self.tmp_subgraph = subgraph
            return True

    def build_map(self):
        subgraph = self.generate_initial_graph()

        self.scoresList = list(subgraph.edges(data='score'))
        self.scoresList.sort(key=lambda entry: entry[2])

        edges = [(i, j) for i, j, d in self.scoresList]
        data = [subgraph[i][j] for i, j, d in self.scoresList]
        chunk_size = self.chunkScale **int(np.log(len(self.scoresList))/np.log(self.chunkScale))

        found_path = self.found_path
        self.initialCycledNodesSet = self.get_cycled_nodes(subgraph)
        self.initialCycledEdgesSet = self.get_cycled_edges(subgraph)

        exgraph = self.get_reachable_subgraph(subgraph)
        exgraph = self.get_cycle_subgraph(exgraph)
        exgraph = self.get_main_subgraph(exgraph)
        is_invalid = not all([node in exgraph.nodes for node in found_path])
        if is_invalid:
            raise Exception('invalid initial graph: get_reachable_subgraph')
        else:
            subgraph = exgraph.copy()

        self.tmp_subgraph = subgraph
        crt=0
        while crt < len(data):
            edge_chunk = []
            data_chunk = []
            while len(edge_chunk) < chunk_size and crt < len(data):
                subgraph = self.tmp_subgraph
                u,v = edges[crt]
                if subgraph.get_edge_data(u,v):
                    edge_chunk+=[edges[crt]]
                    data_chunk+=[data[crt]]
                crt+=1
            self.chunk_process(edge_chunk, data_chunk, chunk_size, crt)

        tmp_graph = self.tmp_subgraph.copy()
        self.scoresList = list(tmp_graph.edges(data='score'))
        self.scoresList.sort(key=lambda entry: entry[2])
        for u, v, _ in tqdm(self.scoresList):
            edge_data = tmp_graph.get_edge_data(u,v)
            if edge_data == None or edge_data['found_path']:
                continue
            tmp_graph.remove_edge(u,v)
            satisfied = self.check_constraints(tmp_graph)
            if not satisfied:
                tmp_graph.add_edge(u,v, **edge_data)

        exgraph = tmp_graph.copy()
        exgraph = self.get_main_subgraph(exgraph)
        tmp_graph = exgraph

        self.final_graph = tmp_graph.copy()

        return self.final_graph

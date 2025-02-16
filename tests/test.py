import argparse
from pairmap import IntermediateGraphManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    graph_manager = IntermediateGraphManager(args.config)
    new_graphs, node_mols = graph_manager.run()
    final_graph = new_graphs[-1]
    print("Final graph:")
    for u, v in final_graph.edges():
        node_u = final_graph.nodes[u]
        node_v = final_graph.nodes[v]
        print("{}-{} {:.3f}". format(node_u["NAME"], node_v["NAME"], final_graph.get_edge_data(u, v)["similarity"]))

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
        print("{}-{} {:.3f}". format(u["NAME"], v["NAME"], final_graph.get_edge_data(u, v)["similarity"]))

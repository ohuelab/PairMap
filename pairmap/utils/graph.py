import networkx as nx
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS

# find the bridges in a graph
def find_bridges(G):
    """
    Find the bridges in a graph.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    Returns
    -------
    bridges : set
        A set of bridges, where each bridge is a pair of nodes.
    """
    bridges = set()
    visited = {u: False for u in G.nodes()}
    low = {u: 0 for u in G.nodes()}
    pre = {u: 0 for u in G.nodes()}
    parent = {u: None for u in G.nodes()}
    count = 0
    for u in G.nodes():
        if not visited[u]:
            stack = [(u, iter(G.neighbors(u)))]
            visited[u] = True
            count += 1
            pre[u] = count
            low[u] = count
            while stack:
                u, children = stack[-1]
                for v in children:
                    if not visited[v]:
                        parent[v] = u
                        stack.append((v, iter(G.neighbors(v))))
                        visited[v] = True
                        count += 1
                        pre[v] = count
                        low[v] = count
                        break
                    elif v != parent[u]:
                        low[u] = min(low[u], pre[v])
                else:
                    stack.pop()
                    if parent[u] is not None:
                        low[parent[u]] = min(low[parent[u]], low[u])
                        if low[u] > pre[parent[u]]:
                            if u < parent[u]:
                                bridges.add((u, parent[u]))
                            else:
                                bridges.add((parent[u], u))
    return bridges

def save_molimages(intermediate_list, image_dir="image_dir", force=False):
    if force and os.path.exists(image_dir):
        for f in os.listdir(image_dir):
            os.remove(os.path.join(image_dir, f))
    elif not force and os.path.exists(image_dir):
        raise ValueError(f'{image_dir} already exists')
    os.makedirs(image_dir, exist_ok=True)
    molsdict={}
    for mol in intermediate_list:
        k = mol.GetProp('_Name')
        smiles = mol.GetProp('smiles')
        molsdict[k]=Chem.MolFromSmiles(smiles)

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

def draw_graph(graph, fontsize=20, imagefile='graph.png', pos = None):
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


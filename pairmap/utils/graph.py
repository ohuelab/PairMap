

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

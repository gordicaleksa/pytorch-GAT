`networkx.all_pairs_shortest_path_length`: 计算未加权图中所有节点之间最短路径的长度

```
def get_dist_matrix(G):
    N = number_of_nodes(G)
    dist_matrix = np.zeros((N, N))

    spl = nx.all_pairs_shortest_path_length(G) ## dict
    for u in spl:
        for v in spl[u]:
            if (u < v): 
                dist_matrix[u][v] = dist_matrix[v][u] = 1
    
    return dist_matrix
```
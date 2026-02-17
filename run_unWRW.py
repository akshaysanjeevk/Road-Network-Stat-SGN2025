import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 
import osmnx as osm
from fn_lib import *
from tqdm import tqdm
import os
plt.rcParams['text.usetex'] = True



def unW_RW_EL(G, n, tmax):
    # preserve original node labels
    original_nodes = list(G.nodes)

    G = nx.convert_node_labels_to_integers(G)

    N = G.number_of_nodes()
    
    idx2node = dict(enumerate(original_nodes))
    
    # adjacency list
    neighbors = {i: list(G.neighbors(i)) for i in range(N)}

    x = np.zeros((n, tmax), dtype=int)
    x[:, 0] = np.random.choice(N, n)

    for t in tqdm(range(1, tmax)):
        xt = x[:, t - 1]
        for i in range(n):
            node = xt[i]
            nbrs = neighbors[node]
            x[i, t] = node if len(nbrs) == 0 else np.random.choice(nbrs)

    return x, idx2node

def unW_RW_EL_nonbacktrack_deg2(G, n, tmax):
    # preserve original node labels
    original_nodes = list(G.nodes)

    G = nx.convert_node_labels_to_integers(G)
    N = G.number_of_nodes()

    idx2node = dict(enumerate(original_nodes))

    # adjacency list
    neighbors = {i: list(G.neighbors(i)) for i in range(N)}

    x = np.zeros((n, tmax), dtype=int)
    x[:, 0] = np.random.choice(N, n)

    for t in tqdm(range(1, tmax)):
        xt_prev = x[:, t - 1]

        for i in range(n):
            u = xt_prev[i]
            nbrs = neighbors[u]
            deg = len(nbrs)

            # isolated node
            if deg == 0:
                x[i, t] = u
                continue

            # no previous history available
            if t == 1:
                x[i, t] = np.random.choice(nbrs)
                continue

            v = x[i, t - 2]  # previous node

            # degree-1 node: forced move
            if deg == 1:
                x[i, t] = nbrs[0]

            # degree-2 node: non-backtracking
            elif deg == 2:
                if v == nbrs[0]:
                    x[i, t] = nbrs[1]
                elif v == nbrs[1]:
                    x[i, t] = nbrs[0]
                else:
                    # came from elsewhere (rare but possible)
                    x[i, t] = np.random.choice(nbrs)

            # all other cases: standard RW
            else:
                x[i, t] = np.random.choice(nbrs)

    return x, idx2node

def PopNodes(X, idx2node, q=20):

    
    counts = np.bincount(X.ravel())
    P = counts / counts.sum()

    # indices with nonzero occupation
    idx = np.nonzero(P)[0]
    probs = P[idx]

    # sort by decreasing probability
    order = np.argsort(probs)[::-1]

    top_idx = idx[order][:q]
    top_probs = probs[order][:q]

    # map back to true labels
    top_nodes = np.array([idx2node[i] for i in top_idx])

    return top_nodes, top_probs

def topNPlotMap(citycode, X, idx2node):
    topN, topPr = PopNodes(X, idx2node, q=20)
    G = OpenGraph(citycode)
    xs = [G.nodes[n]["x"] for n in topN]
    ys = [G.nodes[n]["y"] for n in topN]

    fig, ax = osm.plot_graph(
        G,
        figsize=(10,10),
        node_size=.5,    
        node_color="blue",
        edge_color="grey",
        bgcolor="white",
        show=False,
        close=False
    )

    ax.scatter(
        xs, ys,
        c="red",
        s=5,
        zorder=5,
        label="Top RW nodes"
    )

    ax.legend()
    plt.title(f"{citycode}")
    plt.savefig(f"./topN_maps/{citycode}_RW.pdf")

if __name__ == "__main__": 
    cities = os.listdir("./graph_data") 
    
    # RWcompleted = os.listdir("./RW_data")
    # completed_cities = {fname[3:6] for fname in RWcompleted}
    # cities = [city for city in cities if city[:3] not in completed_cities]
    # print(cities)
    
# _________________________________
#   SINGLE CITY RUN
    city = cities[0][:3] 
    G0 = OpenGraph(city) 
    n = int(1e2); tmax = int(1e2) 
    # X = unW_RW_EL(G0, n, tmax) 
    # np.savez_compressed(f"./RW_data/RW_{city}_n{n}_t{tmax}.npz", arr=X)
    X, idx2node = unW_RW_EL(G0, n, tmax)
    print("check 1/3: simulation executed")
    np.savez_compressed(
        f"./RW_data/RW_{city}_n{n}_t{tmax}.npz",
        X=X,
        idx2node=np.array([idx2node[i] for i in range(len(idx2node))])
    )
    print("fcheck 2/3: ile saved")
    topNPlotMap(city, X, idx2node)
    print("check 3/3: map plotted and saved")
    
    
# _________________________________
#   BULK RUN
    # for city in tqdm(cities, total=len(cities)): 
    #     city = city[:3] 
    #     G0 = OpenGraph(city) 
    #     n = int(1e2); tmax = int(1e2) 
    #     # X = unW_RW_EL(G0, n, tmax) 
    #     # np.savez_compressed(f"./RW_data/RW_{city}_n{n}_t{tmax}.npz", arr=X)
    #     X, idx2node = unW_RW_EL(G0, n, tmax)
    #     print("check 1/3: simulation executed")
    #     np.savez_compressed(
    #         f"./RW_data/RW_{city}_n{n}_t{tmax}.npz",
    #         X=X,
    #         idx2node=np.array([idx2node[i] for i in range(len(idx2node))])
    #     )
    #     print("fcheck 2/3: ile saved")
    #     topNPlotMap(city, X, idx2node)
    #     print("check 3/3: map plotted and saved")

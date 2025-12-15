import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 
import osmnx as osm
from fn_lib import *
from tqdm import tqdm
import os
plt.rcParams['text.usetex'] = True


# def unW_RW(G0, n, tmax):
#     A = nx.adjacency_matrix(G0).toarray()
#     n0 = np.random.choice(A.shape[0], n)
#     x = np.zeros((n, tmax), dtype=int)
#     x[:, 0] = n0
#     for t in tqdm(range(1, tmax)):

#         xt = x[:, t-1]         
#         Aw = A[xt]             
#         deg = Aw.sum(axis=1)   
#         # # optional safety (isolated nodes)
#         # deg[deg == 0] = 1
#         T = Aw / deg[:, None]
#         r = np.random.rand(n)
#         Tcdf = np.cumsum(T, axis=1)
#         x[:, t] = (Tcdf >= r[:, None]).argmax(axis=1) #ITS updation
#     return x

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

import numpy as np

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
    
    for city in tqdm(cities, total=len(cities)): 
        city = city[:3] 
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

# def unWRWplot(n, tmax, source="graph",G0=None, filename=None):
#     if source=="graph":
#         X = unW_RW(G0, n, tmax)
#     elif source == "file":
#         X = np.load(f"./{filename}")['arr']
#     N = np.unique(X)
#     n = X.shape[0]

#     P = np.zeros((N, tmax), dtype=float)

#     rows = X.ravel()                                  # node indices
#     cols = np.repeat(np.arange(tmax), n)              # time indices

#     np.add.at(P, (rows, cols), 1)
#     P /= n

#     plt.imshow(P, vmin=0, vmax=1, aspect="auto")
    
#     return None
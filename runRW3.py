import numpy as np
import networkx as nx
import osmnx as osm
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from fn_lib import *

plt.rcParams['text.usetex'] = True

# def RW3(G, n, tmax):
#     og_nodes = list(G.nodes)
#     G = nx.convert_node_labels_to_integers(G)
#     N = G.number_of_nodes()
#     idx2nodes = dict(enumerate(og_nodes))

#     nbr_nodes = [[] for _ in range(N)]
#     wts = [[] for _ in range(N)]

#     for u, v, d in G.edges(data=True):
#         nbr_nodes[u].append(v)
#         wts[u].append(d['weight'])
#         nbr_nodes[v].append(u)       # remove if directed
#         wts[v].append(d['weight'])

#     nbr_nodes = [np.array(n, dtype=int) for n in nbr_nodes]
#     wts = [np.array(w, dtype=float) for w in wts]

#     # --- initialize walkers ---
#     x = np.zeros((n, tmax), dtype=int)
#     x[:,0] = np.random.choice(N, n)

#     # --- run RW ---
#     for t in tqdm(range(1, tmax)):
#         xt_i = x[:, t-1]
#         xt_h = x[:, t-2] if t > 1 else None

#         for i in range(n):
#             u = xt_i[i]
#             nbr = nbr_nodes[u]
#             weight = wts[u]

#             # avoid previous node if possible
#             if xt_h is not None and nbr.shape!=1:
#                 mask = nbr != xt_h[i]
#                 if mask.any():
#                     nbr = nbr[mask]
#                     weight = weight[mask]

#             # sample next node
#             pvec = weight / weight.sum()
#             x[i, t] = np.random.choice(nbr, p=pvec)

#     return x, idx2nodes

def RW3_fast(G, n, tmax, seed=None):
    rng = np.random.default_rng(seed)

    # relabel nodes to integers for simplicity
    og_nodes = list(G.nodes)
    G = nx.convert_node_labels_to_integers(G)
    N = G.number_of_nodes()
    
    # --- precompute neighbor lists and edge weights ---
    nbr_nodes = [[] for _ in range(N)]
    wts = [[] for _ in range(N)]
    for u, v, d in G.edges(data=True):
        weight = d.get('weight', 1.0)  # default weight = 1
        nbr_nodes[u].append(v)
        wts[u].append(weight)
        if not G.is_directed():
            nbr_nodes[v].append(u)
            wts[v].append(weight)
    
    nbr_nodes = [np.array(n, dtype=int) for n in nbr_nodes]
    wts = [np.array(w, dtype=float) for w in wts]

    # --- initialize walkers ---
    x = np.zeros((n, tmax), dtype=int)
    x[:,0] = rng.choice(N, n)

    # --- run the random walks ---
    for t in tqdm(range(1, tmax), desc="Random Walk"):
        xt_i = x[:, t-1]            # current positions
        xt_h = x[:, t-2] if t > 1 else None  # previous positions

        for i in range(n):
            u = xt_i[i]
            nbr = nbr_nodes[u]
            weight = wts[u]

            # --- non-backtracking logic ---
            if xt_h is not None:
                # mask out previous node only if node has degree > 1
                if len(nbr) > 1:
                    mask = nbr != xt_h[i]
                    nbr = nbr[mask]
                    weight = weight[mask]

            # --- sample next node ---
            pvec = weight / weight.sum()
            x[i, t] = rng.choice(nbr, p=pvec)

    return x, idx2node

def PopNodes(X, idx2node, q=20):

    
    counts = np.bincount(X.ravel())
    P = counts / counts.sum()
    idx = np.nonzero(P)[0]
    probs = P[idx]

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
    
    n = int(1e2); tmax = int(1e2)
# _____________________________________________________
# FOR POOLED RUN:
    # occdist, occdist_ax = plt.subplots(figsize=(5,5))
    # cities = os.listdir("./graph_data")
    # for city in tqdm(cities, total=len(cities)):
    #     G0 = OpenGraph(city)
        
    #     X, idx2node = RW3(G0, n, tmax)
    #     np.savez_compressed(
    #         f"./RW_data/RW3_{city}_n{n}_t{tmax}.npz",
    #         X=X,
    #         idx2node=np.array([idx2node[i] for i in range(len(idx2node))])
    #     )
        
    #     topNPlotMap(city, X, idx2node)
    
# ____________________________________________________
#  FOR SINGLE RUN

    city1 = "CHE"
    G0 = OpenGraph(city1)
    G0 = osm.add_edge_speeds(G0)
    G0 = osm.add_edge_travel_times(G0)
    W_list = np.zeros(nx.number_of_edges(G0))
    for i,( u, v, k, data) in enumerate(G0.edges(keys=True, data=True)):
        data["weight"] = data["travel_time"]
        W_list[i] = data["travel_time"]
    print("1/3 graph obtained. weights initiated")
    X, idx2node = RW3_fast(G0, n, tmax)
    print("2/3 rws completed.")
    np.savez_compressed(
        f"./RW_data/RW3_{city1}_n{n}_t{tmax}.npz",
        X=X,
        idx2node=np.array([idx2node[i] for i in range(len(idx2node))])
    )
    print("3/3 data saved. plot initiated")
    topNPlotMap(city1, X, idx2node)


import numpy as np 
import osmnx as osm
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from fn_lib import *

plt.rcParams['text.usetex'] = True


def OccPDF(X):
    counts = np.bincount(X.ravel())
    P = counts/counts.sum()
    P = P[P>0]
    rankdP = np.sort(P)[::-1]
    
    ranks = np.arange(1, len(rankdP)+1)
    
    fig,ax = plt.subplots(figsize=(5,5))
    plt.loglog(ranks, rankdP, 'o-',
                markersize=5, 
                fillstyle='none',
                color = 'blue')
    ax.set_xlabel("Rank")
    ax.set_ylabel("P")
    
    plt.show
    return fig, ax

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
        edge_linewidth=.5, 
        node_size=.5,    
        node_alpha=.5,
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
    # plt.show()
    plt.savefig(f"./topN_maps/{citycode}_RW.pdf")
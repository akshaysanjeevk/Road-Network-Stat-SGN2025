import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import osmnx as osm
import os
from tqdm import tqdm

def ExtractGraph(City, code):
    G = osm.graph_from_place(f"{City}, India", network_type="drive")

    with open(f"./graph_data/{code}.gpickle", "wb") as f:
        pickle.dump(G, f)
    V,E = nx.number_of_nodes(G), nx.number_of_edges(G)
    print(f"V = {V} \n E= {E}")
    return G 

def OpenGraph(citycode):
    with open(f"./graph_data/{citycode}.gpickle", "rb") as file:
        G = pickle.load(file)
    # V,E = nx.number_of_nodes(G), nx.number_of_edges(G)
    # print(f"\n #Vertices = {V} \n #Edges = {E}")
    return G

def kPDF(G, city, binsize=1, save=False):
    d = np.array([deg for _, deg in G.degree()])
    dmin, dmax = d.min(), d.max()
    binsize=1
    bins = np.arange(dmin, dmax+binsize, binsize)
    counts, edges = np.histogram(d, bins=bins)
    pk = counts/counts.sum()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(edges[1:],pk, '-o',
            color='black', fillstyle='none',
            linewidth=1)
    ax.set_title(f"{city}")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$p(k)$")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 12)
    if save == True:
        fig.savefig(f"./kPDF/{city}.png", dpi=450)
    else:
        None
    return fig


def Weights2Time(code, hist=False):
    G1 = OpenGraph(code)
    G1 = osm.add_edge_speeds(G1)
    G1 =osm.add_edge_travel_times(G1)
    W_list = np.zeros(nx.number_of_edges(G1))
    for i,( u, v, k, data) in enumerate(G1.edges(keys=True, data=True)):
        data["weight"] = data["travel_time"]
        W_list[i] = data["travel_time"]
    if hist:
        return np.histogram(W_list)
    else:
        return G1, W_list
    
def wPDF(cities):
    fig, ax = plt.subplots(4,4, figsize=(10,10), sharey = True, sharex=True)
    for i, city in tqdm(enumerate(cities), total=len(cities)):
        i, j = i//4, i%4
        counts, binlabels = Weights2Time(city, hist=True)
        ax[i, j].set_xscale('log')
        ax[i, j].set_yscale('log')
        ax[i, j].plot(binlabels[1:], counts/counts.sum(),'o',markersize=4, linewidth=2)
        # ax[i, j].set_ylim(0, 1)
        # ax[i, j].set_xlim(0,1)
    return fig, ax



# def
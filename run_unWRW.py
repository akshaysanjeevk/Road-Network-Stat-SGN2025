import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 
import osmnx as osm
from fn_lib import *
from tqdm import tqdm
plt.rcParams['text.usetex'] = True


def unW_RW(G0, n, tmax):
    A = nx.adjacency_matrix(G0).toarray()
    n0 = np.random.choice(A.shape[0], n)
    x = np.zeros((n, tmax), dtype=int)
    x[:, 0] = n0
    for t in tqdm(range(1, tmax)):

        xt = x[:, t-1]         
        Aw = A[xt]             
        deg = Aw.sum(axis=1)   
        # # optional safety (isolated nodes)
        # deg[deg == 0] = 1
        T = Aw / deg[:, None]
        r = np.random.rand(n)
        Tcdf = np.cumsum(T, axis=1)
        x[:, t] = (Tcdf >= r[:, None]).argmax(axis=1) #ITS updation
    return x


def unW_RW_EL(G, n, tmax):
    # Ensure nodes are indexed 0..N-1
    G = nx.convert_node_labels_to_integers(G)
    
    N = G.number_of_nodes()
    
    # Build adjacency list
    neighbors = {i: list(G.neighbors(i)) for i in range(N)}

    x = np.zeros((n, tmax), dtype=int)
    x[:, 0] = np.random.choice(N, n)

    for t in tqdm(range(1, tmax)):
        xt = x[:, t-1]

        for i in range(n):
            node = xt[i]
            nbrs = neighbors[node]

            if len(nbrs) == 0:
                x[i, t] = node
            else:
                x[i, t] = np.random.choice(nbrs)

    return x

def unWRWplot(n, tmax, source="graph",G0=None, filename=None):
    if source=="graph":
        X = unW_RW(G0, n, tmax)
    elif source == "file":
        X = np.load(f"./{filename}")['arr']
    N = np.unique(X)
    n = X.shape[0]

    P = np.zeros((N, tmax), dtype=float)

    rows = X.ravel()                                  # node indices
    cols = np.repeat(np.arange(tmax), n)              # time indices

    np.add.at(P, (rows, cols), 1)
    P /= n

    plt.imshow(P, vmin=0, vmax=1, aspect="auto")
    
    return None




if __name__ == "__main__":
    city = "CHE"
    G0 = OpenGraph(city)    
    n = int(1e3); tmax = int(1e3)
    X = unW_RW_EL(G0, n, tmax)
    np.savez_compressed(f"./RW_{city}_n{n}_t{tmax}.npz", arr=X)
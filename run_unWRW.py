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

def unWRWplot(G0, n, tmax):
    X = unW_RW(G0, n, tmax)
    N = A0.shape[0]
    n = X.shape[0]

    P = np.zeros((N, tmax), dtype=float)

    rows = X.ravel()                                  # node indices
    cols = np.repeat(np.arange(tmax), n)              # time indices

    np.add.at(P, (rows, cols), 1)
    P /= n

    plt.imshow(P, vmin=0, vmax=1, aspect="auto")
    
    return None



if __name__ == "__main__":
    city = "test"
    G0 = OpenGraph(city)    
    n = int(1e3); tmax = int(1e3)
    X = unW_RW(G0, n, tmax)
    np.savez_compressed(f"./RW_{city}_n{n}_t{tmax}.npz", arr=X)
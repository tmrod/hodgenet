import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# helpers.py
# functions for reading, processing, and visualizing graph flow data

# flow/graph processing functions

def incidence_matrix(G):
    B1 = nx.incidence_matrix(G, oriented=True)
    B1 = B1.todense()
    return np.array(B1).astype(np.float32)

def graph_laplacian(G):
    B1 = incidence_matrix(G)
    return (B1 @ B1.T).astype(np.float32)

def hodge_laplacian(G):
    B1 = incidence_matrix(G)
    return (B1.T @ B1).astype(np.float32)

def normalized_hodge_laplacian(G):
    L1 = hodge_laplacian(G)
    D = np.sum(np.abs(L1), axis=1) - 2
    Dinv = np.diag(1/D)
    sqrtDinv = np.sqrt(Dinv)
    return sqrtDinv @ L1 @ sqrtDinv

def linegraph_laplacian(G):
    L1 = hodge_laplacian(G)
    D = np.sum(np.abs(L1), axis=1) - 2
    np.fill_diagonal(L1, 0)
    S = np.diag(D) - np.abs(L1)
    Dinv = np.diag(1/D)
    sqrtDinv = np.sqrt(Dinv)
    return sqrtDinv @ S @ sqrtDinv

def node_shift_operator(G):
    A = nx.adjacency_matrix(G.to_undirected(), weight=None).todense().astype(np.float32)
    A = np.array(A) + np.eye(G.number_of_nodes()).astype(np.float32)
    D = np.sum(A, axis=1)
    sqrtDinv = np.diag(np.sqrt(1/D))
    S = (sqrtDinv @ A @ sqrtDinv).astype(np.float32)
    S = np.eye(G.number_of_nodes()) - S/np.max(np.linalg.eigvalsh(S))
    return S.astype(np.float32)

def get_flow(G, strn='weight'):
    f = nx.get_edge_attributes(G, strn)
    return np.array(list(f.values())).astype(np.float32)

def gradient_flow(G, strn='weight'):
    L1 = hodge_laplacian(G)
    f = get_flow(G, strn)
    grad = L1 @ np.linalg.pinv(L1) @ f
    return grad.astype(np.float32)

def node_potentials(G, f):
    B1 = incidence_matrix(G)
    L1 = hodge_laplacian(G)
    pot = B1 @ np.linalg.pinv(L1, rcond=0.001) @ f
    pot -= np.min(pot)
    return pot.astype(np.float32)

# random mask generation
def random_mask(G, N):
    E = G.number_of_edges()
    if N > E:
        N = E
    idxs = np.random.permutation(E)[0:N]

    mask = np.zeros(E)
    mask[idxs] = 1
    mask = np.diag(mask)

    return mask.astype(np.float32)
    
# data reading functions
def graph_from_tntp(fname):
    df = pd.read_csv(fname, header=0, delim_whitespace=True)
    df.rename(columns={'Volume' : 'weight'}, inplace = True)
    G = nx.from_pandas_edgelist(df, source='From', target='To',
                                edge_attr='weight', create_using=nx.DiGraph())

    #for (i, j) in list(G.edges):
    #    if i < j:
    #        if G.has_edge(j, i):
    #            G[j][i]['weight'] -= G[i][j]['weight']
    #        else:
    #            G.add_edge(j, i, weight=-G[i][j]['weight'])
    #        G.remove_edge(i,j)

    return G

# convex optimization functions
def gen_convopt_gradient(G, M, f, gamma):
    B = incidence_matrix(G)
    E = G.number_of_edges()
    I = np.eye(E)
    
    X = np.concatenate((B@(I-M), gamma*I), axis=0)
    Y = np.concatenate((B@M@f, np.zeros(E)))
    
    def grad(fU):
        return 2*X.T@(X@fU + Y)
    
    return grad

def gdescent(grad, f0, alpha, iters):
    f = f0.copy()
    for i in range(iters):
        g = grad(f)
        f -= alpha*g
        print(f'iter {i} grad {np.linalg.norm(g)}\r', end='')
    return f

def interpolate(G, M, f, gamma, alpha, iters):
    grad = gen_convopt_gradient(G, M, M@f, gamma)
    E = G.number_of_edges()
    f_interp = gdescent(grad, M@f, alpha, iters)
    return f_interp
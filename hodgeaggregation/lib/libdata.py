import networkx as nx
import numpy as np
from scipy.linalg import null_space
from scipy.sparse.linalg import eigs

def SBM(num_communities=5, size_communities=20, p=0.8, q=0.2):
    sizes = size_communities*np.ones(num_communities).astype(int)
    ps = (p-q)*np.eye(num_communities) + q*np.array(np.ones([num_communities, num_communities]))
    G = nx.stochastic_block_model(sizes, ps)

    max_degree_nodes = []
    for C in range(num_communities):
        nodes_in_C = list(range(C*size_communities, (C+1)*size_communities))
        degrees_in_C = dict(G.degree(nodes_in_C))
        idx = max(degrees_in_C, key=degrees_in_C.get)
        max_degree_nodes.append(idx)

    assert len(max_degree_nodes)==num_communities

    max_degree_edges = []
    edgelist = list(G.edges())
    for C, node in zip(range(num_communities), max_degree_nodes):
        nodes_in_C = list(range(C*size_communities, (C+1)*size_communities))
        for potential_neighbor in nodes_in_C:
            edge = (node, potential_neighbor)
            if edge in edgelist:
                max_degree_edges.append(edgelist.index(edge))
                break
            elif tuple(reversed(edge)) in edgelist:
                max_degree_edges.append(edgelist.index(tuple(reversed(edge))))
                break

    assert len(max_degree_nodes)==len(max_degree_edges)
    
    return G, max_degree_nodes, max_degree_edges

# graph matrix creation

def hodgelaplacian(G):
    B = nx.incidence_matrix(G)
    L1 = B.T @ B
    return L1

def scaledhodgelaplacian(G):
    L1 = hodgelaplacian(G)
    maxeig = np.linalg.eigvalsh(L1.todense())[-1]
    return L1 / maxeig
    
def linegraphlaplacian(G):
    L1 = hodgelaplacian(G)
    Llg = -abs(L1)
    for i in range(Llg.shape[0]):
        Llg[i,i] = 0
    D = -np.array(np.sum(Llg, axis=0))
    for i in range(Llg.shape[0]):
        Llg[i,i]=D[0,i]
    return Llg

def scaledlinegraphlaplacian(G):
    Llg = linegraphlaplacian(G)
    maxeig = np.linalg.eigvalsh(Llg.todense())[-1]
    return Llg/maxeig

def incidencematrix(G):
    return nx.incidence_matrix(G)

def adjacencymatrix(G):
    return nx.adjacency_matrix(G)

def scaledadjacencymatrix(G):
    A = adjacencymatrix(G)
    maxeig = np.max(np.abs(np.linalg.eigvalsh(A.todense())))
    return A / maxeig

def laplacianmatrix(G):
    return nx.laplacian_matrix(G)

def scaledlaplacianmatrix(G):
    L = laplacianmatrix(G)
    maxeig = np.max(np.linalg.eigvalsh(L.todense()))
    return L / maxeig
    
# signal creation

def generateflows(G, diffusionmatrix, maxdiffuse, diffusetimelist, nodesourcelist):
    assert len(diffusetimelist) == len(nodesourcelist)
    
    # list of flow inducing operators for all diffusion times
    B = incidencematrix(G)
    ZS = np.array([B.T@np.linalg.matrix_power(diffusionmatrix, t) for t in range(maxdiffuse)])

    # columns of additive noise
    numsamples = len(diffusetimelist)
    #edgecount = G.size()
    #N = noiseenergy*np.random.randn(edgecount, numsamples)

    # return a E x numsamples array of flows
    # i.e. each column is a flow
    #return np.array([ZS[diffusetimelist[i], :, nodesourcelist[i]] + N[:, i] for i in range(numsamples)]).T
    return np.array([ZS[diffusetimelist[i], :, nodesourcelist[i]] for i in range(numsamples)]).T

# aggregation

def aggregator(gso, samplenodes, N):
    assert gso.shape[0] == gso.shape[1]
    # node sampling matrix : nodes sampled x total number of nodes
    P = np.zeros([len(samplenodes), gso.shape[0]])
    for idx, node in enumerate(samplenodes):
        P[idx, node] = 1
    # aggregation matrix : number of shifts x [shape of gso]
    shifts = [np.eye(gso.shape[0])]
    for i in range(N-1):
        shifts.append(gso @ shifts[-1])
    # combined sampling matrix : number of shifts x nodes sampled x number of nodes
    return P @ np.array(shifts)

#!/usr/bin/python3

# convert Anaheim_flow.tntp to two files
# one with the Hodge Laplacian
# one with the flow vector

import pandas as pd
import networkx as nx
import numpy as np

in_file = 'Anaheim_flow.tntp'
#in_file = 'ChicagoSketch_flow.tntp'
#in_file = 'Barcelona_flow.tntp'
l_file  = 'laplacian.npy'
f_file  = 'flow.npy'

df = pd.read_csv(in_file, header=0, delim_whitespace=True)

G = nx.from_pandas_edgelist(df, source='Tail', target='Head', edge_attr='Volume', create_using=nx.DiGraph())
print('Graph Diameter: ', nx.diameter(nx.line_graph(G.to_undirected())))
B = nx.incidence_matrix(G, oriented=True)
L1 = B.T @ B
L1 = L1.todense().astype(np.float32)
f = np.array(list(nx.get_edge_attributes(G, 'Volume').values())).astype(np.float32)

np.save(l_file, L1)
np.save(f_file, f)

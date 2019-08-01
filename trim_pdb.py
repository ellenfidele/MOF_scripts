import numpy as np
import networkx as nx
import pandas as pd
import biopandas.pdb as ppdb
import itertools as it
import scipy.spatial.distance as sp

input_pdb = "../UFF_params/two_layers.adj.pdb"
output_pdb = '../UFF_params/two_layess.new_resid.pdb'

# basic vectors of box
v1 = np.array([26.0261, 0.0, 0.0])
v2 = np.array([-13.0130, 22.5392, 0.0])
v3 = np.array([0.0, 0.0, 6.7587])


# Read pdb file
pdbin = ppdb.PandasPdb()
pdbin.read_pdb(input_pdb)


d_coords = pdbin.df['ATOM']


# Check if atoms are in the box

A = np.array([v1, v2, v3])

out_of_box_df = pd.DataFrame(columns=d_coords.columns)
out_list = []
for i,n in d_coords.iterrows():
    p = np.array([n['x_coord'], n['y_coord'], n['z_coord']])
    Y = np.dot(np.linalg.inv(A.T), p)
    if max(Y) > 1 or min(Y) <= 0:
        out_of_box_df = out_of_box_df.append(d_coords.iloc[i])
        out_list.append(i)

# drop atoms outside

d_coords = d_coords.drop(out_list)


d_coords_cleaned = d_coords.drop(columns=['blank_1', 'blank_2',
                                               'blank_3', 'blank_4', 'segment_id', 'line_idx']).reset_index(drop=True).fillna('')


with open(output_pdb,'w') as out:
    for ind, row in d_coords_cleaned.iterrows():
        print("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:>4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format(*row), file=out)
    print("END",file=out)





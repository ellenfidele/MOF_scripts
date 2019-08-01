import numpy as np
import networkx as nx
import pandas as pd
import biopandas.pdb as ppdb


input_pdb = "../UFF/3x3x6_unitcells.pdb"
output_pdb = "../UFF/3x3x6_unitcells.new_resid.pdb"
id_map = '../UFF/old_to_new.3x3x6_unitcells.txt'


pdb = ppdb.PandasPdb()
pdb.read_pdb(input_pdb)
d_coords = pdb.df['ATOM']


d_connect = pdb.df['OTHERS'].iloc[1:-1, :-1]


coords = d_coords[['x_coord','y_coord', 'z_coord']].values

G = nx.Graph()


dist = np.sqrt(np.sum((coords[:,None,:]-coords[None,:,:])**2,axis=2))


for i in range(len(coords)):
    for j in range(i,len(coords)):
        if dist[i,j]<1.8:
            G.add_edge(i+1,j+1)

d_coords.insert(loc=1,column='new_atom_number',value=0)


resid = 1
for group in nx.connected_components(G):
    for item in group:
        d_coords.loc[d_coords.atom_number==item, 'residue_number'] = int(resid)
        d_coords.loc[d_coords.atom_number==item, 'residue_name'] = 'MOF'
    resid += 1


d_coords = d_coords.astype({'residue_number': 'int'})


d_coords_sorted = d_coords.sort_values(by=['residue_number', 'atom_name'])
d_coords_sorted.new_atom_number = d_coords_sorted.reset_index(drop=True).index + 1


with open(id_map,'w') as mapfile:
    idmap =  d_coords_sorted[['atom_number', 'new_atom_number']].values
    np.savetxt(mapfile, idmap, header='old_atom_number\tnew_atom_number\n', fmt='%d')


d_coords_sorted_cleaned = d_coords_sorted.drop(columns=['atom_number','blank_1', 'blank_2',
                                               'blank_3', 'blank_4', 'segment_id', 'line_idx']).reset_index(drop=True).fillna('')


# get the max len of CONECT section
len_of_con = [len(d_connect['entry'].values[i].split()) for i in range(d_connect.shape[0])]
print("CONECT max len: %d" %max(len_of_con))


# Insert columns to CONECT section 
d_connect[['0','1','2','3','4', '5', '6']]=pd.DataFrame([['','','','','', '', '']], index=d_connect.index)


# assign the values into corresponding column
for ind, row in d_connect.iterrows():
    val = d_connect['entry'].values[ind-1].split()
    for i in range(7):
        if i<len(val):
            d_connect.loc[ind]['%d' %i]=val[i]
        else:
            d_connect.loc[ind]['%d' %i]=''


# construct a map file for replacing the old atom_number with the new
map_dict = {}
with open(id_map,'r') as m:
    for line in m:
        if not line.startswith('#'):
            (key,val) = line.split()
            map_dict[key] = val

# do the replacement
d_connect_new = d_connect.replace(map_dict)


d_connect_new_clean = d_connect_new.drop(columns=['entry'])


with open(output_pdb,'w') as out:
    for ind, row in d_coords_sorted_cleaned.iterrows():
        print("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:>4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format(*row), file=out)
    for ind, row in d_connect_new_clean.iterrows():
        print("{:7s}{:7s}{:7s}{:7s}{:7s}{:7s}{:7s}".format(*row), file=out)
    print("END",file=out)


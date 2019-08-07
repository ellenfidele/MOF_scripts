#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import networkx as nx
import biopandas.pdb as ppdb
import scipy.spatial.distance as sp
import argparse as argp


parser = argp.ArgumentParser(description='Add missing atoms in the MOF ligand molecules. (for MOF74-Mg only)')
parser.add_argument('--pdbin', help='Full path to input pdb', type=str)
parser.add_argument('--pdbout', help='Full path to output pdb', type=str)
args = parser.parse_args()
#args = parser.parse_args(['--pdbin', '../UFF/MOF_structure/chimera_224.merge.pdb',
#                         '--pdbout', '../UFF/MOF_structure/chimera_224.full.pdb'])


file_in = args.pdbin
file_out = args.pdbout


pdb = ppdb.PandasPdb()
pdb.read_pdb(file_in)
d_coords = pdb.df['HETATM']


d_coords['record_name'] = 'ATOM'
d_coords['residue_name'] = 'MOF'
d_coords['atom_number'] = d_coords.index +1


coords = d_coords[['x_coord','y_coord', 'z_coord']].values


G = nx.Graph()


dist = np.sqrt(np.sum((coords[:,None,:]-coords[None,:,:])**2,axis=2))


for i in range(len(d_coords)):
    for j in range(i, len(d_coords)):
        if ('Mg1' in [d_coords.loc[i,'atom_name'], d_coords.loc[j,'atom_name']] 
            and 'O3' in [d_coords.loc[i,'atom_name'], d_coords.loc[j,'atom_name']]
            and dist[i][j]<2.13 and dist[i][j]>2.09):
            G.add_edge(d_coords.loc[i].atom_number, d_coords.loc[j].atom_number)
#             print(i, j)
        elif dist[i][j]<1.8 and dist[i][j]>0:
            G.add_edge(d_coords.loc[i].atom_number, d_coords.loc[j].atom_number)


d_coords.insert(loc=1,column='new_resid',value=0)


resid = 1
for group in nx.connected_components(G):
#     print(len(group))
    for item in group:
        d_coords.loc[d_coords.atom_number==item, 'new_resid'] = int(resid)
    resid += 1


half = []
full = []
for group in nx.connected_components(G):
#     print(group[0])
    if len(group) != 18:
        half.append(d_coords.iloc[list(group)[0]-1]['new_resid'])
    else:
        full.append(d_coords.iloc[list(group)[0]-1]['new_resid'])


half_full_pair = []
for res in half:
    half_O3 = d_coords.loc[(d_coords['new_resid'] == res) & (d_coords['atom_name']=='O3'),'atom_number'].values[0]
    full_res = []
    for ind, distance in enumerate(dist[half_O3-1,:]):
        if (distance > 15.68 and distance < 15.70  and d_coords.iloc[ind]['atom_name'] == 'O3' 
            and d_coords.iloc[ind].new_resid in full):
            full_res.append(d_coords.iloc[ind].new_resid)
        
#     print("half:" , d_coords.iloc[half_O3-1].new_resid, 'full:', d_coords.iloc[full_O3].new_resid)
    half_full_pair.append([res, full_res[0]])


def add_atoms(half_res, full_res, d_coords):
    half_res_atoms = list(d_coords.loc[d_coords['new_resid']==half_res]['atom_number'].values)
    full_res_atoms = list(d_coords.loc[d_coords['new_resid']==full_res]['atom_number'].values)
    exist = []
    transfrom_vec = []
    for full_id, full_atom in enumerate(full_res_atoms):
        full_coords = d_coords.iloc[full_atom-1]['x_coord':'z_coord'].values
        for half_atom in  half_res_atoms:
            half_coords = d_coords.iloc[half_atom-1]['x_coord':'z_coord'].values
            distance = sp.pdist([full_coords, half_coords])
    #         print(distance)
            if distance > 15.68 and distance < 15.70:
                vec = half_coords - full_coords
                transfrom_vec.append(vec)
#                 print(d_coords.iloc[half_atom-1]['atom_name'], half_atom, d_coords.iloc[full_atom-1]['atom_name'], full_atom , vec)
                exist.append(full_id)
    #             full_res_atoms.remove(full_atom)
    rest = np.delete(full_res_atoms, exist, axis=0)
    trans = np.mean(transfrom_vec, axis=0)

    for atom in rest:
        new_atom = d_coords.iloc[atom-1].copy()
        new_atom['x_coord':'z_coord'] = d_coords.iloc[atom-1]['x_coord':'z_coord'] + trans
        new_atom['new_resid'] = half_res
        d_coords = d_coords.append(new_atom, ignore_index=True)
    return d_coords


for pair in half_full_pair:
    d_coords = add_atoms(pair[0], pair[1], d_coords)
    


d_coords_sorted = d_coords.sort_values(by='new_resid')


d_coords_sorted['atom_number'] = d_coords_sorted.index + 1


d_coords_sorted_cleaned = d_coords_sorted.drop(columns=['new_resid','blank_1', 'blank_2',
                                               'blank_3', 'blank_4', 'segment_id', 'line_idx']).reset_index(drop=True).fillna('')


with open(file_out,'w') as out:
    for ind, row in d_coords_sorted_cleaned.iterrows():
#         print(row)
        print("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:>4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format(*row), file=out)
    print("END",file=out)





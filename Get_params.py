#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx
import pandas as pd
import biopandas.pdb as ppdb
import itertools as it
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt
# %matplotlib notebook
# # %config InlineBackend.figure_format='svg'


input_pdb = "../structures/chimera_2x2x2_new.full.pdb"


output_params = '../UFF/UFF_params.0806.txt'


uffprm_path = '/Users/ellenwang/Documents/openbabel/share/openbabel/2.3.2/UFF.prm'


atom_name_type_dic = {'H1': 'H_',
 'C1': 'C_2',
 'C2': 'C_R',
 'C3': 'C_R',
 'C4': 'C_R',
 'O1': 'O_2',
 'O2': 'O_2',
 'O3': 'O_3',
 'Mg1': 'Mg6'}


atom_charge_dic = {'H1': 0.1828,
 'C1': 0.7904,
 'C2': -0.2774,
 'C3': -0.2284,
 'C4': 0.4362,
 'O1': -0.6949,
 'O2': -0.8149,
 'O3': -0.8804,
 'Mg1': 1.4902}


replace_map = {
    'H_': 'Hm',
    'C_2': 'C2m',
    'C_R': 'CRm',
    'O_2': 'O2m',
    'O_3': 'O3m',
}


atomtypes = {
    'C_2': [ 6  ,   12.0107 ,     0.7904  ,   "A"  ,   3.431000e-01 , 4.393000e-01],
    'C_R': [ 6  ,    12.0107  ,   -0.2774  ,  "A"  ,   3.431000e-01 , 4.393000e-01],
    'O_2': [8   ,   15.9994   ,  -0.8149  ,   "A"  ,   3.118000e-01 , 2.510000e-01],
    'O_3': [8   ,   15.9994   ,  -0.8149  ,   "A"  ,   3.118000e-01 , 2.510000e-01],
    'H_': [ 1  ,     1.0079 ,     0.1828 ,    "A"  ,   2.571000e-01 , 1.841000e-01],
    'Mg6':[12  ,    24.3050 ,     1.4902 ,    "A"   ,  2.691000e-01,  4.644000e-01]
}


atomtypes


atom_parms_df = pd.DataFrame(columns=['atom', 'num', 'Du'])
uff_params_df = pd.DataFrame(columns=[ 'param','Atom', 'r1','theta0','x1','D1','zeta','Zi','Vi','Uj','Xi','Hard','Radius'])

with open(uffprm_path, 'r') as f:
    for i, line in enumerate(f):
        if line.startswith('atom'):
            temp = line.split()[:3]
            temp_df = pd.DataFrame([temp], columns=['atom', 'num', 'Du'])
            atom_parms_df = atom_parms_df.append(temp_df, ignore_index=True)
        if line.startswith('param'):
            temp = line.split()
            temp_df = pd.DataFrame([temp], columns=['param', 'Atom', 'r1','theta0','x1','D1','zeta','Zi','Vi','Uj','Xi','Hard','Radius'])
            uff_params_df = uff_params_df.append(temp_df, ignore_index=True)


# uff_params_df


def calc_bond_params(atom1, atom2, r0, uff_params_df):
    Zi = float(uff_params_df.loc[uff_params_df['Atom']==atom1]['Zi'].item())
    Zj = float(uff_params_df.loc[uff_params_df['Atom']==atom2]['Zi'].item())
    kij = 664.12*Zi*Zj/(r0**3)
    return kij*418.4


def find_bond_length(atom1, atom2, avg_bondtypes):
    key = tuple([atom1, atom2])
    if key in avg_bondtypes:
        return avg_bondtypes[key][0]
    elif key[::-1] in avg_bondtypes:
        return avg_bondtypes[key[::-1]][0]
    else:
        raise Exception('bond length not found in avg_bondtypes')


def calc_angle_params(atom1, atom2, atom3, th0, uff_params_df, rij, rjk):
    Zi = float(uff_params_df.loc[uff_params_df['Atom']==atom1]['Zi'].item())
    Zk = float(uff_params_df.loc[uff_params_df['Atom']==atom3]['Zi'].item())
    rik = np.sqrt(rij**2 + rjk**2 - 2*rij*rjk*np.cos(th0/180*np.pi))
    kijk = (664.12/(rij*rjk))*(Zi*Zk/(rik**5))*rij*rjk*(3*rij*rjk*(1-(np.cos(th0/180*np.pi))**2)-rik**2*np.cos(th0/180*np.pi))
    return kijk*4.814


def calc_dihedral_params(atom2, atom3, BO,  uff_params_df):
    Uj = float(uff_params_df.loc[uff_params_df['Atom']==atom2]['Uj'].item())
    Uk = float(uff_params_df.loc[uff_params_df['Atom']==atom3]['Uj'].item())
    V = 5*np.sqrt(Uj*Uk)*(1+4.18*np.log(BO))
    return V*4.184/2


# Read pdb file

pdbin = ppdb.PandasPdb()
pdbin.read_pdb(input_pdb)


d_coords = pdbin.df['ATOM']


d_coords['bondtype'] = ""
for i, n in d_coords.iterrows():
    d_coords.at[i,'bondtype'] = atom_name_type_dic[d_coords.iloc[i]['atom_name']]


coords = d_coords[['x_coord', 'y_coord', 'z_coord']].values


dist = np.sqrt(np.sum((coords[:,None,:]-coords[None,:,:])**2, axis=2))


# Generate network graphics for the connections in MOF

d_coords['atom_number'] = d_coords.index +1


G = nx.Graph()


for i in range(len(d_coords)):
    for j in range(i, len(d_coords)):
        if 'Mg1' in [d_coords.loc[i,'atom_name'], d_coords.loc[j,'atom_name']] and dist[i][j]<2.2 and dist[i][j]>0:
            G.add_edge(d_coords.loc[i].atom_number, d_coords.loc[j].atom_number)
        elif dist[i][j]<1.8 and dist[i][j]>0:
            G.add_edge(d_coords.loc[i].atom_number, d_coords.loc[j].atom_number)


bond_temp = []
for i in G.edges():
    bond_temp.append([i[0],i[1]])
bond_df = pd.DataFrame(bond_temp)


angle_temp = []
for n in G.nodes():
    l = list(it.combinations(G.neighbors(n),2))
    if l:
        for item in l:
            angle_temp.append([item[0], n, item[1]])
angle_df = pd.DataFrame(angle_temp)


dihedral_temp = []
for n1 in G.nodes():
    list_n2 = [x for x in G.neighbors(n1) if d_coords.loc[d_coords['atom_number'].values==x].bondtype.values!='Mg6']
#     print(list_1)
    for n2 in list_n2:
        list_n3 = [x for x in G.neighbors(n2) if d_coords.loc[d_coords['atom_number'].values==x].bondtype.values!='Mg6' and x != n1]
        for n3 in list_n3:
            list_n4 = [x for x in G.neighbors(n3) if x !=n1 and x!=n2]
            for n4 in list_n4:
                dih = [n1, n2, n3, n4]
                if dih[::-1] not in dihedral_temp:
                    dihedral_temp.append([n1, n2, n3, n4])
dihedral_df = pd.DataFrame(dihedral_temp)


def get_dihedral(A, B, C, D):
    r0 = A - B
    r1 = C - B
    r2 = D - C
    
    r1 /= np.linalg.norm(r1)
    v = r0 - np.dot(r0, r1)*r1
    w = r2 - np.dot(r2, r1)*r1
    
    x = np.dot(v, w)
    y = np.dot(np.cross(r1, v), w)
    
    return np.degrees(np.arctan2(y, x))


def get_coords(df, i):
    return np.array([float(df.iloc[i]['x_coord']), float(df.iloc[i]['y_coord']), float(df.iloc[i]['z_coord'])])


def add_dihedral_params(func, phase, kd, pn, df, i):
    df.at[i, 'func'] = func
    df.at[i, 'phase'] = phase
    df.at[i, 'kd'] = kd
    df.at[i, 'pn'] = pn


def add_improper(n1, n2, n3, n4, temp_list):
    temp_list.append([n1, n2, n3, n4])
    temp_list.append([n1, n3, n2, n4])
    temp_list.append([n1, n3, n4, n2])


def find_improper(at1, at2, at3, at4, G, coords_df, out_list):
    for i, n1 in coords_df.loc[coords_df['atom_name'] == at1].iterrows():
        list_n = [x for x in G.neighbors(n1['atom_number'])]
        for i in list_n:
            if coords_df.iloc[i-1]['atom_name'] == at2:
                n2 = i
            elif coords_df.iloc[i-1]['atom_name'] == at3:
                n3 = i
            elif coords_df.iloc[i-1]['atom_name'] == at4:
                n4 = i
        add_improper(n1['atom_number'], n2, n3, n4, out_list)


def get_angle(A, B, C):
    ba = np.array(A) - np.array(B)
    bc = np.array(C) - np.array(B)
    
    cosine_angle = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


improp_dihedral_temp = []    

find_improper('C2', 'C1', 'C3', 'C4', G, d_coords, improp_dihedral_temp)
find_improper('C1', 'C2', 'O1', 'O2', G, d_coords, improp_dihedral_temp)
find_improper('C4', 'O3', 'C2', 'C3', G, d_coords, improp_dihedral_temp)
find_improper('C3', 'C4', 'C2', 'H1', G, d_coords, improp_dihedral_temp)

improp_dihedral_df = pd.DataFrame(improp_dihedral_temp)


# Assign parameters for bond/angle/dihedral/improper dihedral

bondtypes = {}
for i, n in bond_df.iterrows():
#     print(i, n[0], n[1])
    atomA = d_coords.iloc[n[0]-1]['atom_name']
    atomB = d_coords.iloc[n[1]-1]['atom_name']
    key_pair = tuple([atom_name_type_dic[atomA], atom_name_type_dic[atomB]])
#     print(key_pair)
#     atom_a = d_coords.iloc[n[0]-1]['x_coord':'z_coord']
#     atom_b = d_coords.iloc[n[1]-1]['x_coord':'z_coord']
    distance = dist[n[0]-1, n[1]-1]
#     print(key_pair, distance)
    if key_pair in bondtypes:
        bondtypes[key_pair].append(distance)
    elif key_pair[::-1] in bondtypes:
        bondtypes[key_pair[::-1]].append(distance)
    else:
        bondtypes[key_pair] = [distance]


bondtypes.keys()


avg_bondtypes = {}
for key in bondtypes.keys():
#     fig = plt.figure()
    avg = np.average(bondtypes[key])
    kb = calc_bond_params(key[0], key[1], avg, uff_params_df)
    avg_bondtypes[key] = [avg, kb]
#     plt.hist(bondtypes[key], 10, label = '%5.4f' %avg)
#     print(key, avg)
#     plt.title(key)
#     plt.legend(loc='upper right')


avg_bondtypes


angletypes = {}
for i, n in angle_df.iterrows():
    key = tuple([d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype']])
    A_atom = d_coords.iloc[n[0]-1]['x_coord':'z_coord']
    B_atom = d_coords.iloc[n[1]-1]['x_coord':'z_coord']
    C_atom = d_coords.iloc[n[2]-1]['x_coord':'z_coord']
    angle_ijk = get_angle(A_atom, B_atom, C_atom)
    if key in angletypes:
        angletypes[key].append(angle_ijk)
    elif key[::-1] in angletypes:
        angletypes[key[::-1]].append(angle_ijk)
    else:
        angletypes[key] = [angle_ijk]


avg_angletypes = {}
for key in angletypes.keys():
#     fig = plt.figure()
#     plt.hist(angletypes[key], 18, label = key)
#     plt.title(key)
    rij = find_bond_length(key[0], key[1], avg_bondtypes)
    rjk = find_bond_length(key[1], key[2], avg_bondtypes)
    if key == ('O_2', 'Mg6', 'O_2'):
        temp_angles_large = [i for i in angletypes[key] if i > 150]
        temp_angles_small = [i for i in angletypes[key] if i < 150]
        avg_large = np.average(temp_angles_large)
        avg_small = np.average(temp_angles_small)
        kl = calc_angle_params(key[0], key[1], key[2], avg_large, uff_params_df, rij, rjk)
        ks = calc_angle_params(key[0], key[1], key[2], avg_small, uff_params_df, rij, rjk)
        avg_angletypes[key] = {'l': [avg_large, kl], 's':[avg_small, 2*ks]}
        
    elif key == ('O_2', 'Mg6', 'O_3'):
        avg = np.average(angletypes[key])
        k = calc_angle_params(key[0], key[1], key[2], avg, uff_params_df, rij, rjk)
        avg_angletypes[key] = [avg, k*2]
    else:
        avg = np.average(angletypes[key])
        k = calc_angle_params(key[0], key[1], key[2], avg, uff_params_df, rij, rjk)
        avg_angletypes[key] = [avg, k]


avg_angletypes


dihedraltypes = {}
for i in ['func', 'phase', 'kd', 'pn']:
    dihedral_df[i] = ''
for i, n in dihedral_df.iterrows():
    key = tuple([d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], 
                 d_coords.iloc[n[2]-1]['bondtype'], d_coords.iloc[n[3]-1]['bondtype']])
    A_atom = get_coords(d_coords, n[0]-1)
    B_atom = get_coords(d_coords, n[1]-1)
    C_atom = get_coords(d_coords, n[2]-1)
    D_atom = get_coords(d_coords, n[3]-1)
    dih = get_dihedral(np.array(A_atom), np.array(B_atom), np.array(C_atom), np.array(D_atom))
    if key in dihedraltypes:
        dihedraltypes[key].append(dih)
    elif key[::-1] in dihedraltypes:
        dihedraltypes[key[::-1]].append(dih)
    else:
        dihedraltypes[key] = [dih]
    


dihedraltypes.keys()


avg_dihedraltypes = {}
for key in dihedraltypes:
#     fig = plt.figure()
#     plt.hist(dihedraltypes[key], 45)
#     plt.title(key)
#     plt.xlim([-180, 180])
#     if key == ('C_R', 'C_R', 'O_3', 'Mg6'):
#         temp_dih_large = [i for i in dihedraltypes[key] if abs(np.cos(np.deg2rad(i))) > 0.5]
#         temp_dih_small = [i for i in dihedraltypes[key] if abs(np.cos(np.deg2rad(i))) < 0.5]
#         avg_large = np.average(np.absolute(np.cos(np.deg2rad(temp_dih_large))))
#         avg_small = np.average(np.absolute(np.cos(np.deg2rad(temp_dih_small))))
#         k = calc_dihedral_params(key[1], key[2], 1, uff_params_df)
# #         avg_dihedraltypes[key] = {'l': [avg_large, k], 's':[avg_small, k]}
#         avg = np.average(np.absolute(np.cos(np.deg2rad(dihedraltypes[key]))))
#         avg_dihedraltypes[key] = [avg, k, 3]
#     el
    if ((key[1], key[2])== ('C_R', 'O_3') or (key[2], key[1]) == ('C_R', 'O_3')
          or (key[1], key[2])== ('C_R', 'C_2') or (key[2], key[1]) == ('C_R', 'C_2')):
        avg = np.average(np.absolute(np.cos(np.deg2rad(dihedraltypes[key]))))
#         k = calc_dihedral_params(key[1], key[2], 1, uff_params_df)
        k = 1*4.184*2
        avg_dihedraltypes[key] = [avg, k, 6]
    else:
        avg = np.average(np.absolute(np.cos(np.deg2rad(dihedraltypes[key]))))
        k = calc_dihedral_params(key[1], key[2], 1.5, uff_params_df)
        avg_dihedraltypes[key] = [avg, k, 2]


avg_dihedraltypes


impropertypes = {}
for i in ['func', 'phase', 'kd']:
    improp_dihedral_df[i] = ''
for i, n in improp_dihedral_df.iterrows():
    key = tuple([d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype'], d_coords.iloc[n[3]-1]['bondtype']])
    A_atom = get_coords(d_coords, n[0]-1)
    B_atom = get_coords(d_coords, n[1]-1)
    C_atom = get_coords(d_coords, n[2]-1)
    D_atom = get_coords(d_coords, n[3]-1)
    dih = get_dihedral(np.array(A_atom), np.array(B_atom), np.array(C_atom), np.array(D_atom))
    if dih < 10:
        dih = 0.0
    else:
        raise Exception('improper dihedral angle too large: %5.3f' %dih)
    if key in impropertypes:
        impropertypes[key].append(dih)
#     elif key[::-1] in impropertypes:
#         impropertypes[key[::-1]].append(dih)
    else:
        impropertypes[key] = [dih]


impropertypes.keys()


avg_impropertypes = {}
for key in impropertypes:
    if 'O_2' in key:
        k = 50*4.184
        avg = np.average(impropertypes[key])
        avg_impropertypes[key] = [avg, k/3, 1]
    else:
        k = 6*4.184
        avg = np.average(impropertypes[key])
        avg_impropertypes[key] = [avg, k/3, 1]


avg_impropertypes


# Assemble parameter file

with open(output_params, 'w') as params:
    for k in atom_name_type_dic:
        at = atom_name_type_dic[k]
        print('atomtypes\t %s\t %s\t %d\t %7.4f\t %7.4f\t %c\t %6.4f\t %6.4f\t'
             %(k, at, atomtypes[at][0],atomtypes[at][1], atom_charge_dic[k], atomtypes[at][3], atomtypes[at][4],
              atomtypes[at][5]), file=params)
        
    for k in avg_bondtypes:
        print('bondtypes\t %s\t %s\t %5.3f %10.4f' 
              %(k[0], k[1], avg_bondtypes[k][0], avg_bondtypes[k][1]), file=params)
    
    for k in avg_angletypes:
        if type(avg_angletypes[k]) == dict:
            for k2 in avg_angletypes[k]:
                print('angletypes\t %s\t %s\t %s\t %5.3f\t %10.4f' 
                      %(k[0], k[1], k[2], avg_angletypes[k][k2][0], avg_angletypes[k][k2][1]), file=params)
        else:
            print('angletypes\t %s\t %s\t %s\t %5.3f\t %10.4f' 
                  %(k[0], k[1], k[2], avg_angletypes[k][0], avg_angletypes[k][1]), file=params)
            
    for k in avg_dihedraltypes:
        print('dihedraltypes\t %s\t %s\t %s\t %s\t %5.3f\t %10.4f\t %d\t' 
              %(k[0], k[1], k[2], k[3], avg_dihedraltypes[k][0], avg_dihedraltypes[k][1], avg_dihedraltypes[k][2]), file=params)
        
    for k in avg_impropertypes:
        print('improperdihedraltypes\t %s\t %s\t %s\t %s\t %5.3f\t %10.4f\t %d\t' 
              %(k[0], k[1], k[2], k[3], avg_impropertypes[k][0], avg_impropertypes[k][1], avg_impropertypes[k][2]), file=params)








#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx
import pandas as pd
import biopandas.pdb as ppdb
import itertools as it
import scipy.spatial.distance as sp
import argparse as argp


parser = argp.ArgumentParser(description='Generate itp and top file from pdb')
parser.add_argument('--pdbin', type=str, help="full path to input pdb file")
parser.add_argument('--out_path', type=str, help='path to the output itp and top files')
parser.add_argument('--itp', type=str, help='name of the output itp file')
parser.add_argument('--top', type=str, help='name of the output top file')
parser.add_argument('--params', type=str, help="full path to the parameter file")
args = parser.parse_args()
#args = parser.parse_args(['--pdbin', "../structures/mof74_unit_112_final.pdb",
#                         '--out_path', '../scripts',
#                         '--itp', 'mof74_unit_112_final.ellen.itp',
#                         '--top', 'mof74_unit_112_final.ellen.top', 
#                         '--params', '../UFF/UFF_params.0806.rename.txt'])


input_pdb = args.pdbin
output_itp_fn = args.itp
output_itp = "%s/%s" %(args.out_path, args.itp)
output_top = "%s/%s" %(args.out_path, args.top)
params = args.params


# special_bond = tuple(['UC2', 'UO2'])
special_angle = tuple(['O2m', 'Mg6', 'O2m'])
# special_dihedral = tuple(['CRm', 'CRm', 'O3m', 'Mg6'])


# Read parameter file

print('Reading parameter file...')


bond_func = 1
angle_func = 1
dihedral_func = 3 # Ryckaert- Bellemans dihedral
improp_dihedralfunc = 4 # periodic im- proper dihedral


params_df = pd.read_csv(params, sep='\s+', header = None)


atomtypes_params = params_df.loc[params_df[0] == 'atomtypes']
atomtypes_params.columns = ['atomtypes', 'name', 'bondtype','at.num', 'mass', 'charge', 'ptype', 'sigma', 'epsilon']

bondtypes_params = params_df.loc[params_df[0] == 'bondtypes'].loc[:,'0':'4']
bondtypes_params.columns = ['bondtypes', 'i', 'j','k0', 'kb']

angletypes_params = params_df.loc[params_df[0] == 'angletypes'].loc[:,'0':'5']
angletypes_params.columns = ['angletypes', 'i', 'j', 'k','th0', 'cth']

dihedraltypes_params = params_df.loc[params_df[0] == 'dihedraltypes'].loc[:,'0':'7']
dihedraltypes_params.columns = ['dihedratypes', 'i', 'j', 'k', 'l', 'cos(th0)', 'kd', 'pn']

impropdihedraltypes_params = params_df.loc[params_df[0] == 'improperdihedraltypes'].loc[:,'0':'7']
impropdihedraltypes_params.columns = ['improperdihedraltypes', 'i', 'j', 'k', 'l', 'th0',  'kd', 'pn']


atom_name_type_dic = {}
for i, n in atomtypes_params.iterrows():
    atom_name_type_dic[n['name']] = n['bondtype']


bondtype_dic = {}
single_bondtype_dic = {'func': '', 'kb':''}
for i, n in bondtypes_params.iterrows():
    single_bondtype_dic = {'func': bond_func, 'kb':n['kb']}
    if (n['i'], n['j']) in bondtype_dic.keys():
        bondtype_dic[(n['i'], n['j'])] = [bondtype_dic[(n['i'], n['j'])], single_bondtype_dic]
    else:
        bondtype_dic[(n['i'], n['j'])] = single_bondtype_dic


angletype_dic = {}
single_angletype_dic = {'func':'', 'cth':''}
for i, n in angletypes_params.iterrows():
    single_angletype_dic = {'func':angle_func, 'cth':n['cth']}
    if (n['i'], n['j'], n['k']) in angletype_dic.keys():
        angletype_dic[(n['i'], n['j'], n['k'])] = [angletype_dic[(n['i'], n['j'], n['k'])], single_angletype_dic]
    else:
        angletype_dic[(n['i'], n['j'], n['k'])] = single_angletype_dic


dihedraltype_dic = {}
single_dihedraltype_dic = {'func':'', 'kd':'', 'pn':''}
for i, n in dihedraltypes_params.iterrows():
    single_dihedraltype_dic = {'func':dihedral_func, 'kd':n['kd'], 'pn':n['pn']}
    if (n['i'], n['j'], n['k'], n['l']) in dihedraltype_dic.keys():
        dihedraltype_dic[(n['i'], n['j'], n['k'], n['l'])] = [dihedraltype_dic[(n['i'], n['j'], n['k'], n['l'])], single_dihedraltype_dic]
    else:
        dihedraltype_dic[(n['i'], n['j'], n['k'], n['l'])] = single_dihedraltype_dic


improp_dihedraltype_dic = {}
for i, n in impropdihedraltypes_params.iterrows():
    single_dihedraltype_dic = {'func':improp_dihedralfunc, 'kd':n['kd'], 'pn':n['pn']}
    if (n['i'], n['j'], n['k'], n['l']) in improp_dihedraltype_dic.keys():
        improp_dihedraltype_dic[(n['i'], n['j'], n['k'], n['l'])] = [improp_dihedraltype_dic[(n['i'], n['j'], n['k'], n['l'])], single_dihedraltype_dic]
    else:
        improp_dihedraltype_dic[(n['i'], n['j'], n['k'], n['l'])] = single_dihedraltype_dic


# Read pdb file

print('Reading pdb file...')


pdbin = ppdb.PandasPdb()
pdbin.read_pdb(input_pdb)


d_coords = pdbin.df['ATOM']


d_coords['bondtype'] = ""
d_coords['mass'] = ""
d_coords['charge'] = ""
for i, n in d_coords.iterrows():
    d_coords.at[i,'bondtype'] = atom_name_type_dic[d_coords.iloc[i]['atom_name']]
    d_coords.at[i,'mass'] = atomtypes_params.loc[atomtypes_params['bondtype'] == d_coords.iloc[i]['bondtype']]['mass'].values[0]
    d_coords.at[i,'charge'] = atomtypes_params.loc[atomtypes_params['bondtype'] == d_coords.iloc[i]['bondtype']]['charge'].values[0]


coords = d_coords[['x_coord', 'y_coord', 'z_coord']].values


dist = np.sqrt(np.sum((coords[:,None,:]-coords[None,:,:])**2, axis=2))


# Generate network graphics for the connections in MOF

print('Generating network graphics beased on MOF configureation...')


G = nx.Graph()


for i in range(len(d_coords)):
    for j in range(i, len(d_coords)):
        if ('Mg6' in [d_coords.loc[i,'bondtype'], d_coords.loc[j,'bondtype']] and dist[i][j]<2.2 and dist[i][j]>0):
            G.add_edge(d_coords.loc[i].atom_number, d_coords.loc[j].atom_number)
        elif dist[i][j]<1.8 and dist[i][j]>0:
            G.add_edge(d_coords.loc[i].atom_number, d_coords.loc[j].atom_number)


bond_temp = []
for i in G.edges():
    bond_temp.append([i[0],i[1]])
bond_df = pd.DataFrame(bond_temp)


angle_temp = []
for n in G.nodes():
#     for neigh in  G.neighbors(n):
#         print(neigh)
    l = list(it.combinations(G.neighbors(n),2))
    if l:
        for item in l:
            angle_temp.append([item[0], n, item[1]])
#             print(item[0], n, item[1])
#             G.add_path([l[0], n, l[1]])
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
                    dihedral_temp.append(dih)
                    
dihedral_df = pd.DataFrame(dihedral_temp)


pairs_temp = []
for n1 in G.nodes():
    list_n2 = [x for x in G.neighbors(n1)]
    for n2 in list_n2:
        list_n3 = [x for x in G.neighbors(n2) if x != n1]
        for n3 in list_n3:
            list_n4 = [x for x in G.neighbors(n3) if x !=n1 and x!=n2]
            for n4 in list_n4:
                pr = [n1, n4]
                if (pr[::-1] not in pairs_temp and pr not in bond_temp and 
                    pr not in pairs_temp and pr[::-1] not in bond_temp):
                    pairs_temp.append(pr)
pairs_df = pd.DataFrame(pairs_temp)


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


improp_dihedral_temp = []    

find_improper('C2', 'C1', 'C3', 'C4', G, d_coords, improp_dihedral_temp)
find_improper('C1', 'C2', 'O1', 'O2', G, d_coords, improp_dihedral_temp)
find_improper('C4', 'O3', 'C2', 'C3', G, d_coords, improp_dihedral_temp)
find_improper('C3', 'C4', 'C2', 'H1', G, d_coords, improp_dihedral_temp)
improp_dihedral_df = pd.DataFrame(improp_dihedral_temp)


def get_angle(A, B, C):
    ba = np.array(A) - np.array(B)
    bc = np.array(C) - np.array(B)
    
    cosine_angle = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


# Assign parameters for bond/angle/dihedral/improper dihedral

print('Assigning paraters for bond/angle/dihedral/impropers...')


def add_bond_params(functype, distance, kb, bond_df, i):
    bond_df.at[i, 'func'] = functype
    bond_df.at[i, 'b0'] = distance
    bond_df.at[i, 'kb'] = kb


for i in ['func', 'b0', 'kb']:
    bond_df[i] = ''
for i, n in bond_df.iterrows():
#     print(i, n[0], n[1])
    atomA = d_coords.iloc[n[0]-1]['atom_name']
    atomB = d_coords.iloc[n[1]-1]['atom_name']
    key_pair = tuple([atom_name_type_dic[atomA], atom_name_type_dic[atomB]])
    distance = dist[n[0]-1, n[1]-1]
    if key_pair in bondtype_dic.keys():
        add_bond_params(bond_func, distance, bondtype_dic[key_pair]['kb'], bond_df, i)
    elif key_pair[::-1] in bondtype_dic.keys():
        add_bond_params(bond_func, distance, bondtype_dic[key_pair[::-1]]['kb'], bond_df, i)
    else:
        raise Exception('bondtype %s-%s not fond in parameter file' %(d_coords.iloc[n[0]]['bondtype'], d_coords.iloc[n[1]]['bondtype']))


def add_angle_params(func, th0, cth, angle_df, i):
    angle_df.at[i, 'func'] = func
    angle_df.at[i, 'th0'] = th0
    angle_df.at[i, 'cth'] = cth


for i in ['func', 'th0', 'cth']:
    angle_df[i] = ''

for i, n in angle_df.iterrows():
    key = tuple([d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype']])
    A_atom = d_coords.iloc[n[0]-1]['x_coord':'z_coord']
    B_atom = d_coords.iloc[n[1]-1]['x_coord':'z_coord']
    C_atom = d_coords.iloc[n[2]-1]['x_coord':'z_coord']
    angle_ijk = get_angle(A_atom, B_atom, C_atom)
    if key == special_angle or key[::-1] == special_angle:
        if angle_ijk < 145:
            cth = max([float(angletype_dic[special_angle][0]['cth']),float(angletype_dic[special_angle][1]['cth']) ])
        else:
            cth = min([float(angletype_dic[special_angle][0]['cth']),float(angletype_dic[special_angle][1]['cth']) ])
        add_angle_params(angle_func, angle_ijk, cth, angle_df, i)
    elif key in angletype_dic.keys():
        add_angle_params(angle_func, angle_ijk, angletype_dic[key]['cth'], angle_df, i)
    elif key[::-1] in angletype_dic.keys():
        add_angle_params(angle_func, angle_ijk, angletype_dic[key[::-1]]['cth'], angle_df, i)
    else:
        raise Exception('angletype %s-%s-%s not fond in parameter file' 
                        %(d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype']))
        


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


def add_dihedral_params(func, C,  df, i):
    df.at[i, 'C0'] = C[0]
    df.at[i, 'C1'] = C[1]
    df.at[i, 'C2'] = C[2]
    df.at[i, 'C3'] = C[3]
    df.at[i, 'C4'] = C[4]
    df.at[i, 'C5'] = C[5]
    df.at[i, 'func'] = func


def calc_RB_params(kd, th0, pn):
    if pn == 2:
        C0 = 0.5*kd*(1+np.cos(np.deg2rad(2*th0)))
        C2 = -kd*np.cos(np.deg2rad(2*th0))
        C1 = C3 = C4 = C5 = 0 
    elif pn == 3:
        C0 = 0.5*kd
        C3 = -0.125*kd*np.cos(np.deg2rad(3*th0))
        C1 = 3 * C3
        C2 = C4 = C5 = 0
    return [C0, C1, C2, C3, C4, C5]


for i in ['func', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5']:
    dihedral_df[i] = ''
for i, n in dihedral_df.iterrows():
    key = tuple([d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype'], d_coords.iloc[n[3]-1]['bondtype']])
    A_atom = get_coords(d_coords, n[0]-1)
    B_atom = get_coords(d_coords, n[1]-1)
    C_atom = get_coords(d_coords, n[2]-1)
    D_atom = get_coords(d_coords, n[3]-1)
    dih = get_dihedral(np.array(A_atom), np.array(B_atom), np.array(C_atom), np.array(D_atom))
#     if key == special_dihedral or key[::-1] == special_dihedral:
#         if abs(np.sin(dih*np.pi/180)) < 0.75:
#             kd = max([float(dihedraltype_dic[special_dihedral][0]['kd']), float(dihedraltype_dic[special_dihedral][1]['kd'])])
#         else:
#             kd = min([float(dihedraltype_dic[special_dihedral][0]['kd']), float(dihedraltype_dic[special_dihedral][1]['kd'])])
#         add_dihedral_params(dihedral_func, dih, kd, dihedraltype_dic[special_dihedral][0]['pn'], dihedral_df, i)
#     el
    if key in dihedraltype_dic.keys():
        C = calc_RB_params(float(dihedraltype_dic[key]['kd']), dih, dihedraltype_dic[key]['pn'])
        add_dihedral_params(dihedral_func,C, dihedral_df, i)
    elif key[::-1] in dihedraltype_dic.keys():
        C = calc_RB_params(float(dihedraltype_dic[key[::-1]]['kd']), dih, dihedraltype_dic[key[::-1]]['pn'])
        add_dihedral_params(dihedral_func, C, dihedral_df, i)
    else:
        raise Exception('dihedraltype %s-%s-%s-%s not fond in parameter file' 
                        %(d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype'], d_coords.iloc[n[3]-1]['bondtype']))       


def add_improperdihedral_params(func, phase, kd, pn, df, i):
    df.at[i, 'func'] = func
    df.at[i, 'phase'] = phase
    df.at[i, 'kd'] = kd
    df.at[i, 'pn'] = pn


for i in ['func', 'phase', 'kd', 'pn']:
    improp_dihedral_df[i] = ''
for i, n in improp_dihedral_df.iterrows():
    key = tuple([d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype'], d_coords.iloc[n[3]-1]['bondtype']])
    A_atom = get_coords(d_coords, n[0]-1)
    B_atom = get_coords(d_coords, n[1]-1)
    C_atom = get_coords(d_coords, n[2]-1)
    D_atom = get_coords(d_coords, n[3]-1)
    dih = get_dihedral(np.array(A_atom), np.array(B_atom), np.array(C_atom), np.array(D_atom))
    if dih < 10:
        dih = 180.0
    else:
        raise Exception('improper dihedral angle too large: %5.3f' %dih)
        
    if key in improp_dihedraltype_dic.keys():
        add_improperdihedral_params(improp_dihedralfunc, dih, improp_dihedraltype_dic[key]['kd'], improp_dihedraltype_dic[key]['pn'], improp_dihedral_df, i)
    elif key[::-1] in improp_dihedraltype_dic.keys():
        add_improperdihedral_params(improp_dihedralfunc, dih, improp_dihedraltype_dic[key[::-1]]['kd'], improp_dihedraltype_dic[key[::-1]]['pn'], improp_dihedral_df, i)
    else:
        raise Exception('dihedraltype %s-%s-%s-%s not fond in parameter file' 
                        %(d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype'], d_coords.iloc[n[3]-1]['bondtype']))       


# Assemble top file

print('Assembling top/itp files...')


with open(output_itp, 'w') as itp:
    print('[ moleculetype ]\n; Name       nrexcl\nMOF    3\n', file = itp)


    print('\n[ atoms ]\n; nr type  resnr    residue    atom     cgnr    charge       mass', file = itp)
    for i, n in d_coords.iterrows():
        print('{:>7d}{:>11s}{:>7d}{:>7s}{:>7s}{:>7d}{:>11.4f}{:>11.4f}'.format(
            int(n['atom_number']), n['bondtype'], int(n['residue_number']), n['residue_name'], n['atom_name'],
            int(n['atom_number']), float(n['charge']), float(n['mass'])), file = itp)

    print('\n[ bonds ]\n; i  j  func  b0  kb', file=itp)
    for i, n in bond_df.iterrows():
        print('{:>7d}{:>7d}{:>7d}{:>11.4f}  {:>11.4f}'.format(
            int(n[0]), int(n[1]), int(n['func']), float(n['b0']*0.1), float(n['kb'])), file=itp)

    print('\n[ angles ]\n; i  j  k  func  th0  cth', file = itp)
    for i, n in angle_df.iterrows():
        print('{:>7d}{:>7d}{:>7d}{:>7d}{:>11.1f}  {:>11.4f}'.format(
            int(n[0]), int(n[1]), int(n[2]), int(n['func']), float(n['th0']), float(n['cth'])), file = itp)

    print('\n[ dihedrals ]\n; i  j  k  l  C0  C1  C2  C3  C4  C5', file = itp)
    for i, n in dihedral_df.iterrows():
        print('{:>7d}{:>7d}{:>7d}{:>7d}{:>7d}{:>11.4f}  {:>11.4f} {:>11.4f}  {:>11.4f} {:>11.4f}  {:>11.4f}'.format(
            int(n[0]), int(n[1]), int(n[2]), int(n[3]), int(n['func']), float(n['C0']), float(n['C1']), float(n['C2']), 
        float(n['C3']), float(n['C4']), float(n['C5'])), file = itp)

    print('\n[ dihedrals ]\n; improper dihedrals\n; i  j  k  l  func  phase  kd', file = itp)
    for i, n in improp_dihedral_df.iterrows():
        print('{:>7d}{:>7d}{:>7d}{:>7d}{:>7d}{:>11.4f}  {:>11.4f}{:>7d}'.format(
            int(n[0]), int(n[1]), int(n[2]), int(n[3]), int(n['func']), float(n['phase']), float(n['kd']), int(n['pn'])), file = itp)
        
    print('\n[ pairs ]\n', file=itp)
    for i, n in pairs_df.iterrows():
        print('{:>7d}{:>7d}'.format(int(n[0]), int(n[1])), file = itp)


at = atomtypes_params.drop(columns=['name'])


at_uniq = at.drop_duplicates()


with open(output_top, 'w') as top:
    print('; Include forcefield parameters\n#include "oplsaa.ff/forcefield.itp" \n', file=top)
    
    print('[ atomtypes ]\n; bond_type    mass    charge   ptype          sigma      epsilon', file = top)
    for i, n in at_uniq.iterrows():
        print('{:9s}{:>3d}{:>13.4f}{:>12.4f}     {:2s}    {:11e}  {:11e}'.format(
            n['bondtype'],int(n['at.num']),float(n['mass']),float(n['charge']),n['ptype'], float(n['sigma']), float(n['epsilon'])), file=top)
    
    print('\n; Include kubisiak forcefield parameters for TFSI\n#include "kubisiak_ffnonbonded.itp"\n#include "kubisiak_ffbond.itp"\n', file=top)
    
    print('; Include anion topology\n#include "TFSI_KLU4_from_ATB.itp"\n', file=top)
    
    print('; Include parameter for Mg cations\n#include "oplsaa.ff/ions.itp" \n', file=top)
    
    print('; Include solvent topology\n#include "PC.itp"\n', file=top)
    
    print('; Include MOF topology\n#include "%s"\n' %output_itp_fn, file=top)


print('Done.')





import numpy as np
import networkx as nx
import pandas as pd
import biopandas.pdb as ppdb
import itertools as it
import scipy.spatial.distance as sp


input_pdb = "../structures/mof74_unit_112_final.pdb"
output_top = "mof74_unit_112_final.itp"


params = '../UFF/UFFparams.newname.txt'


# special_bond = tuple(['UC2', 'UO2'])
special_angle = tuple(['UO2', 'Mg6', 'UO2'])
special_dihedral = tuple(['UCR', 'UCR', 'UO2', 'Mg6'])


# Read parameter file

bond_func = 1
angle_func = 2
dihedral_func = 1
improp_dihedralfunc = 2

params_df = pd.read_csv(params, sep='\s+', header = None)

atomtypes_params = params_df.loc[params_df[0] == 'atomtypes']
atomtypes_params.columns = ['atomtypes', 'name', 'bondtype','at.num', 'mass', 'charge', 'ptype', 'sigma', 'epsilon']

bondtypes_params = params_df.loc[params_df[0] == 'bondtypes'].loc[:,'0':'3']
bondtypes_params.columns = ['bondtypes', 'i', 'j', 'kb']

angletypes_params = params_df.loc[params_df[0] == 'angletypes'].loc[:,'0':'4']
angletypes_params.columns = ['angletypes', 'i', 'j', 'k', 'cth']

dihedraltypes_params = params_df.loc[params_df[0] == 'dihedraltypes'].loc[:,'0':'6']
dihedraltypes_params.columns = ['dihedratypes', 'i', 'j', 'k', 'l',  'kd', 'pn']

impropdihedraltypes_params = params_df.loc[params_df[0] == 'improperdihedraltypes'].loc[:,'0':'5']
impropdihedraltypes_params.columns = ['improperdihedraltypes', 'i', 'j', 'k', 'l',  'kd']



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
    single_dihedraltype_dic = {'func':improp_dihedralfunc, 'kd':n['kd']}
    if (n['i'], n['j'], n['k'], n['l']) in improp_dihedraltype_dic.keys():
        improp_dihedraltype_dic[(n['i'], n['j'], n['k'], n['l'])] = [improp_dihedraltype_dic[(n['i'], n['j'], n['k'], n['l'])], single_dihedraltype_dic]
    else:
        improp_dihedraltype_dic[(n['i'], n['j'], n['k'], n['l'])] = single_dihedraltype_dic


# Read pdb file

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

G = nx.Graph()


for i in range(len(d_coords)):
    for j in range(i, len(d_coords)):
        if 'Mg' in [d_coords.loc[i,'element_symbol'], d_coords.loc[j,'element_symbol']] and dist[i][j]<2.2 and dist[i][j]>0:
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
pairs_temp = []
for n1 in G.nodes():
    list_n2 = [x for x in G.neighbors(n1) if d_coords.loc[d_coords['atom_number'].values==x].element_symbol.values!='Mg']
    for n2 in list_n2:
        list_n3 = [x for x in G.neighbors(n2) if d_coords.loc[d_coords['atom_number'].values==x].element_symbol.values!='Mg' and x != n1]
        for n3 in list_n3:
            list_n4 = [x for x in G.neighbors(n3) if x !=n1 and x!=n2]
            for n4 in list_n4:
                dih = [n1, n2, n3, n4]
                if dih.reverse() not in dihedral_temp:
                    dihedral_temp.append([n1, n2, n3, n4])
                    pairs_temp.append([n1, n4])
dihedral_df = pd.DataFrame(dihedral_temp)
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


def add_bond_params(functype, distance, kb, bond_df, i):
    bond_df.at[i, 'func'] = functype
    bond_df.at[i, 'b0'] = distance
    bond_df.at[i, 'kb'] = kb



for i in ['func', 'b0', 'kb']:
    bond_df[i] = ''
for i, n in bond_df.iterrows():
    atomA = d_coords.iloc[n[0]-1]['atom_name']
    atomB = d_coords.iloc[n[1]-1]['atom_name']
    key_pair = tuple([atom_name_type_dic[atomA], atom_name_type_dic[atomB]])
    distance = dist[n[0]-1, n[1]-1]
#     if key_pair == special_bond or key_pair[::-1] == special_bond:
#             if 'O1' in [atomA, atomB]:
#                 kb = max([float(bondtype_dic[special_bond][0]['kb']),float(bondtype_dic[special_bond][1]['kb']) ])
#             else:
#                 kb = min([float(bondtype_dic[special_bond][0]['kb']),float(bondtype_dic[special_bond][1]['kb']) ])
#             add_bond_params(bond_func, distance, kb, bond_df, i)
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
        raise Exception('angletype %s-%s-%s not fond in parameter file' %(d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype']))
        


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



for i in ['func', 'phase', 'kd', 'pn']:
    dihedral_df[i] = ''
for i, n in dihedral_df.iterrows():
    key = tuple([d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype'], d_coords.iloc[n[3]-1]['bondtype']])
    A_atom = get_coords(d_coords, n[0]-1)
    B_atom = get_coords(d_coords, n[1]-1)
    C_atom = get_coords(d_coords, n[2]-1)
    D_atom = get_coords(d_coords, n[3]-1)
    dih = get_dihedral(np.array(A_atom), np.array(B_atom), np.array(C_atom), np.array(D_atom))
    if key == special_dihedral or key[::-1] == special_dihedral:
        if abs(np.sin(dih*np.pi/180)) < 0.75:
            kd = max([float(dihedraltype_dic[special_dihedral][0]['kd']), float(dihedraltype_dic[special_dihedral][1]['kd'])])
        else:
            kd = min([float(dihedraltype_dic[special_dihedral][0]['kd']), float(dihedraltype_dic[special_dihedral][1]['kd'])])
        add_dihedral_params(dihedral_func, dih, kd, dihedraltype_dic[special_dihedral][0]['pn'], dihedral_df, i)
    elif key in dihedraltype_dic.keys():
        add_dihedral_params(dihedral_func, dih, dihedraltype_dic[key]['kd'], dihedraltype_dic[key]['pn'], dihedral_df, i)
    elif key[::-1] in dihedraltype_dic.keys():
        add_dihedral_params(dihedral_func, dih, dihedraltype_dic[key[::-1]]['kd'], dihedraltype_dic[key[::-1]]['pn'], dihedral_df, i)
    else:
        raise Exception('dihedraltype %s-%s-%s-%s not fond in parameter file' %(d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype'], d_coords.iloc[n[3]-1]['bondtype']))       


def add_improperdihedral_params(func, phase, kd, df, i):
    df.at[i, 'func'] = func
    df.at[i, 'phase'] = phase
    df.at[i, 'kd'] = kd


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
    if key in improp_dihedraltype_dic.keys():
        add_improperdihedral_params(improp_dihedralfunc, dih, improp_dihedraltype_dic[key]['kd'], improp_dihedral_df, i)
    else:
        raise Exception('dihedraltype %s-%s-%s-%s not fond in parameter file' %(d_coords.iloc[n[0]-1]['bondtype'], d_coords.iloc[n[1]-1]['bondtype'], d_coords.iloc[n[2]-1]['bondtype'], d_coords.iloc[n[3]-1]['bondtype']))       


# Assemble top file


with open(output_top, 'w') as top:
    print('[ moleculetype ]\n; Name       nrexcl\nMOF    3\n', file = top)

#     print('[ atomtypes ]\n; name  bond_type    mass    charge   ptype          sigma      epsilon', file = top)
#     for i, n in atomtypes_params.iterrows():
#         print('{:4s} {:9s}{:>3d}{:>13.4f}{:>12.4f}     {:2s}    {:11e}  {:11e}'.format(
#             n['name'], n['bondtype'],int(n['at.num']),float(n['mass']),float(n['charge']),n['ptype'], float(n['sigma']), float(n['epsilon'])), file=top)

    print('\n[ atoms ]\n; nr type  resnr    residue    atom     cgnr    charge       mass', file = top)
    for i, n in d_coords.iterrows():
        print('{:>7d}{:>11s}{:>7d}{:>7s}{:>7s}{:>7d}{:>11.4f}{:>11.4f}'.format(
            int(n['atom_number']), n['bondtype'], int(n['residue_number']), n['residue_name'], n['atom_name'],
            int(n['atom_number']), float(n['charge']), float(n['mass'])), file = top)

    print('\n[ bonds ]\n; i  j  func  b0  kb', file=top)
    for i, n in bond_df.iterrows():
        print('{:>7d}{:>7d}{:>7d}{:>11.4f}  {:>11.4f}'.format(
            int(n[0]), int(n[1]), int(n['func']), float(n['b0']*0.1), float(n['kb'])), file=top)

    print('\n[ angles ]\n; i  j  k  func  th0  cth', file = top)
    for i, n in angle_df.iterrows():
        print('{:>7d}{:>7d}{:>7d}{:>7d}{:>11.1f}  {:>11.4f}'.format(
            int(n[0]), int(n[1]), int(n[2]), int(n['func']), float(n['th0']), float(n['cth'])), file = top)

    print('\n[ dihedrals ]\n; i  j  k  l  func  phase  kd  pn', file = top)
    for i, n in dihedral_df.iterrows():
        print('{:>7d}{:>7d}{:>7d}{:>7d}{:>7d}{:>11.4f}  {:>11.4f}{:>7d}'.format(
            int(n[0]), int(n[1]), int(n[2]), int(n[3]), int(n['func']), float(n['phase']), float(n['kd']), int(n['pn'])), file = top)

    print('\n[ dihedrals ]\n; improper dihedrals\n; i  j  k  l  func  phase  kd', file = top)
    for i, n in improp_dihedral_df.iterrows():
        print('{:>7d}{:>7d}{:>7d}{:>7d}{:>7d}{:>11.4f}  {:>11.4f}'.format(
            int(n[0]), int(n[1]), int(n[2]), int(n[3]), int(n['func']), float(n['phase']), float(n['kd'])), file = top)
        
    print('\n[ pairs ]\n', file=top)
    for i, n in pairs_df.iterrows():
        print('{:>7d}{:>7d}'.format(int(n[0]), int(n[1])), file = top)



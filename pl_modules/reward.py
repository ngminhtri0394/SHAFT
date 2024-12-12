from metrics.eval_metrics import *
from common.bonds_dictionary import bonds_dictionary
import time

def bond_distance(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)


def get_bond_data(structure: Structure):
    bond_data = {}
    for c1 in structure:
        c = structure.get_neighbors(site=c1, r=4)
        for c2 in c:
            Rij = bond_distance(c1.coords, c2.coords)

            if c1.specie.number < c2.specie.number:
                k = str(c1.specie.number)+'_'+str(c2.specie.number)
            else:
                k = str(c2.specie.number)+'_'+str(c1.specie.number)
            if k in bond_data.keys():
                available = False
                for bb in bond_data[k]:
                    if abs(bb-Rij) < 1e-5:
                        available = True
                        break
                if not available:
                    bond_data[k] += [Rij]
            else:
                bond_data[k] = [Rij]
    return bond_data


def get_bs_from_bond_data(m: Structure, vpen_min=0.01,vpen_max=0.1,vpen_minmax=0.001):
    bond_avg_atom = bonds_dictionary['mean']
    bond_min_atom = bonds_dictionary['min']
    
    bond_data = get_bond_data(m)
    if len(bond_data.keys())==0:
        return 5.0
    bs = 0
    for bk in bond_data.keys():
        bondlist = np.array(bond_data[bk])
        bond_min_dif = np.exp(-bondlist + bond_min_atom[bk])
        bond_dif = bondlist - bond_avg_atom[bk] 
        bond_dist_dif = np.mean(np.abs(bond_dif)+ bond_min_dif**2)
        bs += bond_dist_dif
    return bs/len(bond_data.keys())

def get_valid_score(structure):
    atom_types = [s.specie.Z for s in structure]
    elems, comps = get_composition(atom_types)
    return max(float(smact_validity(elems, comps)),0.0)

def reward_pref_bond_dict(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001,reward_min=1e-5):
    pred = proxy(states)
    forme_norm = 10
    forme_score = (-pred)/forme_norm
    forme_score = np.exp(forme_score)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 1
    start =  time.time()
    bond_score = np.array([get_bs_from_bond_data(state.structure) for state in states])
    print('bs run time: ', time.time()-start)
    bond_score = np.exp(-bond_score)
    density_score = np.array([alpha_density*np.exp(-(m.structure.density-mean_density)**2/(2*(std_density**2))) for m in states])  
    valid_score = np.array([get_valid_score(state.structure) for state in states])
    wes = 0.2
    wbs = 0.5
    wds = 0.2
    wvs = 0.1
    reward = wes*forme_score+ wds*density_score + wbs*bond_score + wvs*valid_score
    reward = np.clip(reward,a_min=reward_min,a_max=None)
    return reward, forme_score, bond_score, valid_score, density_score 


def reward_pref_bond_dict_no_forme(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001,reward_min=1e-5):
    # pred = proxy(states)
    # forme_norm = 10
    # forme_score = (-pred)/forme_norm
    # forme_score = np.exp(forme_score)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 1
    bond_score = np.array([get_bs_from_bond_data(state.structure) for state in states])
    bond_score = np.exp(-bond_score)
    density_score = np.array([alpha_density*np.exp(-(m.structure.density-mean_density)**2/(2*(std_density**2))) for m in states])  
    valid_score = np.array([get_valid_score(state.structure) for state in states])
    wes = 0.2
    wbs = 0.5
    wds = 0.2
    wvs = 0.1
    reward = wds*density_score + wbs*bond_score + wvs*valid_score
    reward = np.clip(reward,a_min=reward_min,a_max=None)
    return reward, 0, bond_score, valid_score, density_score 


def reward_pref_bond_dict_no_bond(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001,reward_min=1e-5):
    pred = proxy(states)
    forme_norm = 10
    forme_score = (-pred)/forme_norm
    forme_score = np.exp(forme_score)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 1
    # bond_score = np.array([get_bs_from_bond_data(state.structure) for state in states])
    # bond_score = np.exp(-bond_score)
    density_score = np.array([alpha_density*np.exp(-(m.structure.density-mean_density)**2/(2*(std_density**2))) for m in states])  
    valid_score = np.array([get_valid_score(state.structure) for state in states])
    wes = 0.2
    wbs = 0.5
    wds = 0.2
    wvs = 0.1
    reward = wes*forme_score+ wds*density_score + wvs*valid_score
    reward = np.clip(reward,a_min=reward_min,a_max=None)
    return reward, forme_score, 0, valid_score, density_score 


def reward_pref_bond_dict_no_density(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001,reward_min=1e-5):
    pred = proxy(states)
    forme_norm = 10
    forme_score = (-pred)/forme_norm
    forme_score = np.exp(forme_score)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 1
    bond_score = np.array([get_bs_from_bond_data(state.structure) for state in states])
    bond_score = np.exp(-bond_score)
    valid_score = np.array([get_valid_score(state.structure) for state in states])
    wes = 0.2
    wbs = 0.5
    wds = 0.2
    wvs = 0.1
    reward = wes*forme_score + wbs*bond_score + wvs*valid_score
    reward = np.clip(reward,a_min=reward_min,a_max=None)
    return reward, forme_score, bond_score, valid_score, 0 


def reward_pref_bond_dict_no_valid(states, proxy, vpen_min=0.01, vpen_max=0.1, vpen_minmax=0.001,reward_min=1e-5):
    pred = proxy(states)
    forme_norm = 10
    forme_score = (-pred)/forme_norm
    forme_score = np.exp(forme_score)
    mean_density = 3.0
    std_density = 1.0
    alpha_density = 1
    bond_score = np.array([get_bs_from_bond_data(state.structure) for state in states])
    bond_score = np.exp(-bond_score)
    density_score = np.array([alpha_density*np.exp(-(m.structure.density-mean_density)**2/(2*(std_density**2))) for m in states])  
    wes = 0.2
    wbs = 0.5
    wds = 0.2
    wvs = 0.1
    reward = wes*forme_score+ wds*density_score + wbs*bond_score 
    reward = np.clip(reward,a_min=reward_min,a_max=None)
    return reward, forme_score, bond_score, 0, density_score 

reward_functions_dict = {'reward_pref_bond_dict':reward_pref_bond_dict,
                         'reward_pref_bond_dict_no_forme':reward_pref_bond_dict_no_forme,
                         'reward_pref_bond_dict_no_bond':reward_pref_bond_dict_no_bond,
                         'reward_pref_bond_dict_no_density':reward_pref_bond_dict_no_density,
                         'reward_pref_bond_dict_no_valid':reward_pref_bond_dict_no_valid}

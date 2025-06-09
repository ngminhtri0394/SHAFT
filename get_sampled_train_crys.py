import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import multiprocessing as mp
import argparse

def parallel_loading(idx):
    sampled_mols = []
    sampled_bss = []
    sampled_dss = []
    sampled_ess = []
    sampled_vss = []
    sampled_rewards = []
    if not os.path.isfile(path + 'saved_data/'+str(idx)+'_sampled_mols.pkl.gz'):
        return None
    try:
        a = pickle.load(gzip.open(path + 'saved_data/'+str(idx)+'_sampled_mols.pkl.gz'))
    except:
        print('Load failed: ',path + 'saved_data/'+str(idx)+'_sampled_mols.pkl.gz')
        return None
    print('Loaded file: ', path + 'saved_data/'+str(idx)+'_sampled_mols.pkl.gz')
    for (sampled_mol, sampled_reward, sampled_bs, sampled_ds, sampled_es, sampled_vs,_) in a:
        sampled_mols.extend(sampled_mol)
        sampled_bss.extend(sampled_bs)
        sampled_dss.extend(sampled_ds)
        sampled_ess.extend(sampled_es)
        sampled_vss.extend(sampled_vs)
        sampled_rewards.extend(sampled_reward)
    return (sampled_mols, sampled_bss, sampled_dss, sampled_ess, sampled_vss, sampled_rewards)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process sampling parameters.')
parser.add_argument('--k', type=int, default=100, help='Number of top-k structures to save')
parser.add_argument('--path', type=str, required=True, help='Path to the data directory')
parser.add_argument('--last_idx', type=int, default=45000, help='Last index to process')
parser.add_argument('--traj_length', type=int, default=3, help='Minimum trajectory length')
args = parser.parse_args()

k = args.k
path = args.path
last_idx = args.last_idx
traj_length = args.traj_length

sampled_mols = []
sampled_rewards = []
sampled_bss = []
sampled_dss = []
sampled_ess = []
sampled_vss = []



method_name = path.split('/')[3]

idx = 0
for idx in tqdm(range(0, last_idx, 100)):
    try:
        a = pickle.load(gzip.open(path + 'saved_data/'+str(idx)+'_sampled_mols.pkl.gz'))
    except:
        print('Load failed: ',path + 'saved_data/'+str(idx)+'_sampled_mols.pkl.gz')
        continue
    for (sampled_mol, sampled_reward, sampled_bs, sampled_ds, sampled_es, sampled_vs,_) in a:
        sampled_mols.extend(sampled_mol)
        sampled_bss.extend(sampled_bs)
        sampled_dss.extend(sampled_ds)
        sampled_ess.extend(sampled_es)
        sampled_vss.extend(sampled_vs)
        sampled_rewards.extend(sampled_reward)


print('Total number of mols: ', len(sampled_mols))
sampled_mols = np.array(sampled_mols)
sampled_rewards = np.array(sampled_rewards)
sampled_bss = np.array(sampled_bss)
sampled_dss = np.array(sampled_dss)
sampled_ess = np.array(sampled_ess)
sampled_vss = np.array(sampled_vss)
sort = np.argsort(sampled_rewards)[::-1]

if not os.path.isdir(path+'/top_'+str(k)+'_full/'):
    os.makedirs(path+'/top_'+str(k)+'_full/')


i = -1
j = 0
while j < k:
    i+=1
    if len(sampled_mols[sort[i]].atomic_numbers) < traj_length:
        continue
    structure = sampled_mols[sort[i]].structure
    atom = np.array([s.species for s in structure])

    print('Top ', j, ' reward ', sampled_rewards[sort[i]],
           ' bond score ', sampled_bss[sort[i]],
           ' formation energy score ', sampled_ess[sort[i]],
           ' density score ', sampled_dss[sort[i]],
           ' validity score ', sampled_vss[sort[i]],
           )
    structure.to(filename=path+'top_'+str(k)+'_full/'+'top_'+str(j)+'.cif')
    j+=1

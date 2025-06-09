import argparse
import gzip
import os
import pickle
import warnings
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from pymatgen.analysis.structure_matcher import StructureMatcher
warnings.filterwarnings('ignore')
import torch
from tqdm import tqdm
import os
import matgl
from pl_modules.reward import *

parser = argparse.ArgumentParser(description='Evaluate all metrics')
parser.add_argument('--d', type=str, required=True, help='Directory containing structure files')
args = parser.parse_args()

directory_in_str = args.directory

def form_e_eval():
    directory = os.fsencode(directory_in_str)
    eform_model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
    energy = []
    density = []
    smactvalid = []
    strucvalid = []
    count_invalid = 0
    structures = []
    rewards = []
    files = []
    count = 0
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        files.append(filename)
        try:
            structure = Structure.from_file(directory_in_str+ filename)
        except:
            continue
        atom_types = []
        for s in structure:
            atom_types.append(s.specie.Z)
        elems, comps = get_composition(atom_types)
        smact_valid = smact_validity(elems, comps)
        struct_valid = structure_validity(structure)
        density.append(structure.density)
        smactvalid.append(smact_valid)
        strucvalid.append(struct_valid)
        if (smact_valid == False) or (struct_valid == False):
            continue
        eform = eform_model.predict_structure(structure).cpu().detach().numpy()
        energy.append(eform)
        structures.append(structure)

        count+=1


    valid = smactvalid and strucvalid
    print('Smact validity:', np.mean(np.array(smactvalid)))
    print('Struct validity:', np.mean(np.array(strucvalid)))
    print('Valid:',np.mean(np.array(valid)))
    print('Average energy:', np.mean(np.array(energy)))
    stable = np.array([e < 0 for e in energy])
    print('Stable material:', np.mean(stable))
    print('Average density:', np.mean(np.array(density)))
    print(directory_in_str)

def match_rate_eval():
    directory = os.fsencode(directory_in_str)
    matcher = StructureMatcher(angle_tol=10,ltol=1.0,stol=1.0)
    # matcher = StructureMatcher()
    pairs = []
    rmss = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        file_id = filename
        opt_file_id = file_id + '_optimized.cif'
        filename = os.fsdecode(file)
        if os.path.isfile(directory_in_str+'optimized/'+opt_file_id):
            try:
                struct1 = Structure.from_file(directory_in_str+filename)
                struct2 = Structure.from_file(directory_in_str+'optimized/'+opt_file_id)
            except:
                print('cannot find path')
                continue
            match = matcher.fit(struct1=struct1,struct2=struct2)
            RMS = matcher.get_rms_dist(struct1=struct1, struct2=struct2)
            pairs.append(match)
            if match:
                rmss.append(RMS[0])
        else:
            print('error ', directory_in_str+'optimized/'+opt_file_id)
    pairs = np.array(pairs)
    rmss = np.array(rmss)
    print(directory_in_str)
    print('Match rate:', pairs.sum()/len(pairs))
    print('Average dist: ', rmss.sum()/len(rmss))


form_e_eval()
match_rate_eval()
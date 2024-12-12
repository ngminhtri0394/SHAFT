import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from common.data_utils import *
from pymatgen.core.periodic_table import Element
from pl_modules.structure import CrystalStructureCData
from torch_geometric.data import Data
import math
from alignn.data import get_torch_dataset
from torch.utils.data import DataLoader
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from pl_modules.policy_align import ALIGNN, ALIGNNConfig
from tqdm import tqdm
import tempfile
from pl_modules.graphs import PygGraph as CPygGraph
from pl_modules.PygStructureDT import *
from common.alignn import get_figshare_model

class BaseEnv:
    def __init__(self,device, ele_set='small'):
        self.all_symbol = all_chemical_symbols
        if ele_set == 'small':
            self.atoms = chemical_symbols
        elif ele_set == 'battery':
            self.atoms = battery_symbols
        else:
            self.atoms = large_chemical_symbols
        # list of Element
        self.atom_ele = [Element(atom) for atom in self.atoms]
        self.device = device
    
    def reset(self):
        raise NotImplementedError
    
    def add_atom_to(self, structure, atomidx, coord_pos):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

    def structures2repr(self, structure=None, for_proxy=False):
        raise NotImplementedError



    

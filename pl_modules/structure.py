import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from common.data_utils import *
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import math

class BaseCrystalStructure:
    def __init__(self, req_config):
        self.atomic_numbers = []
        self.atoms = [] 
        self.frac_coords = []
        self.lattice = [4.0, 4.0, 4.0, 90.0, 90.0, 90.0]
        self.num_atoms = 0
        self._structure = None
        self._structure_graph = None
        self.set_spacegroup()
        self.req_config = req_config

    def get_valid_mask_spacegroup(self):
        # logit-level masking
        num_atom_sg = [len(Structure.from_spacegroup(sg,lattice=Lattice.from_parameters(*self.lattice),
                                                species=self.atoms,
                                                coords=self.frac_coords,
                                                coords_are_cartesian=False).species) for sg in range(1,231)]
        return num_atom_sg < self.req_config['max_atom']
    
    def get_valid_mask_lattice():
        # action level masking
        pass
    
    def get_valid_mask_atom_coord():
        # action level masking
        pass

    def get_valid_mask_sg(self, max_atom):
        mask = [True] * 230
        for i in range(1,231):
            temp_struct = Structure.from_spacegroup(i,
                                                    lattice=Lattice.from_parameters(*self.lattice),
                                                    species=self.atoms,
                                                    coords=self.frac_coords,
                                                    coords_are_cartesian=False)
            if len(temp_struct.atomic_numbers) > max_atom:
                mask[i-1] = False
        return mask
    
    def get_valid_mask_atom_type(self, max_traj_len):
        # check number of element used
        used_ele = set([e.symbol for e in self.atoms])
        used_ele = [self.req_config['ele_choice'].index(e) for e in used_ele]
        num_ele_used = len(used_ele)
        if num_ele_used >= self.req_config['max_ele']:
            mask = [e in used_ele for e in range(self.req_config['len_ele_list'])]
            # print(mask)
            return mask
        if num_ele_used == self.req_config['max_ele'] - 1:
            # check if any required ele is used
            res = any(ele in self.req_config['req_ele'] for ele in used_ele)
            if not res:
                # if haven't used any required ele, set valid to those required
                mask = [e in self.req_config['req_ele'] for e in range(self.req_config['len_ele_list'])]
            else:
                # just pick any atom type
                mask = [True] * self.req_config['len_ele_list']
            return mask
        if num_ele_used < self.req_config['max_ele'] - 1:
            if len(self.atoms) == max_traj_len - 1:
                res = any(ele in self.req_config['req_ele'] for ele in used_ele)
                if not res:
                    # if haven't used any required ele, set valid to those required
                    mask = [e in self.req_config['req_ele'] for e in range(self.req_config['len_ele_list'])]
                else:
                    # just pick any atom type
                    mask = [True] * self.req_config['len_ele_list']
                return mask

        mask = [True] * self.req_config['len_ele_list']
        return mask

    def set_spacegroup(self,spacegroup=1):
        # network output is 0-229
        spacegroup += 1
        if spacegroup < 1 or spacegroup > 230:
            spacegroup = 1
        self.spacegroup=spacegroup

    def add_atom(self, atomidx, atom, coord_pos):
        raise NotImplementedError
    
    def delete_atoms(self, atom_mask):
        raise NotImplementedError
    
    @property
    def structure(self):
        self._structure = Structure.from_spacegroup(self.spacegroup,
                                                    lattice=Lattice.from_parameters(*self.lattice),
                                                    species=self.atoms,
                                                    coords=self.frac_coords,
                                                    coords_are_cartesian=False)
        return self._structure
    
    def from_structure_obj(self,structure):
        self.atoms = structure.species
        self.frac_coords = structure.frac_coords
        l = [structure._lattice.a, structure._lattice.b, structure._lattice.c]
        l.extend(structure._lattice.angles)
        self.lattice = l 
    
    @property
    def crystalNN_structure_graph(self):
        search_cutoff = 13
        CrystalNN = local_env.CrystalNN(
                    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False,search_cutoff=search_cutoff)
        crystal_graph = None
        while crystal_graph is None:
            try:
                crystal_graph = StructureGraph.with_local_env_strategy(
                self.structure, CrystalNN)
            except:
                # print('error with distance')
                # print(self.frac_coords)
                search_cutoff += 1
                CrystalNN = local_env.CrystalNN(
                    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False,search_cutoff=search_cutoff)
                pass
        self._structure_graph = crystal_graph
        return self._structure_graph
    
    def copy(self): # shallow copy
        raise NotImplementedError
    
    def as_dict(self):
        return {'atomidxs': self.atomic_numbers,
                'atoms': self.atoms,
                'frac_coords': self.frac_coords,
                'lattice': self.lattice,
                'num_atoms': self.num_atoms}



class CrystalStructureCData(BaseCrystalStructure):
    def __init__(self, req_config):
        super().__init__(req_config)
        self.complete = False
    
    def add_atom(self, atomidx, atom, coord_pos):
        self.atomic_numbers.append(atomidx)
        self.atoms.append(atom)
        self.frac_coords.append(coord_pos)
        self.num_atoms += 1
        self._structure = None
        self._structure_graph = None
        return None
    
    def add_atom_and_change_lattice(self, atomidx, atom, coord_pos,lattice):
        self.atomic_numbers.append(atomidx)
        self.atoms.append(atom)
        self.frac_coords.append(coord_pos)
        self.num_atoms += 1
        self._structure = None
        self._structure_graph = None
        self.lattice = lattice
        return None
        
    def set_lattice(self, lattice,isradiant=False):
        lattice_input = np.copy(lattice)
        if isradiant:
            lattice_input[3] = math.degrees(lattice_input[3])
            lattice_input[4] = math.degrees(lattice_input[4])
            lattice_input[5] = math.degrees(lattice_input[5])
        
        self.lattice = lattice
        self._structure = None
        self._structure_graph = None
        return None

    def delete_atoms(self, atom_mask):
        reindex = np.cumsum(np.asarray(atom_mask,np.int32)) - 1
        self.num_atoms = np.sum(np.asarray(atom_mask, dtype=np.int32))
        self.atoms = np.asarray(self.atoms)[atom_mask].tolist()
        self.frac_coords = np.asarray(self.frac_coords)[atom_mask].tolist()
        self.atomic_numbers = np.asarray(self.atomic_numbers)[atom_mask].tolist()
        self._structure = None
        self._structure_graph = None
        return reindex
    
    def copy(self): # shallow copy
        o = CrystalStructureCData(self.req_config)
        o.complete = self.complete
        o.atomic_numbers = list(self.atomic_numbers)
        o.atoms = list(self.atoms)
        o.frac_coords = list(self.frac_coords)
        o.lattice = list(self.lattice)
        o.num_atoms = self.num_atoms
        o.spacegroup = self.spacegroup
        return o

    def get_spacegroup_number(self):
        structure = self.structure
        sga = SpacegroupAnalyzer(structure,symprec=0.1)
        return sga.get_space_group_number()


    def validate_spacegroup(self):
        if self.spacegroup != self.get_spacegroup_number():
            return False
        else:
            return True
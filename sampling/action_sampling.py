from abc import ABC, abstractmethod
from policy.graph_backbone import GCNBackbone
from typing import List
from pl_modules.structure import CrystalStructureCData
import torch
import math
import numpy as np

class ActionsSampler(ABC):
    def __init__(self,
                 estimators: List[GCNBackbone],
                 req_config,
                temperature: float = 1.0,
                sf_bias: float = 0.0,
                epsilon: float = 0.0) -> None:
        super().__init__()
        self.req_config = req_config
        self.estimators = estimators
        self.temperature = temperature
        self.sf_bias = sf_bias
        self.epsilon = epsilon
        

    @abstractmethod
    def sample(self, states):
        """
        Args:
            states (States): A batch of states.

        Returns:
            Tuple[Tensor[batch_size], Tensor[batch_size]]: A tuple of tensors containing the log probabilities of the sampled actions, and the sampled actions.
        """
        pass
    
class HierarchicalActionSampler(ActionsSampler):
    """
        For Hierarchical Hybrid Enviroment which split one single action into many task level
    """
    def __init__(self, estimators: List[GCNBackbone], req_config: List,
                 temperature: float = 1, sf_bias: float = 0, 
                 epsilon: float = 0, min_stop=3,
                 min_angle = 1.39626, max_angle=0.87267,
                 min_length = 4.0, max_length = 11.0,
                 device= torch.device('cuda')) -> None:
        super().__init__(estimators, req_config, temperature, sf_bias, epsilon)
        self.min_stop = min_stop
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_length = min_length
        self.max_length = max_length
        self.epsilon = 1e-6
        self.device = device


    def get_raw_logits(self, states, policy,subgoal):
        """
        Get logits from policy 
        """
        # logits = policy(states,subgoal)
        logits = policy(states, subgoal)
        return logits

    def get_logits(self,states, policy,subgoal=None):
        """
        Get logits from policy and and mask the illigel actions
        """
        logits = self.get_raw_logits(states, policy,subgoal)
        return logits

    def get_dist(self, states, policy):
        """
        Get distribution of the from the action
        """

        return policy.to_dist(states)
        
    def get_probs(self,states,policy,subgoal=None):
        """
        Get probs 
        """
        logits = self.get_logits(states,policy,subgoal)
        probs = [] # probs for each action types
        for logit in logits: #iterate all action types performed by an agent 
            prob = torch.softmax(logit / self.temperature, dim=-1)
            probs.append(prob)
        return probs, logits
    
    def lattice_sample_to_action(self, lattice):
        lattice[:,0:3] = self.max_length*lattice[:,0:3] + self.min_length
        lattice[:,3:] = self.max_angle*lattice[:,3:] + self.min_angle
        return lattice
    
    def lattice_action_to_sample(self, lattice_action):
        lattice_action[:,0:3] = (lattice_action[:,0:3] - self.min_length)/self.max_length
        lattice_action[:,3:] = (lattice_action[:,3:] - self.min_angle)/self.max_angle
        return lattice_action

    def adjust_lattice_based_on_action(self, lattices, sg):
        spacegroup_action = sg.clone().cpu().detach().numpy()
        spacegroup_action = np.squeeze(spacegroup_action)
        # spacegroup from model is 0-229
        for idx, sg in enumerate(spacegroup_action):
            if sg in [0, 1]:
                # print('Adjust to Triclinic')    
                if (lattices[idx,3] + lattices[idx,4] + lattices[idx,5] > 5.4):
                    lattices[idx,5] = math.pi/2
            elif sg in list(range(2, 15)):
                # print('adjust to Monoclinic')
                lattices[idx,3] = math.pi/2
                lattices[idx,5] = math.pi/2
            elif sg in list(range(15, 74)):
                # print('adjust to Orthorhombic')
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = math.pi/2
            elif sg in list(range(74, 142)):
                # print('adjust to Tetragonal')
                lattices[idx,1] = lattices[idx,0]
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = math.pi/2
            elif sg in list(range(142, 194)):
                # print('Adjust to hexagonal')
                lattices[idx,1] = lattices[idx,0]
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = 2*math.pi/3
            elif sg in list(range(194, 230)):
                # print('Adjust to cubic')
                lattices[idx,1] = lattices[idx,0]
                lattices[idx,2] = lattices[idx,0]
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = math.pi/2
            else:
                continue
        return lattices


    def sample(self,states,states_reps,
               n_trajectories,
               step=0,
               maxblock=3):
        # Array of actions from all level
        levels_action = []
        logprobs = 0

        probs_esg, logits_esg = self.get_probs(states=states_reps,policy=self.estimators[0])
        dists_esg = self.estimators[0].to_dist(probs_esg, logits_esg)
        exit_dist = dists_esg[0]
        sg_dist = dists_esg[1]

        with torch.no_grad():
            if step > self.min_stop:
                exit_action = exit_dist.sample()
            else:
                exit_action = torch.zeros(n_trajectories,dtype=torch.float).to(device=self.device)
            sg_action = sg_dist.sample()

        exit_logprobs = exit_dist.log_prob(exit_action)

        probs_la, logits_la = self.get_probs(states=states_reps,policy=self.estimators[1],subgoal=sg_action)

        mask = torch.tensor(np.array([state.get_valid_mask_atom_type(max_traj_len=maxblock) for state in states])).to(device=self.device)
        logits_la[6] = torch.where(mask, logits_la[6], -float("inf"))
        dists_la = self.estimators[1].to_dist(probs_la, logits_la)
        lattice_dist = dists_la[0]
        atom_dist = dists_la[1]
        atomtype_dist = dists_la[2]

        with torch.no_grad():
            lattice_action = lattice_dist.sample()
            frac_action = atom_dist.sample()
            atype_action = atomtype_dist.sample()


        sg_logprobs = sg_dist.log_prob(sg_action)
        lattice_params_logprobs = lattice_dist.log_prob(torch.clamp(lattice_action, min=self.epsilon, max=1.0-self.epsilon))
        atom_coord_logprobs = atom_dist.log_prob(frac_action)   
        atype_logprobs = atomtype_dist.log_prob(atype_action)

        lattice_action = self.lattice_sample_to_action(lattice_action)
        lattice_action = self.adjust_lattice_based_on_action(lattice_action, sg_action)
        
        levels_action = [torch.unsqueeze(exit_action,1), 
                         torch.unsqueeze(sg_action,1), 
                         torch.squeeze(lattice_action), 
                         frac_action, 
                         torch.unsqueeze(atype_action,1)]
        action = torch.cat(levels_action, dim=1)
        logprobs += sg_logprobs + lattice_params_logprobs.sum(axis=1) + atom_coord_logprobs.sum(axis=1) + atype_logprobs

        if step > self.min_stop:
            action, logprobs = self.set_action_and_logprob_on_stop(action, logprobs, exit_logprobs, n_trajectories)
        return action, logprobs

    def set_action_and_logprob_on_stop(self, action, logprobs, exit_logprob, n_trajectories):
        terminated_mask = torch.where(action[...,0] == 1.0, True, False)
        action[terminated_mask] = torch.full(size=(int(terminated_mask.float().sum()),12), 
                                             fill_value=-float("inf"), 
                                             device=self.device)
        logprobs[terminated_mask] = exit_logprob[terminated_mask]
        return action, logprobs

class BackwardActionsSampler(HierarchicalActionSampler):
    """
    Base class for backward action sampling methods.
    """
    def __init__(self, estimators: List[GCNBackbone], req_config, temperature: float = 1, sf_bias: float = 0, epsilon: float = 0) -> None:
        super().__init__(estimators=estimators, req_config=req_config , temperature=temperature, sf_bias=sf_bias, epsilon=epsilon)

    def get_bw_dists(self,states_reps):
        # action [haction_1,haction_2,...,haction_n]
        dists= []
        for idx, p_level in enumerate(self.estimators):
            # Get probs and logits from the current state
            probs, logits = self.get_probs(states=states_reps,policy=p_level) 
            # Get distribution from probs (discret)/logits(continous)
            dist = p_level.to_dist(probs, logits) # [n_trajectories]
            dists.extend(dist)
        return dists
    
    def get_sg_bw_dists(self, states_reps):
        graph, sg, lattice, fpretrain = states_reps
        probs, logits = self.get_probs(states=states_reps,policy=self.estimators[0],subgoal=sg) 
        return self.estimators[0].to_dist(probs, logits)
    
    def get_al_bw_dists(self, states_reps, sg_action):
        probs, logits = self.get_probs(states=states_reps,policy=self.estimators[1],subgoal=sg_action) 
        return self.estimators[1].to_dist(probs, logits)


class FlatActionSampler(ActionsSampler):
    """
        For Hierarchical Hybrid Enviroment which split one single action into many task level
    """
    def __init__(self, estimators: List[GCNBackbone], req_config: List,
                 temperature: float = 1, sf_bias: float = 0, 
                 epsilon: float = 0, min_stop=3,
                 min_angle = 1.39626, max_angle=0.87267,
                 min_length = 4.0, max_length = 11.0,
                 device= torch.device('cuda')) -> None:
        super().__init__(estimators, req_config, temperature, sf_bias, epsilon)
        self.min_stop = min_stop
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_length = min_length
        self.max_length = max_length
        self.epsilon = 1e-6
        self.device = device


    def get_raw_logits(self, states, policy,subgoal):
        """
        Get logits from policy 
        """
        # logits = policy(states,subgoal)
        logits = policy(states, subgoal)
        return logits

    def get_logits(self,states, policy,subgoal=None):
        """
        Get logits from policy and and mask the illigel actions
        """
        logits = self.get_raw_logits(states, policy,subgoal)
        return logits

    def get_dist(self, states, policy):
        """
        Get distribution of the from the action
        """

        return policy.to_dist(states)
        
    def get_probs(self,states,policy,subgoal=None):
        """
        Get probs 
        """
        logits = self.get_logits(states,policy,subgoal)
        probs = [] # probs for each action types
        for logit in logits: #iterate all action types performed by an agent 
            prob = torch.softmax(logit / self.temperature, dim=-1)
            probs.append(prob)
        return probs, logits
    
    def lattice_sample_to_action(self, lattice):
        lattice[:,0:3] = self.max_length*lattice[:,0:3] + self.min_length
        lattice[:,3:] = self.max_angle*lattice[:,3:] + self.min_angle
        return lattice
    
    def lattice_action_to_sample(self, lattice_action):
        lattice_action[:,0:3] = (lattice_action[:,0:3] - self.min_length)/self.max_length
        lattice_action[:,3:] = (lattice_action[:,3:] - self.min_angle)/self.max_angle
        return lattice_action

    def adjust_lattice_based_on_action(self, lattices, sg):
        spacegroup_action = sg.clone().cpu().detach().numpy()
        spacegroup_action = np.squeeze(spacegroup_action)
        # spacegroup from model is 0-229
        for idx, sg in enumerate(spacegroup_action):
            if sg in [0, 1]:
                # print('Adjust to Triclinic')    
                if (lattices[idx,3] + lattices[idx,4] + lattices[idx,5] > 5.4):
                    lattices[idx,5] = math.pi/2
            elif sg in list(range(2, 15)):
                # print('adjust to Monoclinic')
                lattices[idx,3] = math.pi/2
                lattices[idx,5] = math.pi/2
            elif sg in list(range(15, 74)):
                # print('adjust to Orthorhombic')
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = math.pi/2
            elif sg in list(range(74, 142)):
                # print('adjust to Tetragonal')
                lattices[idx,1] = lattices[idx,0]
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = math.pi/2
            elif sg in list(range(142, 194)):
                # print('Adjust to hexagonal')
                lattices[idx,1] = lattices[idx,0]
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = 2*math.pi/3
            elif sg in list(range(194, 230)):
                # print('Adjust to cubic')
                lattices[idx,1] = lattices[idx,0]
                lattices[idx,2] = lattices[idx,0]
                lattices[idx,3] = math.pi/2
                lattices[idx,4] = math.pi/2
                lattices[idx,5] = math.pi/2
            else:
                continue
        return lattices


    def sample(self,states,states_reps,
               n_trajectories,
               step=0,
               maxblock=3):
        # Array of actions from all level
        levels_action = []
        logprobs = 0

        probs, logits = self.get_probs(states=states_reps,policy=self.estimators[0])
        
        mask = torch.tensor(np.array([state.get_valid_mask_atom_type(max_traj_len=maxblock) for state in states])).to(device=self.device)
        logits[8] = torch.where(mask, logits[8], -float("inf"))

        dists = self.estimators[0].to_dist(probs, logits)
        exit_dist = dists[0]
        sg_dist = dists[1]
        lattice_dist = dists[2]
        atom_dist = dists[3]
        atomtype_dist = dists[4]

        with torch.no_grad():
            if step > self.min_stop:
                exit_action = exit_dist.sample()
            else:
                exit_action = torch.zeros(n_trajectories,dtype=torch.float).to(device=self.device)
            sg_action = sg_dist.sample()
            lattice_action = lattice_dist.sample()
            frac_action = atom_dist.sample()
            atype_action = atomtype_dist.sample()


        exit_logprobs = exit_dist.log_prob(exit_action)
        sg_logprobs = sg_dist.log_prob(sg_action)
        lattice_params_logprobs = lattice_dist.log_prob(torch.clamp(lattice_action, min=self.epsilon, max=1.0-self.epsilon))
        atom_coord_logprobs = atom_dist.log_prob(frac_action)   
        atype_logprobs = atomtype_dist.log_prob(atype_action)

        lattice_action = self.lattice_sample_to_action(lattice_action)
        lattice_action = self.adjust_lattice_based_on_action(lattice_action, sg_action)
        
        levels_action = [torch.unsqueeze(exit_action,1), 
                         torch.unsqueeze(sg_action,1), 
                         torch.squeeze(lattice_action), 
                         frac_action, 
                         torch.unsqueeze(atype_action,1)]
        action = torch.cat(levels_action, dim=1)
        logprobs += sg_logprobs + lattice_params_logprobs.sum(axis=1) + atom_coord_logprobs.sum(axis=1) + atype_logprobs

        if step > self.min_stop:
            action, logprobs = self.set_action_and_logprob_on_stop(action, logprobs, exit_logprobs, n_trajectories)
        return action, logprobs

    def set_action_and_logprob_on_stop(self, action, logprobs, exit_logprob, n_trajectories):
        terminated_mask = torch.where(action[...,0] == 1.0, True, False)
        action[terminated_mask] = torch.full(size=(int(terminated_mask.float().sum()),12), 
                                             fill_value=-float("inf"), 
                                             device=self.device)
        logprobs[terminated_mask] = exit_logprob[terminated_mask]
        return action, logprobs
    


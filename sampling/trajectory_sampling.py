import copy
from sampling.action_sampling import *
from env.graph_crystal_env import HierGraphCrystalEnv


class BaseTrajectoriesSampler:
    def __init__(self, 
                 req_config: List,
                 action_sampler: ActionsSampler,
                 bwaction_sampler: BackwardActionsSampler,
                 env: HierGraphCrystalEnv = None,
                 device = torch.device('cuda'),
                 max_blocks = 4) -> None:
        self.req_config = req_config
        self.action_sampler = action_sampler
        self.bwaction_sampler = bwaction_sampler
        self.env = env
        self.device = device
        self.max_blocks = max_blocks

    @abstractmethod
    def sample_trajectories(self, n_trajectories):
        pass
    
    @abstractmethod
    def sample(self, n_trajectories):
        pass
    
    @abstractmethod
    def evaluate_backward_logprobs(self, trajectories, actionss,n_trajectories):
        pass


class TrajectoriesSampler(BaseTrajectoriesSampler):
    def __init__(self, 
                 req_config,
                 action_sampler: HierarchicalActionSampler,
                 bwaction_sampler: BackwardActionsSampler,
                 env: HierGraphCrystalEnv = None,
                 device = torch.device('cuda'),
                 max_blocks = 20,
                 min_stop = 4) -> None:
        super().__init__(action_sampler=action_sampler, bwaction_sampler= bwaction_sampler, env= env, device= device, max_blocks= max_blocks, req_config=req_config)
        self.min_stop = min_stop
        self.device = device

    def sample_trajectories(self):
        n_trajectories = self.env.batch_size

        trajectories = [copy.deepcopy(self.env.structures)]
        
        trajectories_logprobs = torch.zeros((n_trajectories,), device=self.device)
        all_logprobs = []
        actionss = []
        states_reps = []
        t = 0
        while any([x.complete == False for x in self.env.structures]) and (t < self.max_blocks):
            step_logprobs = torch.full((n_trajectories,), -float("inf"), device=self.device)
            actions = torch.full((n_trajectories, 12), -float("inf"), device=self.device)
            non_terminal_mask = np.asarray([i.complete == False for i in self.env.structures])

            if np.sum(non_terminal_mask) == 0: #all states have been terminated
                break

            non_terminal_states_reps = self.env.structures2repr(mask=non_terminal_mask)
            non_terminal_states = self.env.structures[non_terminal_mask]
            non_terminal_actions, logprobs = self.action_sampler.sample(states=non_terminal_states,states_reps=non_terminal_states_reps, 
                                                                        n_trajectories=n_trajectories,
                                                                        step=t, maxblock=self.max_blocks)
            states_reps.append(non_terminal_states_reps)
            actions[non_terminal_mask] = non_terminal_actions
            self.env.step(actions, t, self.min_stop)
            trajectories_logprobs[non_terminal_mask] += logprobs
            trajectories.append(copy.deepcopy(self.env.structures))
            actionss.append(actions)
            step_logprobs[non_terminal_mask] = logprobs
            all_logprobs.append(step_logprobs)
            t += 1

        trajectories = np.array(trajectories)
        trajectories = np.transpose(trajectories)
        actionss = torch.stack(actionss, dim=1)
        all_logprobs = torch.stack(all_logprobs, dim=1)
        return trajectories, actionss, trajectories_logprobs, all_logprobs, states_reps


    def sample(self, n_trajectories):
        return self.sample_trajectories(n_trajectories)
    

    def evaluate_backward_logprobs(self, trajectories, actionss, n_trajectories, states_reps):
        logprobs = torch.zeros((trajectories.shape[0],), device=self.device)
        # for i in range(trajectories.shape[1] - 2, 1, -1):
        #     non_sink_mask = np.asarray([m.complete != True for m in trajectories[:, i]])
        #     state_rep=states_reps[i]
        #     action = actionss[:,i-1][non_sink_mask]
        # for i in range(len(states_reps)):
        #     state_rep=states_reps[i]
        # for i in range(trajectories.shape[1]):
        #     non_sink_mask = np.asarray([m.complete != True for m in trajectories[:, i]])
            
        for i in range(trajectories.shape[1] - 2, 1, -1):
            state_rep=states_reps[i]
            if state_rep == None:
                print('skip')
                continue
            all_step_logprobs = torch.full(
                (trajectories.shape[0],), -float("inf"), device=self.device
            )
            non_sink_mask = np.asarray([m.complete != True for m in trajectories[:, i]])
            if np.sum(non_sink_mask) == 0:
                continue

            action = actionss[:,i-1][non_sink_mask] 
            sg_dist = self.bwaction_sampler.get_sg_bw_dists(states_reps=state_rep)[1]
            al_dist = self.bwaction_sampler.get_al_bw_dists(states_reps=state_rep,sg_action=action[:,1])
            lattice_dist = al_dist[0]
            atom_dist = al_dist[1]
            at_dist = al_dist[2]

            lattice_action = self.action_sampler.lattice_action_to_sample(action[:,2:8])
            
            step_logprobs = (
                        sg_dist.log_prob(action[:,1])
                        +lattice_dist.log_prob(torch.clamp(lattice_action,self.action_sampler.epsilon,1.0-self.action_sampler.epsilon)).sum(dim=1)
                        +atom_dist.log_prob(action[:,8:11]).sum(dim=1)
                        +at_dist.log_prob(action[:,11])
            )

            if torch.any(torch.isnan(step_logprobs)):
                raise ValueError("NaN in backward logprobs")

            if torch.any(torch.isinf(step_logprobs)):
                raise ValueError("Inf in backward logprobs")

            logprobs[non_sink_mask] += step_logprobs
            all_step_logprobs[non_sink_mask] = step_logprobs

        return logprobs    


class FlatTrajectoriesSampler(TrajectoriesSampler):
    def evaluate_backward_logprobs(self, trajectories, actionss, n_trajectories, states_reps):
        logprobs = torch.zeros((trajectories.shape[0],), device=self.device)
        for i in range(trajectories.shape[1] - 2, 1, -1):
            non_sink_mask = np.asarray([m.complete != True for m in trajectories[:, i]])
            state_rep=states_reps[i]
            action = actionss[:,i-1][non_sink_mask]
        for i in range(len(states_reps)):
            state_rep=states_reps[i]
        for i in range(trajectories.shape[1]):
            non_sink_mask = np.asarray([m.complete != True for m in trajectories[:, i]])
            
        for i in range(trajectories.shape[1] - 2, 1, -1):
            state_rep=states_reps[i]
            all_step_logprobs = torch.full(
                (trajectories.shape[0],), -float("inf"), device=self.device
            )
            non_sink_mask = np.asarray([m.complete != True for m in trajectories[:, i]])
            if np.sum(non_sink_mask) == 0:
                continue

            action = actionss[:,i-1][non_sink_mask] 
            dist = self.bwaction_sampler.get_bw_dists(states_reps=state_rep)
            sg_dist = dist[1]
            lattice_dist = dist[2]
            atom_dist = dist[3]
            at_dist = dist[4]

            lattice_action = self.action_sampler.lattice_action_to_sample(action[:,2:8])
            step_logprobs = (
                            sg_dist.log_prob(action[:,1])
                            +lattice_dist.log_prob(torch.clamp(lattice_action,self.action_sampler.epsilon,1.0-self.action_sampler.epsilon)).sum(dim=1)
                            +atom_dist.log_prob(action[:,8:11]).sum(dim=1)
                            +at_dist.log_prob(action[:,11])
            )
            

            if torch.any(torch.isnan(step_logprobs)):
                raise ValueError("NaN in backward logprobs")

            if torch.any(torch.isinf(step_logprobs)):
                raise ValueError("Inf in backward logprobs")

            logprobs[non_sink_mask] += step_logprobs
            all_step_logprobs[non_sink_mask] = step_logprobs

        return logprobs
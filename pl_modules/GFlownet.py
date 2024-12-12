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

warnings.filterwarnings('ignore')
from sampling.action_sampling import BackwardActionsSampler, FlatActionSampler
from sampling.trajectory_sampling import FlatTrajectoriesSampler
from env.graph_crystal_env import HierGraphCrystalEnv, HierGraphMEGNetCrystalEnv
from pl_modules.proxy import M3gnetDGL_Proxy
import torch
# torch.set_printoptions(profile="full")
from policy.crystal_policy import CrystalPolicy, BWCrystalPolicy
from policy.graph_backbone import GCNBackbone, MEGNetBackbone
import os
from functools import partialmethod
from tqdm import tqdm, trange
from metrics.eval_metrics import *
from pl_modules.reward import reward_functions_dict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class GFlownet():
    def __init__(self,
                 max_blocks,
                 device,
                 lr,
                 lr_Z,
                 scheduler_milestone,
                 gamma_scheduler,
                 initlogZ,
                 n_iterations,
                 alpha_schedule,
                 alpha,
                 clampmin,
                 clampmax,
                 batch_size,
                 save_freq,
                 phidden_dim,
                 use_pretrain,
                 pretrain_model_name,
                 proxy_model,
                 with_stop,
                 min_stop,
                 policy_nn,
                 ele_set,
                 max_ele,
                 req_ele,
                 max_atom,
                 vpen_min,
                 vpen_max,
                 vpen_minmax,
                 reward, 
                 reward_beta,
                 reward_min):
        self.reward = reward_functions_dict[reward]
        self.vpen_min = vpen_min
        self.vpen_max = vpen_max
        self.vpen_minmax = vpen_minmax
        self.reward_min = reward_min
        self.max_blocks = max_blocks
        self.phidden_dim = phidden_dim
        self.with_stop = with_stop
        self.min_stop = min_stop
        self.device = torch.device(device)
        self.lr = lr
        self.lr_Z = lr_Z
        self.scheduler_milestone = scheduler_milestone
        self.gamma_scheduler = gamma_scheduler
        self.initlogZ = initlogZ
        self.n_iterations = n_iterations
        self.alpha_schedule = alpha_schedule
        self.alpha = alpha
        self.clampmin = clampmin
        self.clampmax = clampmax
        self.batch_size = batch_size
        self.sampled_mols = []
        self.sampled_reward = []
        self.sampled_bs = []
        self.sampled_es = []
        self.sampled_vs = []
        self.sampled_ds = []
        self.train_infos = []
        self.save_freq = save_freq
        self.use_pretrain = use_pretrain
        self.pretrain_model_name = pretrain_model_name
        self.logZ = nn.Parameter(torch.tensor(self.initlogZ).to(device=self.device))   
        self.proxy_model = proxy_model
        self.proxy = M3gnetDGL_Proxy(self.proxy_model)
        self.policy_nn = policy_nn
        self.ele_set = ele_set
        self.max_atom = max_atom
        self.reward_beta = reward_beta
        if self.pretrain_model_name == 'matformer':
            self.pretrain_dim = 128
        else: 
            self.pretrain_dim = 256
        self.init_env()
        self.num_elementset = len(self.env.atoms)
        req_elementidx = [self.env.atoms.index(e) for e in req_ele]
        self.req_config = {'max_ele':max_ele,
                           'req_ele':req_elementidx,
                           'max_atom':max_atom,
                           'len_ele_list': len(self.env.atoms),
                           'ele_choice': self.env.atoms}
        self.env.set_req(self.req_config)
        self.env.reset()
        self.init_backbone()
        self.init_policy()
        self.init_sampler()


    def init_env(self):
        if self.policy_nn == 'graph':
            self.env = HierGraphCrystalEnv(device=self.device, pretrain_model=self.pretrain_model_name, ele_set=self.ele_set,batch_size=self.batch_size)
        elif self.policy_nn =='graph_megnet':
            self.env = HierGraphMEGNetCrystalEnv(device=self.device, pretrain_model=self.pretrain_model_name, ele_set=self.ele_set,batch_size=self.batch_size,max_atom=self.max_atom)
            
    def init_backbone(self):
        if self.policy_nn == 'graph_megnet':
            self.backbone = MEGNetBackbone(use_pretrain=self.use_pretrain,
                                                hidden_dim=self.phidden_dim,
                                                pretrain_dim=self.pretrain_dim)

    def init_policy(self, path=None):
        if self.policy_nn == 'graph':
            raise NotImplementedError
            # self.sgpolicy = GCNSpaceGroupPolicy(use_pretrain=self.use_pretrain,
            #                                 hidden_dim=self.phidden_dim,
            #                                 pretrain_dim=self.pretrain_dim,
            #                                 n_atom=n_atom).to(device=self.device)
            # self.latticeatompolicy = GCNLatticeAtomPolicy(use_pretrain=self.use_pretrain,
            #                                 hidden_dim=self.phidden_dim,
            #                                 pretrain_dim=self.pretrain_dim,
            #                                 n_element=self.num_elementset,
            #                                 n_atom=n_atom).to(device=self.device)
            # self.bwsgpolicy = BWGCNSpaceGroupPolicy(use_pretrain=self.use_pretrain,
            #                                         hidden_dim=self.phidden_dim,
            #                                         pretrain_dim=self.pretrain_dim,
            #                                         n_atom=n_atom).to(device=self.device)
            # self.bwlatticeatompolicy = BWGCNLatticeAtomPolicy(use_pretrain=self.use_pretrain,
            #                                 hidden_dim=self.phidden_dim,
            #                                 pretrain_dim=self.pretrain_dim,
            #                                 n_element=self.num_elementset,
            #                                 n_atom=n_atom).to(device=self.device)
        if self.policy_nn == 'graph_megnet':
            backbone = MEGNetBackbone(use_pretrain=self.use_pretrain,
                                            hidden_dim=self.phidden_dim,
                                            pretrain_dim=self.pretrain_dim)
            bwbackbone = MEGNetBackbone(use_pretrain=self.use_pretrain,
                                            hidden_dim=self.phidden_dim,
                                            pretrain_dim=self.pretrain_dim)
            self.policy = CrystalPolicy(backbone=backbone,use_pretrain=self.use_pretrain,
                                            hidden_dim=self.phidden_dim,
                                            pretrain_dim=self.pretrain_dim,
                                            n_element=self.num_elementset,batchsize=self.batch_size).to(device=self.device)
            self.bwpolicy = BWCrystalPolicy(backbone=bwbackbone,use_pretrain=self.use_pretrain,
                                            hidden_dim=self.phidden_dim,
                                            pretrain_dim=self.pretrain_dim,
                                            n_element=self.num_elementset,batchsize=self.batch_size).to(device=self.device)
        if path != None:
            self.policy.load_state_dict(torch.load(f'{path}'+'/saved_weight_to_sample/policy_fp.pt')['model_state_dict'])
            self.bwpolicy.load_state_dict(torch.load(f'{path}'+'/saved_weight_to_sample/policy_bw.pt')['model_state_dict'])
                         
        self.bwhpolicylist = [self.bwpolicy]
        self.hpolicylist = [self.policy]
                

    def init_sampler(self):
        self.action_sampler = FlatActionSampler(estimators=self.hpolicylist,
                                                                min_stop=self.min_stop,
                                                                req_config=self.req_config,
                                                                device=self.device)
        self.bw_sampler = BackwardActionsSampler(estimators=self.bwhpolicylist,
                                                                    req_config=self.req_config)
        self.trajectory_sampling = FlatTrajectoriesSampler(action_sampler=self.action_sampler, 
                                                        bwaction_sampler= self.bw_sampler, 
                                                        env=self.env,
                                                        max_blocks=self.max_blocks,
                                                        min_stop=self.min_stop,
                                                        req_config=self.req_config
                                                        )

    
    def save_info(self,iter):
        exp_dir = HydraConfig.get().run.dir
        sampled = zip(self.sampled_mols, self.sampled_reward, 
                      self.sampled_bs, self.sampled_ds, self.sampled_es, self.sampled_vs)
        if not os.path.isdir(f'{exp_dir}/'+'saved_data/'):
            os.makedirs(f'{exp_dir}/'+'saved_data/')
        
        pickle.dump(sampled,
                        gzip.open(f'{exp_dir}/' +'saved_data/' + str(iter) + '_sampled_mols.pkl.gz', 'wb'))

        pickle.dump(self.train_infos,
                        gzip.open(f'{exp_dir}/' +'saved_data/' + str(iter) + '_train_info.pkl.gz', 'wb'))
        if self.policy_nn == 'graph_dimenetpp' or self.policy_nn == 'graph_megnet':
            torch.save({'model_state_dict': self.hpolicylist[0].state_dict(),
                    }, f'{exp_dir}/'+'policy_fp.pt')
            torch.save({'model_state_dict': self.bwhpolicylist[0].state_dict(),
                    }, f'{exp_dir}/'+'policy_bw.pt')
        else:
            torch.save(self.hpolicylist[0], f'{exp_dir}/'+'policy_fp_sg.pt')

            torch.save(self.bwhpolicylist[0], f'{exp_dir}/'+'policy_bw_sg.pt')

        torch.save(self.logZ, f'{exp_dir}/'+'logz.pt')
        self.sampled_mols = []
        self.sampled_reward = []
        self.sampled_bs = []
        self.sampled_es = []
        self.sampled_vs = []
        self.sampled_ds = []
        
    def train_model_with_proxy(self):
        print('Exp dir: ', HydraConfig.get().run.dir)
        optimizer = torch.optim.Adam(self.hpolicylist[0].parameters(), lr=self.lr)
        optimizer.add_param_group({"params": self.bwhpolicylist[0].parameters(), "lr": self.lr})
        optimizer.add_param_group({"params": [self.logZ], "lr": self.lr_Z})
        # self.scheduler_milestone = 5000
        # self.gamma_scheduler = 1.0
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i * self.scheduler_milestone for i in range(1, 10)],
            gamma=self.gamma_scheduler,
        )

        ####################
        current_alpha = self.alpha * self.alpha_schedule
        tr = trange(self.n_iterations+1)
        currentmax = 0
        for i in tr:
            # print('Epoch: ', i)
            if i % 1000 == 0:
                current_alpha = max(current_alpha / self.alpha_schedule, 1.0)
                # print(f"current optimizer LR: {optimizer.param_groups[0]['lr']}")

            optimizer.zero_grad()

            trajectories, actionss, logprobs, all_logprobs,states_reps = self.trajectory_sampling.sample_trajectories()
            last_states = trajectories[:,-1]
            
            reward, forme_score, bond_score, valid_score, density_score  = self.reward(last_states, self.proxy, self.vpen_min, self.vpen_max, self.vpen_minmax,self.reward_min)
            
            self.sampled_mols.append(last_states)
            self.sampled_reward.append(reward)
            self.sampled_bs.append(bond_score)
            self.sampled_es.append(forme_score)
            self.sampled_vs.append(valid_score)
            self.sampled_ds.append(density_score)

            reward = np.power(reward,self.reward_beta)
            logrewards = torch.Tensor(reward).to(device=self.device).log()
            max_reward = np.max(reward)
            mean_bs = np.mean(bond_score)
            mean_es = np.mean(forme_score)
            mean_vs = np.mean(valid_score)
            mean_ds = np.mean(density_score)

            bw_logprobs = self.trajectory_sampling.evaluate_backward_logprobs(trajectories, actionss, self.batch_size, states_reps)

            loss = torch.mean((self.logZ + logprobs - bw_logprobs - logrewards) ** 2)
            loss.backward()
            if torch.isnan(loss):
                print(self.logZ)
                print(logprobs)
                print(bw_logprobs)
                print(logrewards)
                print(reward)
                raise ValueError('loss is nan')
            # print('Loss: ', loss.item())
            # print('Max reward: ', max_reward)
            mean_reward = np.mean(reward)
            # print('Mean reward: ', np.mean(reward))
            if max_reward > currentmax:
                currentmax = max_reward
            # print('Current max: ', currentmax)
            tr.set_postfix({'r': np.mean(reward),
                            'bs': mean_bs,
                            'es': mean_es,
                            'vs': mean_vs,
                            'ds': mean_ds})
            self.train_infos.append((loss.item(),max_reward,mean_reward,currentmax,mean_bs,mean_es,mean_vs,mean_ds))
            # clip the gradients for bw_model
            for model_bw in self.bwhpolicylist:
                for p in model_bw.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(self.clampmin, self.clampmax).nan_to_num_(0.0)
            for model in self.hpolicylist:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(self.clampmin, self.clampmax).nan_to_num_(0.0)
            optimizer.step()
            scheduler.step()
            if i % self.save_freq == 0:
                self.save_info(i)
            # tr.set_postfix(Loss=loss.item(),MeanReward=max_reward,logz=self.logZ.item())
            self.env.reset()

    def load_sampling_model(self,save_path):
        self.init_policy(save_path)
        self.init_sampler()

    def sample(self, number_to_sample):
        save_path = HydraConfig.get().run.dir
        self.load_sampling_model(save_path=save_path)
        sampled_mols = []
        if not os.path.isdir(save_path+'/sample/'):
            os.makedirs(save_path+'/sample/')
        idx = 0
        for i in tqdm(range(0, number_to_sample, self.batch_size)):
            trajectories, _, _, _,_ = self.trajectory_sampling.sample_trajectories()
            cdstructure = trajectories[:,-1]
            for s in cdstructure:
                struct = s.structure
                sampled_mols.append(struct)
                struct.to(filename=save_path+'/sample/'+str(idx)+'.cif')
                idx += 1
            self.env.reset()
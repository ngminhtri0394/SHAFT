from policy.graph_backbone import GCNBackbone, DimeNetPlusPlusBackbone
from policy.backbone import MLPPolicyBackbone
from torch.distributions import Categorical, Bernoulli
import torch.nn as nn
import torch 


class GCNSpaceGroupPolicy(GCNBackbone):
    def __init__(self, dim=4, hidden_dim=128, n_hidden=2, output_dim=32, prob_output=231, use_pretrain=False, pretrain_dim=128,atom_dist_n_comp=1,batchsize=32):
        super().__init__(dim, hidden_dim, n_hidden, output_dim, use_pretrain, pretrain_dim)
        # This version consider the current space group and decide whether to keep the space group
        self.softmax = nn.Softmax(dim=1)
        self.outlayer = nn.Linear(output_dim, prob_output)
        self.spacegroup_emb =nn.Embedding(230, output_dim)

        self.PFs0 = nn.ParameterDict(
            {
                "logits": nn.Parameter(torch.zeros(batchsize,prob_output-1)),
                "logits_exit": nn.Parameter(torch.zeros(batchsize)),
            }
        )
    
    def forward(self, s, sub):
        graph, _, _, _ = s
        if graph is not None:
            out = super().forward(s)
            out = self.outlayer(out)
            out_stop = out[...,0]
            out_sg = out[...,1:]
        else:
            out_sg = self.PFs0["logits"]
            out_stop = self.PFs0["logits_exit"]
        return [out_stop, out_sg]
    
    def to_dist_from_state(self, s, sub):
        logits = self.forward(s, sub)
        probs = self.softmax(logits)
        dist = Categorical(probs=probs)
        return logits, probs, dist
    
    def to_dist(self, probs, logits):
        sg_dist = Categorical(logits=logits[1])
        exit_dist = Bernoulli(logits=logits[0])
        return [exit_dist, sg_dist]

        
class BWGCNSpaceGroupPolicy(GCNBackbone):
    def __init__(self, dim=4, hidden_dim=128, n_hidden=2, output_dim=32, prob_output=231, use_pretrain=False, pretrain_dim=128,atom_dist_n_comp=1,batchsize=32):
        super().__init__(dim, hidden_dim, n_hidden, output_dim, use_pretrain, pretrain_dim)
        # This version consider the current space group and decide whether to keep the space group
        self.softmax = nn.Softmax(dim=1)
        self.outlayer = nn.Linear(output_dim, prob_output)
        self.spacegroup_emb =nn.Embedding(230, output_dim)
    
    def forward(self, s, sub):
        out = super().forward(s)
        out = self.outlayer(out)
        out_stop = out[...,0]
        out_sg = out[...,1:]
        return [out_stop, out_sg]
    
    def to_dist_from_state(self, s, sub):
        logits = self.forward(s, sub)
        probs = self.softmax(logits)
        dist = Categorical(probs=probs)
        return logits, probs, dist
    
    def to_dist(self, probs, logits):
        sg_dist = Categorical(logits=logits[1])
        exit_dist = Bernoulli(logits=logits[0])
        return [exit_dist, sg_dist]
    

class SpaceGroupPolicy(torch.nn.Module):
    def __init__(self, backbone, dim=4, hidden_dim=128, n_hidden=2, output_dim=32, 
                 prob_output=231, use_pretrain=False, pretrain_dim=128,
                 atom_dist_n_comp=1,batchsize=32,dist_config='discrete'):
        super().__init__()
        self.backbone = backbone
        # This version consider the current space group and decide whether to keep the space group
        self.softmax = nn.Softmax(dim=1)
        self.outlayer = nn.Linear(output_dim, prob_output)
        self.spacegroup_emb =nn.Embedding(230, output_dim)

        self.PFs0 = nn.ParameterDict(
            {
                "logits": nn.Parameter(torch.zeros(batchsize,prob_output-1)),
                "logits_exit": nn.Parameter(torch.zeros(batchsize)),
            }
        )

        if dist_config =='discrete':
            self.sg_dist = Categorical
            self.exit_dist = Bernoulli
        else:
            NotImplementedError
    
    def forward(self, s, sub):
        graph, _, _, _ = s
        if graph is not None:
            out = self.backbone(s)
            # print('out: ',out)
            out = self.outlayer(out)
            out_stop = out[...,0]
            out_sg = out[...,1:]
        else:
            out_sg = self.PFs0["logits"]
            out_stop = self.PFs0["logits_exit"]
        return [out_stop, out_sg]
    
    def to_dist(self, probs, logits):
        sg_dist = self.sg_dist(logits=logits[1])
        exit_dist = self.exit_dist(logits=logits[0])
        return [exit_dist, sg_dist]
        
class BWSpaceGroupPolicy(torch.nn.Module):
    def __init__(self, backbone, dim=4, hidden_dim=128, n_hidden=2, output_dim=32,
                  prob_output=231, use_pretrain=False, pretrain_dim=128,
                  atom_dist_n_comp=1,batchsize=32,dist_config='discrete'):
        super().__init__()
        self.backbone = backbone
        # This version consider the current space group and decide whether to keep the space group
        self.softmax = nn.Softmax(dim=1)
        self.outlayer = nn.Linear(output_dim, prob_output)
        self.spacegroup_emb =nn.Embedding(230, output_dim)
    
    def forward(self, s, sub):
        out = self.backbone(s)
        out = self.outlayer(out)
        out_stop = out[...,0]
        out_sg = out[...,1:]
        return [out_stop, out_sg]
    
    def to_dist(self, probs, logits):
        sg_dist = Categorical(logits=logits[1])
        exit_dist = Bernoulli(logits=logits[0])
        return [exit_dist, sg_dist]
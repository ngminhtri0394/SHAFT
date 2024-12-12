from policy.graph_backbone import GCNBackbone
from torch.distributions import MultivariateNormal, VonMises, Normal, Categorical, MixtureSameFamily,Beta
import torch
import torch.nn as nn
  

class LatticeAtomPolicy(torch.nn.Module):
    def __init__(self, backbone, dim=4, hidden_dim=64, n_hidden=2, output_dim=32, use_pretrain=True, 
                 pretrain_dim=128, n_element=8, device=torch.device('cuda'),
                 lattice_dist_n_dim=6, lattice_dist_n_comp=1,
                 atom_dist_n_dim=3, atom_dist_n_comp=1, min_beta=0.1, max_beta=2.0, batchsize=32):
        super().__init__()
        self.backbone = backbone
        self.prob_output = lattice_dist_n_dim*lattice_dist_n_comp*3 + atom_dist_n_dim*atom_dist_n_comp*3 + n_element
        self.n_element = n_element
        sg_dim = 64
        self.spacegroup_emb =nn.Embedding(230, sg_dim)
        self.outlayer = nn.Linear(output_dim + sg_dim, self.prob_output)
        self.device = device
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.lattice_dist_n_dim = lattice_dist_n_dim
        self.lattice_dist_n_comp = lattice_dist_n_comp
        self.atom_dist_n_dim = atom_dist_n_dim
        self.atom_dist_n_comp = atom_dist_n_comp

        self.PFs0 = nn.ParameterDict(
            {
                "log_alpha_lattice": nn.Parameter(torch.zeros(batchsize,lattice_dist_n_dim*lattice_dist_n_comp)),
                "log_alpha_atom": nn.Parameter(torch.zeros(batchsize,atom_dist_n_dim*atom_dist_n_comp)),
                "log_beta_lattice": nn.Parameter(torch.zeros(batchsize,lattice_dist_n_dim*lattice_dist_n_comp)),
                "log_beta_atom": nn.Parameter(torch.zeros(batchsize,atom_dist_n_dim*atom_dist_n_comp)),
                "logits_atom": nn.Parameter(torch.zeros(batchsize,atom_dist_n_dim*atom_dist_n_comp)),
                "logits_lattice": nn.Parameter(torch.zeros(batchsize,lattice_dist_n_dim*lattice_dist_n_comp)),
                "logits_type": nn.Parameter(torch.zeros(batchsize,n_element))
            })


    def forward(self, s, sub):
        atom, sg, lattice, fpretrain = s
        if atom != None:
            out = self.backbone(s)
            sgemb = self.spacegroup_emb(sub.type(torch.int64))
            out = torch.cat((out, sgemb), dim=1)
            out = self.outlayer(out)

            pos = self.lattice_dist_n_dim*self.lattice_dist_n_comp
            mixture_log_lattice =  out[..., :pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
            alpha_lattice = self.max_beta*torch.sigmoid(out[..., pos:2*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta
            beta_lattice = self.max_beta*torch.sigmoid(out[..., 2*pos:3*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta
            apos = self.atom_dist_n_comp*self.atom_dist_n_dim
            mixture_log_atom =  out[..., 3*pos:3*pos+apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
            alpha_atom = self.max_beta*torch.sigmoid(out[..., 3*pos+apos:3*pos+2*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp) ) + self.min_beta
            beta_atom = self.max_beta*torch.sigmoid(out[..., 3*pos+2*apos:3*pos+3*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)) + self.min_beta
            type_logits = out[...,3*pos+3*apos:].reshape(-1, self.n_element)
        else:
            mixture_log_atom = self.PFs0['logits_atom'].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
            log_alpha_atom = self.PFs0['log_alpha_atom'].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
            alpha_atom = self.max_beta*torch.sigmoid(log_alpha_atom) + self.min_beta
            log_beta_atom = self.PFs0['log_beta_atom'].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
            beta_atom  = self.max_beta*torch.sigmoid(log_beta_atom) + self.min_beta

            mixture_log_lattice = self.PFs0['logits_lattice'].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
            log_alpha_lattice = self.PFs0['log_alpha_lattice'].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
            alpha_lattice = self.max_beta*torch.sigmoid(log_alpha_lattice) + self.min_beta
            log_beta_lattice = self.PFs0['log_beta_lattice'].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
            beta_lattice = self.max_beta*torch.sigmoid(log_beta_lattice) + self.min_beta

            type_logits = self.PFs0['logits_type'].reshape(-1, self.n_element)

        return [mixture_log_lattice, alpha_lattice, beta_lattice,
                mixture_log_atom, alpha_atom, beta_atom, type_logits]
    
    def to_dist(self, probs, logits):
        lattice_dist = MixtureSameFamily(Categorical(logits=logits[0]),
                                         Beta(logits[1], logits[2]))
        atom_dist = MixtureSameFamily(Categorical(logits=logits[3]),
                                         Beta(logits[4], logits[5]))

        atomtype_dist = Categorical(logits=logits[6])
        return [lattice_dist, atom_dist, atomtype_dist]
        
        
class BWLatticeAtomPolicy(torch.nn.Module):
    def __init__(self, backbone, dim=4, hidden_dim=64, n_hidden=2, output_dim=32, use_pretrain=True, 
                 pretrain_dim=128, n_element=8, device=torch.device('cuda'),
                 lattice_dist_n_dim=6, lattice_dist_n_comp=1,
                 atom_dist_n_dim=3, atom_dist_n_comp=1, min_beta=0.1, max_beta=2.0, batchsize=32):
        # n_atom=3, dim=4, hidden_dim=128, n_hidden=2, output_dim=28,use_pretrain=False,pretrain_dim=128
        super().__init__()
        self.backbone = backbone
        self.prob_output = lattice_dist_n_dim*lattice_dist_n_comp*3 + atom_dist_n_dim*atom_dist_n_comp*3 + n_element
        self.n_element = n_element
        sg_dim = 64
        self.spacegroup_emb =nn.Embedding(230, sg_dim)
        self.outlayer = nn.Linear(output_dim + sg_dim, self.prob_output)
        self.device = device
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.lattice_dist_n_dim = lattice_dist_n_dim
        self.lattice_dist_n_comp = lattice_dist_n_comp
        self.atom_dist_n_dim = atom_dist_n_dim
        self.atom_dist_n_comp = atom_dist_n_comp


    def forward(self, s, sub):
        out = self.backbone(s)
        sgemb = self.spacegroup_emb(sub.type(torch.int64))
        out = torch.cat((out, sgemb), dim=1)
        out = self.outlayer(out)

        pos = self.lattice_dist_n_dim*self.lattice_dist_n_comp
        mixture_log_lattice =  out[..., :pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
        alpha_lattice = self.max_beta*torch.sigmoid(out[..., pos:2*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta

        beta_lattice = self.max_beta*torch.sigmoid(out[..., 2*pos:3*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta
        apos = self.atom_dist_n_comp*self.atom_dist_n_dim
        mixture_log_atom =  out[..., 3*pos:3*pos+apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
        alpha_atom = self.max_beta*torch.sigmoid(out[..., 3*pos+apos:3*pos+2*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp) ) + self.min_beta
        beta_atom = self.max_beta*torch.sigmoid(out[..., 3*pos+2*apos:3*pos+3*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)) + self.min_beta
        type_logits = out[...,3*pos+3*apos:].reshape(-1, self.n_element)

        return [mixture_log_lattice, alpha_lattice, beta_lattice,
                mixture_log_atom, alpha_atom, beta_atom, type_logits]
    
    def to_dist(self, probs, logits):
        lattice_dist = MixtureSameFamily(Categorical(logits=logits[0]),
                                         Beta(logits[1], logits[2]))
        atom_dist = MixtureSameFamily(Categorical(logits=logits[3]),
                                         Beta(logits[4], logits[5]))

        atomtype_dist = Categorical(logits=logits[6])
        return [lattice_dist, atom_dist, atomtype_dist]
    

class GCNLatticeAtomPolicy(torch.nn.Module):
    def __init__(self, backbone, dim=4, hidden_dim=64, n_hidden=2, output_dim=32, use_pretrain=True, 
                 pretrain_dim=128, n_element=8, device=torch.device('cuda'),
                 lattice_dist_n_dim=6, lattice_dist_n_comp=1,
                 atom_dist_n_dim=3, atom_dist_n_comp=1, min_beta=0.1, max_beta=2.0, batchsize=32):
        super().__init__()
        self.backbone = backbone
        self.prob_output = lattice_dist_n_dim*lattice_dist_n_comp*3 + atom_dist_n_dim*atom_dist_n_comp*3 + n_element
        self.n_element = n_element
        sg_dim = 64
        self.spacegroup_emb =nn.Embedding(230, sg_dim)
        self.outlayer = nn.Linear(output_dim + sg_dim, self.prob_output)
        self.device = device
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.lattice_dist_n_dim = lattice_dist_n_dim
        self.lattice_dist_n_comp = lattice_dist_n_comp
        self.atom_dist_n_dim = atom_dist_n_dim
        self.atom_dist_n_comp = atom_dist_n_comp

        self.PFs0 = nn.ParameterDict(
            {
                "log_alpha_lattice": nn.Parameter(torch.zeros(batchsize,lattice_dist_n_dim*lattice_dist_n_comp)),
                "log_alpha_atom": nn.Parameter(torch.zeros(batchsize,atom_dist_n_dim*atom_dist_n_comp)),
                "log_beta_lattice": nn.Parameter(torch.zeros(batchsize,lattice_dist_n_dim*lattice_dist_n_comp)),
                "log_beta_atom": nn.Parameter(torch.zeros(batchsize,atom_dist_n_dim*atom_dist_n_comp)),
                "logits_atom": nn.Parameter(torch.zeros(batchsize,atom_dist_n_dim*atom_dist_n_comp)),
                "logits_lattice": nn.Parameter(torch.zeros(batchsize,lattice_dist_n_dim*lattice_dist_n_comp)),
                "logits_type": nn.Parameter(torch.zeros(batchsize,n_element))
            })


    def forward(self, s, sub):
        atom, sg, lattice, fpretrain = s
        if atom != None:
            out = self.backbone(s)
            sgemb = self.spacegroup_emb(sub.type(torch.int64))
            out = torch.cat((out, sgemb), dim=1)
            out = self.outlayer(out)

            pos = self.lattice_dist_n_dim*self.lattice_dist_n_comp
            mixture_log_lattice =  out[..., :pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
            alpha_lattice = self.max_beta*torch.sigmoid(out[..., pos:2*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta
            beta_lattice = self.max_beta*torch.sigmoid(out[..., 2*pos:3*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta
            apos = self.atom_dist_n_comp*self.atom_dist_n_dim
            mixture_log_atom =  out[..., 3*pos:3*pos+apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
            alpha_atom = self.max_beta*torch.sigmoid(out[..., 3*pos+apos:3*pos+2*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp) ) + self.min_beta
            beta_atom = self.max_beta*torch.sigmoid(out[..., 3*pos+2*apos:3*pos+3*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)) + self.min_beta
            type_logits = out[...,3*pos+3*apos:].reshape(-1, self.n_element)
        else:
            mixture_log_atom = self.PFs0['logits_atom'].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
            log_alpha_atom = self.PFs0['log_alpha_atom'].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
            alpha_atom = self.max_beta*torch.sigmoid(log_alpha_atom) + self.min_beta
            log_beta_atom = self.PFs0['log_beta_atom'].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
            beta_atom  = self.max_beta*torch.sigmoid(log_beta_atom) + self.min_beta

            mixture_log_lattice = self.PFs0['logits_lattice'].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
            log_alpha_lattice = self.PFs0['log_alpha_lattice'].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
            alpha_lattice = self.max_beta*torch.sigmoid(log_alpha_lattice) + self.min_beta
            log_beta_lattice = self.PFs0['log_beta_lattice'].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
            beta_lattice = self.max_beta*torch.sigmoid(log_beta_lattice) + self.min_beta

            type_logits = self.PFs0['logits_type'].reshape(-1, self.n_element)

        return [mixture_log_lattice, alpha_lattice, beta_lattice,
                mixture_log_atom, alpha_atom, beta_atom, type_logits]
    
    def to_dist(self, probs, logits):
        lattice_dist = MixtureSameFamily(Categorical(logits=logits[0]),
                                         Beta(logits[1], logits[2]))
        atom_dist = MixtureSameFamily(Categorical(logits=logits[3]),
                                         Beta(logits[4], logits[5]))

        atomtype_dist = Categorical(logits=logits[6])
        return [lattice_dist, atom_dist, atomtype_dist]
        
        
class BWGCNLatticeAtomPolicy(torch.nn.Module):
    def __init__(self, backbone, dim=4, hidden_dim=64, n_hidden=2, output_dim=32, use_pretrain=True, 
                 pretrain_dim=128, n_element=8, device=torch.device('cuda'),
                 lattice_dist_n_dim=6, lattice_dist_n_comp=1,
                 atom_dist_n_dim=3, atom_dist_n_comp=1, min_beta=0.1, max_beta=2.0, batchsize=32):
        # n_atom=3, dim=4, hidden_dim=128, n_hidden=2, output_dim=28,use_pretrain=False,pretrain_dim=128
        super().__init__()
        self.backbone = backbone
        self.prob_output = lattice_dist_n_dim*lattice_dist_n_comp*3 + atom_dist_n_dim*atom_dist_n_comp*3 + n_element
        self.n_element = n_element
        sg_dim = 64
        self.spacegroup_emb =nn.Embedding(230, sg_dim)
        self.outlayer = nn.Linear(output_dim + sg_dim, self.prob_output)
        self.device = device
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.lattice_dist_n_dim = lattice_dist_n_dim
        self.lattice_dist_n_comp = lattice_dist_n_comp
        self.atom_dist_n_dim = atom_dist_n_dim
        self.atom_dist_n_comp = atom_dist_n_comp


    def forward(self, s, sub):
        out = self.backbone(s)
        sgemb = self.spacegroup_emb(sub.type(torch.int64))
        out = torch.cat((out, sgemb), dim=1)
        out = self.outlayer(out)

        pos = self.lattice_dist_n_dim*self.lattice_dist_n_comp
        mixture_log_lattice =  out[..., :pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
        alpha_lattice = self.max_beta*torch.sigmoid(out[..., pos:2*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta

        beta_lattice = self.max_beta*torch.sigmoid(out[..., 2*pos:3*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta
        apos = self.atom_dist_n_comp*self.atom_dist_n_dim
        mixture_log_atom =  out[..., 3*pos:3*pos+apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
        alpha_atom = self.max_beta*torch.sigmoid(out[..., 3*pos+apos:3*pos+2*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp) ) + self.min_beta
        beta_atom = self.max_beta*torch.sigmoid(out[..., 3*pos+2*apos:3*pos+3*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)) + self.min_beta
        type_logits = out[...,3*pos+3*apos:].reshape(-1, self.n_element)

        return [mixture_log_lattice, alpha_lattice, beta_lattice,
                mixture_log_atom, alpha_atom, beta_atom, type_logits]
    
    def to_dist(self, probs, logits):
        lattice_dist = MixtureSameFamily(Categorical(logits=logits[0]),
                                         Beta(logits[1], logits[2]))
        atom_dist = MixtureSameFamily(Categorical(logits=logits[3]),
                                         Beta(logits[4], logits[5]))

        atomtype_dist = Categorical(logits=logits[6])
        return [lattice_dist, atom_dist, atomtype_dist]
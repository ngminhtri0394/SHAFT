from policy.graph_backbone import GCNBackbone, DimeNetPlusPlusBackbone
from policy.backbone import MLPPolicyBackbone
from torch.distributions import MultivariateNormal, VonMises, Normal, Categorical, MixtureSameFamily,Beta,Bernoulli
import torch
import torch.nn as nn

class CrystalPolicy(torch.nn.Module):
    def __init__(self, backbone, dim=4, hidden_dim=64, n_hidden=2, output_dim=32, use_pretrain=True, prob_output=231,
                 pretrain_dim=128, n_element=8, num_sg = 230, device=torch.device('cuda'),
                 lattice_dist_n_dim=6, lattice_dist_n_comp=1,
                 atom_dist_n_dim=3, atom_dist_n_comp=1, min_beta=0.1, max_beta=2.0, batchsize=32):
        super().__init__()
        self.backbone = backbone
        self.num_sg = num_sg
        self.prob_output = 1 + self.num_sg + lattice_dist_n_dim*lattice_dist_n_comp*3 + atom_dist_n_dim*atom_dist_n_comp*3 + n_element
        self.n_element = n_element
        sg_dim = 64
        self.spacegroup_emb =nn.Embedding(230, sg_dim)
        self.outlayer = nn.Linear(output_dim, self.prob_output)
        self.device = device
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.lattice_dist_n_dim = lattice_dist_n_dim
        self.lattice_dist_n_comp = lattice_dist_n_comp
        self.atom_dist_n_dim = atom_dist_n_dim
        self.atom_dist_n_comp = atom_dist_n_comp

        self.PFs0 = nn.ParameterDict(
            {
                "logits": nn.Parameter(torch.zeros(batchsize,prob_output-1)),
                "logits_exit": nn.Parameter(torch.zeros(batchsize)),
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
            out = self.outlayer(out)

            out_stop = out[...,0]
            out_sg = out[...,1:(1 + self.num_sg)]

            pos =  self.lattice_dist_n_dim*self.lattice_dist_n_comp
            offset = 1 + self.num_sg 
            mixture_log_lattice =  out[..., offset:offset + pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
            alpha_lattice = self.max_beta*torch.sigmoid(out[..., offset + pos: offset + 2*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta
            beta_lattice = self.max_beta*torch.sigmoid(out[..., offset + 2*pos: offset + 3*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta
            apos = self.atom_dist_n_comp*self.atom_dist_n_dim
            mixture_log_atom =  out[..., offset + 3*pos:offset + 3*pos+apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
            alpha_atom = self.max_beta*torch.sigmoid(out[..., offset + 3*pos+apos:offset + 3*pos+2*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp) ) + self.min_beta
            beta_atom = self.max_beta*torch.sigmoid(out[..., offset + 3*pos+2*apos:offset + 3*pos+3*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)) + self.min_beta
            type_logits = out[...,offset + 3*pos+3*apos:].reshape(-1, self.n_element)
        else:
            out_sg = self.PFs0["logits"]
            out_stop = self.PFs0["logits_exit"]

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

        return [out_stop, out_sg, mixture_log_lattice, alpha_lattice, beta_lattice,
                mixture_log_atom, alpha_atom, beta_atom, type_logits]
    
    def to_dist(self, probs, logits):
        sg_dist = Categorical(logits=logits[1])

        exit_dist = Bernoulli(logits=logits[0])

        lattice_dist = MixtureSameFamily(Categorical(logits=logits[2]),
                                         Beta(logits[3], logits[4]))
        atom_dist = MixtureSameFamily(Categorical(logits=logits[5]),
                                         Beta(logits[6], logits[7]))

        atomtype_dist = Categorical(logits=logits[8])
        return [exit_dist, sg_dist, lattice_dist, atom_dist, atomtype_dist]
        
        
class BWCrystalPolicy(torch.nn.Module):
    def __init__(self, backbone, dim=4, hidden_dim=64, n_hidden=2, output_dim=32, use_pretrain=True, prob_output=231,
                 pretrain_dim=128, n_element=8, num_sg = 230, device=torch.device('cuda'),
                 lattice_dist_n_dim=6, lattice_dist_n_comp=1,
                 atom_dist_n_dim=3, atom_dist_n_comp=1, min_beta=0.1, max_beta=2.0, batchsize=32):
        super().__init__()
        self.backbone = backbone
        self.num_sg = num_sg
        self.prob_output = 1 + self.num_sg + lattice_dist_n_dim*lattice_dist_n_comp*3 + atom_dist_n_dim*atom_dist_n_comp*3 + n_element
        self.n_element = n_element
        sg_dim = 64
        self.spacegroup_emb =nn.Embedding(230, sg_dim)
        self.outlayer = nn.Linear(output_dim, self.prob_output)
        self.device = device
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.lattice_dist_n_dim = lattice_dist_n_dim
        self.lattice_dist_n_comp = lattice_dist_n_comp
        self.atom_dist_n_dim = atom_dist_n_dim
        self.atom_dist_n_comp = atom_dist_n_comp

    def forward(self, s, sub):
        out = self.backbone(s)
        out = self.outlayer(out)

        out_stop = out[...,0]
        out_sg = out[...,1:(1 + self.num_sg)]

        pos = self.lattice_dist_n_dim*self.lattice_dist_n_comp
        offset = 1 + self.num_sg 
        mixture_log_lattice =  out[..., offset:offset + pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) 
        alpha_lattice = self.max_beta*torch.sigmoid(out[..., offset + pos:offset + 2*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta
        beta_lattice = self.max_beta*torch.sigmoid(out[..., offset + 2*pos:offset + 3*pos].reshape(-1, self.lattice_dist_n_dim, self.lattice_dist_n_comp) ) + self.min_beta
        apos = self.atom_dist_n_comp*self.atom_dist_n_dim
        mixture_log_atom =  out[..., offset + 3*pos:offset + 3*pos+apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)
        alpha_atom = self.max_beta*torch.sigmoid(out[..., offset + 3*pos+apos:offset + 3*pos+2*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp) ) + self.min_beta
        beta_atom = self.max_beta*torch.sigmoid(out[..., offset + 3*pos+2*apos:offset + 3*pos+3*apos].reshape(-1, self.atom_dist_n_dim, self.atom_dist_n_comp)) + self.min_beta
        type_logits = out[...,offset + 3*pos+3*apos:].reshape(-1, self.n_element)
    
        return [out_stop, out_sg, mixture_log_lattice, alpha_lattice, beta_lattice,
                mixture_log_atom, alpha_atom, beta_atom, type_logits]
    
    def to_dist(self, probs, logits):
        sg_dist = Categorical(logits=logits[1])

        exit_dist = Bernoulli(logits=logits[0])

        lattice_dist = MixtureSameFamily(Categorical(logits=logits[2]),
                                         Beta(logits[3], logits[4]))
        atom_dist = MixtureSameFamily(Categorical(logits=logits[5]),
                                         Beta(logits[6], logits[7]))

        atomtype_dist = Categorical(logits=logits[8])
        return [exit_dist, sg_dist, lattice_dist, atom_dist, atomtype_dist]
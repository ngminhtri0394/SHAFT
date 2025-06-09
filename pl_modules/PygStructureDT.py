from pymatgen.io.jarvis import JarvisAtomsAdaptor
# from matformer.models.pyg_att import Matformer
import torch
from pl_modules.graphs import PygGraph
from re import X
import numpy as np
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.data.batch import Batch
import itertools
from torch.utils.data import DataLoader

try:
    import torch
    from tqdm import tqdm
except Exception as exp:
    print("torch/tqdm is not installed.", exp)
    pass


def pyg_compute_bond_cosines(lg):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine

def pyg_compute_bond_angle(lg):
    """Compute bond angle from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    a = (r1 * r2).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(r1, r2).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return angle

def prepare_pyg_batch(
    batch: Tuple[Data, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    batch = (
        g.to(device),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


def prepare_pyg_line_graph_batch(
    batch: Tuple[Tuple[Data, Data, torch.Tensor], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, lattice, t = batch
    batch = (
        (
            g.to(device),
            lg.to(device),
            lattice.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


class PygStandardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: Data):
        """Apply standardization to atom_features."""
        h = g.x
        g.x = (h - self.mean) / self.std
        return g
    
def atoms_to_graph(atoms,include_coor=True):
    """Convert structure dict to DGLGraph."""
    atoms = JarvisAtomsAdaptor.get_atoms(atoms)
    return PygGraph.atom_dgl_multigraph(
                    atoms,
                    neighbor_strategy='k-nearest',
                    cutoff=8.0,
                    atom_features="atomic_number",
                    max_neighbors=12,
                    compute_line_graph=False,
                    use_canonize=True,
                    use_lattice=True,
                    use_angle=False,
                    include_coor=include_coor
                )

class PygStructureDT(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        graphs: Sequence[Data],
        atom_features="atomic_number",
        transform=None,
        line_graph=False,
        neighbor_strategy="",
        lineControl=True,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.graphs = graphs
        self.line_graph = line_graph
        self.labels = torch.tensor(np.zeros(len(self.graphs))).type(
            torch.get_default_dtype()
        )

        # self.ids = self.df[id_tag]
        # self.atoms = self.df['atoms']
        # print("mean %f std %f"%(self.labels.mean(), self.labels.std()))

        self.transform = transform

        features = self._get_attribute_lookup(atom_features)

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for g in graphs:
            z = g.x
            g.atomic_number = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.x.size(0) == 1:
                f = f.unsqueeze(0)
            g.x = f

        self.prepare_batch = prepare_pyg_batch
        self.graphs = []
        for g in graphs:
            g.edge_attr = g.edge_attr.float()
            self.graphs.append(g)
        self.line_graphs = self.graphs

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        # if self.line_graph:
        #     return g, self.line_graphs[idx], label, label

        return g, label

    def setup_standardizer(self, ids):
        """Atom-wise feature standardization transform."""
        x = torch.cat(
            [
                g.x
                for idx, g in enumerate(self.graphs)
                if idx in ids
            ]
        )
        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = PygStandardize(
            self.atom_feature_mean, self.atom_feature_std
        )

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.tensor(labels)


class PygStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        graphs: Sequence[Data],
        atom_features="atomic_number",
        transform=None,
        line_graph=False,
        neighbor_strategy="",
        lineControl=True,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.graphs = graphs
        self.line_graph = line_graph
        self.labels = torch.tensor(np.zeros(len(self.graphs))).type(
            torch.get_default_dtype()
        )


        self.transform = transform

        self.prepare_batch = prepare_pyg_batch
        # self.graphs = []
        # for g in tqdm(graphs):
        #     g.edge_attr = g.edge_attr.float()
        #     self.graphs.append(g)
        # self.line_graphs = self.graphs

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        # if self.line_graph:
        #     return g, self.line_graphs[idx], label, label

        return g, label

    def setup_standardizer(self, ids):
        """Atom-wise feature standardization transform."""
        x = torch.cat(
            [
                g.x
                for idx, g in enumerate(self.graphs)
                if idx in ids
            ]
        )
        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = PygStandardize(
            self.atom_feature_mean, self.atom_feature_std
        )

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.tensor(labels)

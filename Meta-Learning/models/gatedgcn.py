import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch.distributions import Normal
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_scatter import scatter, scatter_add
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.utils import to_dense_batch

import copy

def mdn_loss_fn(pi, sigma, mu, y, eps1=1e-10, eps2=1e-10):
    normal = Normal(mu, sigma)
    #loss = th.exp(normal.log_prob(y.expand_as(normal.loc)))
    #loss = th.sum(loss * pi, dim=1)
    #loss = -th.log(loss)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    #loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)
    prob = (torch.log(pi + eps1) + loglik).exp().sum(1)
    loss = -torch.log(prob + eps2)
    return loss, prob

def glorot_orthogonal(tensor, scale):
	"""Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
	if tensor is not None:
		torch.nn.init.orthogonal_(tensor.data)
		scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
		tensor.data *= scale.sqrt()
            
class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax,
                              PE=pe_LapPE)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = F.relu(x)
        e = F.relu(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


@register_layer('gatedgcnconv')
class GatedGCNGraphGymLayer(nn.Module):
    """GatedGCN layer.
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GatedGCNLayer(in_dim=layer_config.dim_in,
                                   out_dim=layer_config.dim_out,
                                   dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                   residual=False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                   **kwargs)

    def forward(self, batch):
        return self.model(batch)

class GatedGCN(nn.Module):
	"""A graph transformer
	"""
	def __init__(
			self,
			in_channels, 
			edge_features=10,
			num_hidden_channels=128,
			dropout_rate=0.1,
			num_layers=4,
			residual=True,
			equivstable_pe=False,
			**kwargs
			):
		super(GatedGCN, self).__init__()
		
		# Initialize model parameters
		self.residual = residual
		self.dropout_rate = dropout_rate
		self.num_layers = num_layers
		
		self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
		self.edge_encoder = nn.Linear(edge_features, num_hidden_channels) 
		
		gt_block_modules = [GatedGCNLayer(
							num_hidden_channels,
							num_hidden_channels,
							dropout_rate, 
							residual,
							equivstable_pe=equivstable_pe) for _ in range(num_layers)]
		
		self.gt_block = nn.ModuleList(gt_block_modules)
	
	def forward(self, data):		
		data.x = self.node_encoder(data.x)
		data.edge_attr = self.edge_encoder(data.edge_attr)
			
		# Apply a given number of intermediate geometric attention layers to the node and edge features given
		for gt_layer in self.gt_block:
			data = gt_layer(data)
		
		# Apply final layer to update node representations by merging current node and edge representations
		#data.x = node_feats
		#data.edge_attr = edge_feats
		#return node_feats
		return data

#==============================================
class GenScore_GGCN(nn.Module):
	def __init__(self):
		super(GenScore_GGCN, self).__init__()
		self.in_channels=128
		self.hidden_dim=128
		self.n_gaussians=10
		self.dropout_rate=0.15
		self.dist_threshold=7.0
		self.ligand_model = GatedGCN(in_channels=41, 
                                edge_features=10, 
                                num_hidden_channels=self.in_channels, 
                                residual=True,
                                dropout_rate=self.dropout_rate,
                                equivstable_pe=False,
                                num_layers=6
                            )
		self.target_model = GatedGCN(in_channels=41, 
                                edge_features=5, 
                                num_hidden_channels=self.in_channels, 
                                residual=True,
                                dropout_rate=self.dropout_rate,
                                equivstable_pe=False,
                                num_layers=6
                            )
		self.MLP = nn.Sequential(nn.Linear(self.in_channels*2, self.hidden_dim), 
								nn.BatchNorm1d(self.hidden_dim), 
								nn.ELU(), 
								nn.Dropout(p=self.dropout_rate)) 
		
		self.z_pi = nn.Linear(self.hidden_dim, self.n_gaussians)
		self.z_sigma = nn.Linear(self.hidden_dim, self.n_gaussians)
		self.z_mu = nn.Linear(self.hidden_dim, self.n_gaussians)
		self.atom_types = nn.Linear(self.in_channels, 17)
		self.bond_types = nn.Linear(self.in_channels*2, 4)
		
		#self.dist_threshold = self.dist_threshold	
		self.device = 'cpu'
        
		self.mdn_weight = 1.0
		self.affi_weight = 0.0
		self.aux_weight = 0.001
            
	def net_forward(self, data_ligand, data_target):
		
		data_ligand = data_ligand.to(self.device)
		data_target = data_target.to(self.device)

		data_ligand = copy.copy(data_ligand)
		data_target = copy.copy(data_target)

		h_l = self.ligand_model(data_ligand)
		h_t = self.target_model(data_target)
		
		h_l_x, l_mask = to_dense_batch(h_l.x, h_l.batch, fill_value=0)
		h_t_x, t_mask = to_dense_batch(h_t.x, h_t.batch, fill_value=0)
		h_l_pos, _ = to_dense_batch(h_l.pos, h_l.batch, fill_value=0)
		h_t_pos, _ = to_dense_batch(h_t.pos, h_t.batch, fill_value=0)
		
		#assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
		(B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
		self.B = B
		self.N_l = N_l
		self.N_t = N_t
				
		# Combine and mask
		h_l_x = h_l_x.unsqueeze(-2)
		h_l_x = h_l_x.repeat(1, 1, N_t, 1) # [B, N_l, N_t, C_out]
		
		h_t_x = h_t_x.unsqueeze(-3)
		h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]
		
		C = torch.cat((h_l_x, h_t_x), -1)
		self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
		self.C = C = C[C_mask]
		C = self.MLP(C)
		
		# Get batch indexes for ligand-target combined features
		C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
		C_batch = C_batch.to(self.device)
		C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]
		
 		# Outputs
		pi = F.softmax(self.z_pi(C), -1)
		sigma = F.elu(self.z_sigma(C))+1.1
		mu = F.elu(self.z_mu(C))+1
		atom_types = self.atom_types(h_l.x)
		bond_types = self.bond_types(torch.cat([h_l.x[h_l.edge_index[0]], h_l.x[h_l.edge_index[1]]], axis=1))      
		
		
		dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos.view(B,-1,3))[C_mask]
		return pi, sigma, mu, dist.unsqueeze(1).detach(), atom_types, bond_types, C_batch

	def forward(self,task,dist_threshold=None):
		if dist_threshold is None:
			dist_threshold = self.dist_threshold
		pdb_ids = task["pdb_ids"];ligs = task["ligs"];prots = task["prots"];labels = task["labels"]
		# Use deepcopy to prevent inplace modification issues with autograd
		#ligs = copy.deepcopy(ligs);prots = copy.deepcopy(prots);labels = copy.deepcopy(labels)
		ligs = ligs.to(self.device);prots = prots.to(self.device);labels = labels.to(self.device)
		atom_labels = torch.argmax(ligs.x[:,:17],dim=1,keepdim=False)
		bond_labels = torch.argmax(ligs.edge_attr[:,:4],dim=1,keepdim=False)
		pi, sigma, mu, dist, atom_types, bond_types, batch = self.net_forward(ligs,prots)
		mdn, prob = mdn_loss_fn(pi, sigma, mu, dist)
		mdn = mdn[torch.where(dist <= dist_threshold)[0]]
		mdn = mdn.mean()
		batch = batch.to(self.device)
		y = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0))
		labels = labels.float().type_as(y).to(self.device)
		# mainly for scoring task
		try:
			affi = torch.corrcoef(torch.stack([y, labels]))[1, 0]
		except:
			affi = torch.tensor(0.0,requires_grad=True)
		atom = F.cross_entropy(atom_types, atom_labels)
		bond = F.cross_entropy(bond_types, bond_labels)
		loss = mdn * self.mdn_weight + affi * self.affi_weight + atom * self.aux_weight + bond * self.aux_weight
		torch.cuda.empty_cache()
		return (
            loss,
            torch.tensor(mdn.detach().item()),
            torch.tensor(affi.detach().item()),
            torch.tensor(atom.detach().item()),
            torch.tensor(bond.detach().item()),
            y.detach(),
            batch,
        )

	def compute_euclidean_distances_matrix(self, X, Y):
		# Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
		# (X-Y)^2 = X^2 + Y^2 -2XY
		X = X.double()
		Y = Y.double()
		
		dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)	
		return torch.nan_to_num((dists**0.5).view(self.B, self.N_l,-1,24),10000).min(axis=-1)[0]
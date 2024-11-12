from typing import Sequence, Optional

from deeplay import DeeplayModule

from deeplay.components.mlp import MultiLayerPerceptron

import torch
import torch.nn as nn

class MinCutPooling(DeeplayModule):
    """
    MinCut graph pooling as described in 'Spectral Clustering with Graph Neural Networks for Graph Pooling'.

    Parameters
    ----------
    num_clusters: int
        The number of clusters to which each graph is pooled.
    hidden_features: Sequence[int]
        The number of hidden features for the Multi-Layer Perceptron (MLP) used for selecting clusters for the pooling.
    reduce_self_connection: Optional[bool]
        Whether to reduce self-connections in the adjacency matrix. Defaults to True.
    threshold: Optional[float]
        A threshold value to apply to the adjacency matrix to binarize the edges. If None, no threshold is applied.

    Evaluation
    ----------
        >>> MinCut = dl.components.gnn.pooling.MinCutPooling(hidden_features = [8], num_clusters = 5, reduce_self_connection = True, threshold = 0.25).build()
        >>> inp = {}
        >>> inp["x"] = torch.randn(10, 16)
        >>> inp['batch'] = torch.zeros(10, dtype=int)
        >>> inp["edge_index"] = torch.randint(0, 10, (2, 20))
        >>> output = MinCut(inp)

    """
    
    num_clusters: int
    hidden_features: Sequence[int]
    reduce_self_connection: Optional[bool]
    threshold: Optional[float]

    def __init__(
            self,
            num_clusters: int,
            hidden_features: Sequence[int],
            reduce_self_connection: Optional[bool] = True,
            threshold: Optional[float] = None,
            ):
        super().__init__()

        self.num_clusters = num_clusters
        self.reduce_self_connection = reduce_self_connection
        self.threshold = threshold
                
        class ClusterMatrixForBatch(DeeplayModule):
            def forward(self, S, B):

                unique_graphs = torch.unique(B)
                num_graphs = len(unique_graphs)

                S_ = torch.zeros(S.shape[0] * S.shape[1] * num_graphs)

                row_indices = torch.arange(S.shape[0]).repeat_interleave(S.shape[1])
                col_indices = B.repeat_interleave(S.shape[1]) * S.shape[1] + torch.arange(S.shape[1]).repeat(S.shape[0])

                S_[row_indices * (S.shape[1] * num_graphs) + col_indices] = S.view(-1)

                B_ = torch.arange(num_graphs).repeat_interleave(S.shape[1])

                return  S_.reshape([S.shape[0], -1]), B_
        
        class Reduce(DeeplayModule):
            def forward(self, x, s):
                return torch.matmul(s.transpose(-2,-1), x) 
            
        class Connect(DeeplayModule):
            def forward(self, A, s):
                if A.is_sparse:
                    return torch.spmm(s.transpose(-2,-1), torch.spmm(A, s))
                elif (not A.is_sparse) & (A.size(0) == 2):
                    A = torch.sparse_coo_tensor(
                        A,
                        torch.ones(A.size(1)),
                        (s.size(0),) * 2,
                        device=A.device,
                    )
                    return torch.spmm(s.transpose(-2,-1), torch.spmm(A, s))
                elif (not A.is_sparse) & len({A.size(0), A.size(1), s.size(0)}) == 1:
                    return s.transpose(-2,-1) @ A.type(s.dtype) @ s
                else:
                    raise ValueError(
                        "Unsupported adjacency matrix format.",
                        "Ensure it is a pytorch sparse tensor, an edge index tensor, or a square dense tensor.",
                        "Consider updating the propagate layer to handle alternative formats.",
                    )     

        class ReduceSelfConnection(DeeplayModule):
            def __init__(
                    self,
                    eps: Optional[float] = 1e-15,
                ):
                super().__init__()
                self.eps = eps
            
            def forward(self, A):        
                ind = torch.arange(A.shape[0])
                A[ind, ind] = 0                         
                d = torch.einsum('jk->j', A)            
                d = torch.sqrt(d)[None] #+ self.eps      
                d = torch.where(torch.isinf(d), torch.tensor(0.0, device=d.device), d)      # Replace infinities in `d` with 0
                A = (A / d) / d.transpose(-2,-1)      
                return A
            
        class MinCutLoss(DeeplayModule):
            def forward(self, A, S):
                n_nodes = S.size(0)            # number of nodes
                n_clusters = S.size(1)         # number of clusters in total (= number of clusters per graph * num graphs)

                if A.is_sparse:
                    degree = torch.sum(A, dim=0)
                elif (not A.is_sparse) & (A.size(0) == 2):
                    A = torch.sparse_coo_tensor(
                        A,
                        torch.ones(A.size(1)),
                        (n_nodes,) * 2,      
                        device=A.device,
                    )
                    degree = torch.sum(A, dim=0)
                elif (not A.is_sparse) & len({A.size(0), A.size(1)}) == 1:
                    degree = torch.sum(A, dim=0)
                else:
                    raise ValueError(
                        "Unsupported adjacency matrix format.",
                        "Ensure it is a pytorch sparse tensor, an edge index tensor, or a square dense tensor.",
                        "Consider updating the propagate layer to handle alternative formats.",
                    )   

                D = torch.eye(n_nodes) * degree

                denominator_trace = torch.trace(torch.matmul(S.transpose(-2, -1), torch.matmul(D, S)))
                denominator_trace[denominator_trace == float('inf')] = 0                # Replace infinities in `D` with 0

                # cut loss:
                # L_cut = - torch.trace(torch.matmul(S.transpose(-2,-1), torch.matmul(A, S))) / (torch.trace(torch.matmul(S.transpose(-2,-1), torch.matmul(D, S))))
                L_cut = - torch.trace(torch.matmul(S.transpose(-2,-1), torch.matmul(A, S))) / denominator_trace

                # orthogonality loss:
                L_ortho = torch.linalg.norm(
                    (torch.matmul(S.transpose(-2,-1), S) / torch.linalg.norm(torch.matmul(S.transpose(-2,-1), S), ord = 'fro'))
                    - (torch.eye(n_clusters) / torch.sqrt(torch.tensor(n_clusters))),
                    ord = 'fro')


                return L_cut, L_ortho

        class ApplyThreshold(DeeplayModule):
            def __init__(self, threshold: float):
                super().__init__()
                self.threshold = threshold

            def forward(self, A):
                return torch.where(A >= threshold, 1.0, 0.0)
            

        class SparseEdgeIndex(DeeplayModule):
            """ output edge index as a sparse tensor """
            def forward(self, A):
                if A.is_sparse:
                    edge_index = A
                    return edge_index
                else:
                    edge_index = A.to_sparse()  
                    return edge_index                


        # select: S = MLP(X)
        self.select = MultiLayerPerceptron(
            in_features=None,
            hidden_features=hidden_features,
            out_features=num_clusters,
            out_activation=nn.Softmax(dim=1))
        self.select.set_input_map("x")
        self.select.set_output_map('s')

        # make S compatible with batches:
        self.batch_compatible = ClusterMatrixForBatch()
        self.batch_compatible.set_input_map("s", "batch")
        self.batch_compatible.set_output_map("s", "batch")

        # mincut loss            
        self.mincut_loss = MinCutLoss()
        self.mincut_loss.set_input_map('edge_index', 's')
        self.mincut_loss.set_output_map('L_cut', 'L_ortho')
        
        # reduce: X' = S^T * X
        self.reduce = Reduce()
        self.reduce.set_input_map("x", 's')
        self.reduce.set_output_map("x")
        
        # connect: A' = S^T * A * S
        self.connect = Connect()
        self.connect.set_input_map('edge_index', 's')
        self.connect.set_output_map("edge_index")

        # reduce self connection
        self.red_self_con = None
        if reduce_self_connection:
            self.red_self_con = ReduceSelfConnection(self.num_clusters)
            self.red_self_con.set_input_map('edge_index')
            self.red_self_con.set_output_map('edge_index')

        # apply threshold to A
        self.apply_threshold = None
        if threshold is not None:
            self.apply_threshold = ApplyThreshold(self.threshold)
            self.apply_threshold.set_input_map('edge_index')
            self.apply_threshold.set_output_map('edge_index')

        # # make A sparse
        self.sparse = SparseEdgeIndex()
        self.sparse.set_input_map('edge_index')
        self.sparse.set_output_map('edge_index')

  
    def forward(self, x):
        x = self.select(x)
        x = self.batch_compatible(x)
        x = self.mincut_loss(x)
        x = self.reduce(x)
        x = self.connect(x)

        if self.red_self_con is not None:
            x = self.red_self_con(x)

        if self.apply_threshold is not None:
            x = self.apply_threshold(x)

        x = self.sparse(x)

        return x
    
    
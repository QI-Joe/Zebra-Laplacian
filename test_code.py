from utils.data_processing import get_data_TPPR, Data
from typing import List, Optional, Union, Any
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.utils import to_torch_coo_tensor, to_torch_sparse_tensor
from modules.memory import EfficentMemory
from torch_geometric.utils import get_laplacian

def print_data_info(datalist: List[Union[Data, Any]], snapshot: int, task: str) -> None:
    for idx, data in enumerate(datalist):
        full_data, train_data, val_data, test_data, node_num, node_edges = data
        print(f"\n\ntotal number of nodes: {node_num}, total number of edges: {node_edges} in task {task}")
        print("full_data edge shape {}, node shape {}".format(full_data.sources.shape, full_data.node_feat.shape))
        
        train_node_label_match, val_node_label_match, test_node_label_match = train_data.robustness_match_tuple, val_data.robustness_match_tuple, test_data.robustness_match_tuple
        t_vector, t1_vector = np.vectorize(full_data.hash_table.get), np.vectorize(test_data.hash_table.get)
        
        """
        Unnecessary convertion for refreshed tmeporal node to original node
        """
        # t_train_node, t_val_node, t1_node = t_vector(train_node_label_match[0]), t_vector(val_node_label_match[0]), t1_vector(test_node_label_match[0])
        
        print("The snapshot number is {}".format(idx))
        print(train_node_label_match[0].shape, val_node_label_match[0].shape, test_node_label_match[0].shape)
        print(f"Train edge number: {train_data.sources.shape[0]}")
        print(f"Validation edge number: {val_data.sources.shape[0]}")
        print(f"Test edge number: {test_data.sources.shape[0]}")
        if idx == snapshot-3:
            print("The last snapshot")
            print(f"Train node number: {train_data.num_nodes}")
            print(f"Validation node number: {val_data.num_nodes}")
            print(f"Test node number: {test_data.num_nodes}")

def compute_laplacian(adj_matrix, device, k=10):
    """
    Compute the normalized Laplacian and extract the top-k eigenvectors.
    adj_matrix: torch.Tensor of shape [N, N]
    """
    # Compute the degree matrix
    degree = torch.sum(adj_matrix, dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-8))
    
    # Normalized Laplacian: L = I - D^-0.5 * A * D^-0.5
    identity = torch.eye(adj_matrix.size(0)).to(device)
    L = identity - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    
    # Compute eigenvalues and eigenvectors, assume symmetry so use eigh
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    # Select the first k eigenvectors (smallest eigenvalues) as spectral features
    spectral_features = eigenvectors[:, :k]  # shape: [N, k]
    return spectral_features, eigenvectors, eigenvalues

if __name__ == "__main__":
    dataset = "tax51"
    snapshot = 13
    task, ratio = "fsl", 50
    datalist: Optional[list[list[Data]]|None] = None
    datalist, graph_num, graph_feat, graph_edge_num = get_data_TPPR(dataset, snapshot, dynamic=False, task=task, ratio = ratio)
    
    views = len(datalist)
    for idx in range(views):
        full_data, train_data, val_data, test_data, n_nodes, n_edges = datalist[idx]
        train_match_list, train_tranucate_label = train_data.robustness_match_tuple
        
        val_match_list, val_tranucate_label = val_data.robustness_match_tuple
        nn_val_match_list, nn_val_tranucate_label = val_data.inductive_match_tuple
        
        test_match_list, test_tranucate_label = test_data.robustness_match_tuple
        nn_test_match_list, nn_test_tranucate_label = test_data.inductive_match_tuple
        
        print(f"\n\nView {idx+1} - total number of nodes: {n_nodes}, total number of edges: {n_edges} in task {task}, full_data edges {full_data.sources.shape}, full_data nodes {full_data.node_feat.shape}"
              f"test data edges {test_data.sources.shape}, test data nodes {test_data.node_feat.shape}")
        # 1. check whether in the match list there is node with -1 label, till now it is confirmed
        # in test_match_list, there must has
        t_labels, convertor = full_data.labels, full_data.hash_table
        t1_labels, t1_convertor = test_data.labels, test_data.hash_table
        
        train_nodes = train_match_list
        train_labels = t_labels[train_nodes]
        assert (train_labels == train_tranucate_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get Train labels cannot match with pre-processed version! Mask or data has problem"
        assert train_labels.min() >= 0, f"at snapshot {idx+1}, iteration index {idx}, we get Train labels contain negative values!"
        
        val_nodes = val_match_list
        val_labels = t_labels[val_nodes]
        assert (val_labels == val_tranucate_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get Validation labels cannot match with pre-processed version! Mask or data has problem!"
        assert val_labels.min() >= 0, f"at snapshot {idx+1}, iteration index {idx}, we get Validation labels contain negative values!"
        
        test_node = test_match_list
        test_labels = t1_labels[test_node]
        assert test_labels.min(0) == -1, f"at snapshot {idx+1}, iteration index {idx}, we get Test labels shoule have -1 value"
        
        nn_val_node = nn_val_match_list
        nn_val_labels = t_labels[nn_val_node]
        assert (nn_val_labels == nn_val_tranucate_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get NN Validation labels cannot match with pre-processed version! Mask or data has problem!"
        assert nn_val_labels.min(0) >=0, f"at snapshot {idx+1}, iteration index {idx}, we get NN Validation labels shoule not have -1 value"
        
        nn_test_node = nn_test_match_list
        nn_test_labels = t1_labels[nn_test_node]
        assert (nn_test_labels == nn_test_tranucate_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get NN Test labels cannot match with pre-processed version! Mask or data has problem!"
        assert nn_test_labels.min(0) >= 0, f"at snapshot {idx+1}, iteration index {idx}, we get NN Test labels should not have -1 value"
              
              
              
    # full_data, train_data, val_data, test_data, n_nodes, n_edges = datalist[0]
    # lap_memory = EfficentMemory(snapshot=snapshot, device='cpu', combination_method='sum')
    # edge_index = torch.from_numpy(np.vstack([train_data.sources, train_data.destinations]))
    # train_nodes = np.array(sorted(np.unique(edge_index.flatten())))
    
    # edge_index_lap, edge_weight_lap = get_laplacian(edge_index, normalization='sym', num_nodes=n_nodes)
    # lap_memory.add_snapshot_memory(first_edge_idx_lap=edge_index_lap, first_edge_value_lap=edge_weight_lap, node_list=train_nodes)

    # # Pick random 10 nodes for each node in train_nodes
    # num_neighbors = 10
    # random_neighbors = np.random.choice(train_nodes, size=(len(train_nodes), num_neighbors), replace=True) # np.random.choice(trian_nodes, size=(trian_nodes.shape[0], num_neighbors), replace=True)

    # get_memory = lap_memory.get_snapshot_memory(src_nodes=train_nodes, node_list=random_neighbors)
    # print("Memory snapshot shape:", get_memory.shape)
    # print("Memory snapshot:", get_memory.sum(axis=1))
    
    
    
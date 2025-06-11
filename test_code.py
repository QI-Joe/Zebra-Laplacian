from utils.data_processing import get_data_TPPR, Data, get_Temporal_data_TPPR_Node_Justification
from typing import List, Optional, Union, Any
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.utils import to_torch_coo_tensor, to_torch_sparse_tensor
from modules.memory import EfficentMemory
from torch_geometric.utils import get_laplacian
import math
import tracemalloc

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

def test_get_data_TPPR():
    """
    Test the get_data_TPPR function to ensure it returns the correct data structure and properties.
    """
    dataset = "tax51"
    snapshot = 8
    task, ratio = "imbalance", 1.5
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
        
        print(f"\n\nView {idx+1} - total number of nodes: {n_nodes}, total number of edges: {n_edges} in task {task}, full_data edges {full_data.sources.shape}, full_data nodes {full_data.node_feat.shape}\n"
              f"test data edges {test_data.sources.shape}, test data nodes {test_data.node_feat.shape}")
        # 1. check whether in the match list there is node with -1 label, till now it is confirmed
        # in test_match_list, there must has
        t_labels, convertor = full_data.labels, full_data.hash_table
        t1_labels, t1_convertor = test_data.labels, test_data.hash_table
        
        train_nodes = train_match_list
        train_labels = t_labels[train_nodes]
        print("train data/label size", train_labels.shape, train_tranucate_label.shape)
        assert (train_labels == train_tranucate_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get Train labels cannot match with pre-processed version! Mask or data has problem"
        assert train_labels.min() >= 0, f"at snapshot {idx+1}, iteration index {idx}, we get Train labels contain negative values!"
        
        val_nodes = val_match_list
        val_labels = t_labels[val_nodes]
        print("validation data/label size", val_labels.shape, val_tranucate_label.shape)
        assert (val_labels == val_tranucate_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get Validation labels cannot match with pre-processed version! Mask or data has problem!"
        if len(val_nodes)>0:
            assert val_labels.min() >= 0, f"at snapshot {idx+1}, iteration index {idx}, we get Validation labels contain negative values!"
        
        test_node = test_match_list
        test_labels = t1_labels[test_node]
        print("test data/label size", test_labels.shape, test_tranucate_label.shape)
        assert test_labels.min() == -1, f"at snapshot {idx+1}, iteration index {idx}, we get Test labels shoule have -1 value"
        
        nn_val_node = nn_val_match_list
        nn_val_labels = t_labels[nn_val_node]
        print("nn validation data/label size", nn_val_labels.shape, nn_val_tranucate_label.shape)
        assert (nn_val_labels == nn_val_tranucate_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get NN Validation labels cannot match with pre-processed version! Mask or data has problem!"
        assert nn_val_labels.min() >=0, f"at snapshot {idx+1}, iteration index {idx}, we get NN Validation labels shoule not have -1 value"
        
        nn_test_node = nn_test_match_list
        nn_test_labels = t1_labels[nn_test_node]
        print("nn test data/label size", nn_test_labels.shape, nn_test_tranucate_label.shape)
        assert (nn_test_labels == nn_test_tranucate_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get NN Test labels cannot match with pre-processed version! Mask or data has problem!"
        assert nn_test_labels.min() >= 0, f"at snapshot {idx+1}, iteration index {idx}, we get NN Test labels should not have -1 value"
        
        def evaluation_simluate_test(data: Data):
            src_combination = np.concatenate([data.sources, data.destinations])
            val_sample_node = np.array(list(set(src_combination)))
            vector_transform = np.vectorize(data.hash_table.get)
            refreshed_node = vector_transform(val_sample_node)

            assert refreshed_node.shape[0] == data.n_unique_nodes, f"at snapshot {idx+1}, iteration index {idx}, we get refreshed node number {refreshed_node.shape[0]} does not match with test data node number {data.n_unique_nodes}"
            # assert refreshed_node.shape[0] == data.labels.shape[0], f"at snapshot {idx+1}, iteration index {idx}, we get refreshed node number {refreshed_node.shape[0]} does not match with test data label number {data.labels.shape[0]}"
            
            y_true = data.labels[refreshed_node]
            """
            shape of y_hat == y_true == refreshed_node <<< data.labels
            """
            val_match_node, val_match_label = data.robustness_match_tuple
            nn_val_match = data.inductive_match_tuple
            
            val_match_mask = np.isin(refreshed_node, val_match_node)
            assert (refreshed_node[val_match_mask] == val_match_node).all(), f"at snapshot {idx+1}, iteration index {idx}, we get refreshed node {refreshed_node} does not match with validation match node {val_match_node}"
            node_mask = val_match_mask & (y_true != -1) # add here due to test data will have -1 label
        
            if True:
                nn_val_match_node, nn_val_match_label = nn_val_match
                nn_val_label_allow2see = y_true[nn_val_match_node]
                assert (nn_val_match_label == nn_val_label_allow2see).all(), f"at snapshot {idx+1}, iteration index {idx}, we get NN Validation labels cannot match with pre-processed version! Mask or data has problem!"

        evaluation_simluate_test(val_data)
        evaluation_simluate_test(test_data)


if __name__ == "__main__":
    dataset = "tax51"
    snapshot = 8
    task, ratio = "None", 1.5
    datalist: Optional[list[list[Data]]|None] = None
    BATCH_SIZE = 20_000
    datalist, graph_num, graph_feat, graph_edge_num = get_Temporal_data_TPPR_Node_Justification(dataset, snapshot, dynamic=False, task=task, ratio = ratio)
    
    views = len(datalist)
    
    for idx in range(views):
        full_data, train_data, val_data, test_data, train_learn, nn_val_data, nn_test_data, n_nodes, n_edges = datalist[idx]
        print(f"\n\nView {idx+1} - total number of nodes: {n_nodes}, total number of edges: {n_edges} in task {task}, full_data edges {full_data.sources.shape}, full_data nodes {full_data.node_feat.shape}\n"
              f"test data edges {test_data.sources.shape}, test data nodes {test_data.node_feat.shape}")
        
        assert n_nodes == full_data.n_unique_nodes, f"at snapshot {idx+1}, iteration index {idx}, we get full_data number of nodes {full_data.n_unique_nodes} does not match with total number of nodes {n_nodes}"
        
        t_num_nodes = n_nodes 
        print(f"Train data edge shape: {train_data.sources.shape}, number of nodes: {train_data.n_unique_nodes}, in ratio of {train_data.n_unique_nodes/t_num_nodes:.4f} of full snapshot graph")
        print(f"Validation data edge shape: {val_data.sources.shape}, number of nodes: {val_data.n_unique_nodes}, in ratio of {val_data.n_unique_nodes/t_num_nodes:.4f} of full snapshot graph")
        print(f"Train data edge/full data edge ratio: {train_data.sources.shape[0]/full_data.sources.shape[0]:.4f}")
        print(f"Validation data edge/full data edge ratio: {val_data.sources.shape[0]/full_data.sources.shape[0]:.4f}")
        print(f"Training Learn Edge {train_learn.sources.shape[0]}, with ratio with full data edge {train_learn.sources.shape[0]/full_data.sources.shape[0]:.4f}, node number {train_learn.n_unique_nodes}, with full data node ratio {train_learn.n_unique_nodes/full_data.n_unique_nodes:.4f}")
        print(f"Test data edge shape: {test_data.sources.shape}, number of nodes: {test_data.n_unique_nodes}")
        
        train_node, train_label = train_data.robustness_match_tuple
        val_node, val_label = val_data.robustness_match_tuple
        nn_val_node, nn_val_label = val_data.robustness_match_tuple
        test_node, test_label = test_data.robustness_match_tuple
        nn_test_node, nn_test_label = nn_test_data.robustness_match_tuple
        num_instance = train_data.sources.shape[0]
        
        def print_unique_counts(name, arr):
            unique, counts = np.unique(arr, return_counts=True)
            print(f"{name} unique values: {unique}")
            print(f"{name} counts: {counts}")
            print(f"Get number of label {len(arr)}, available unique label {(len(arr)- counts[0]) if unique[0]==-1 else len(arr)}")
        
        tracemalloc.start()

        edge_index = torch.from_numpy(np.vstack((train_learn.sources, train_learn.destinations)))
        edge_index_lap, edge_weight_lap = get_laplacian(edge_index, normalization='sym', num_nodes=graph_num+1)

        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10**6:.3f} MB; Peak was {peak / 10**6:.3f} MB in total numer of nodes {graph_num+1}")

        tracemalloc.stop()
        num_batch = math.ceil(num_instance/BATCH_SIZE)
        
        for batch in range(num_batch):
            start = batch*BATCH_SIZE
            end = min((batch+1)*BATCH_SIZE, num_instance)
            batch_src = train_data.sources[start: end]
            batch_end = train_data.destinations[start:end]
            batch_time = train_data.timestamps[start:end]
            batch_train = np.concatenate([batch_src, batch_end])
            batch_train_time = np.concatenate([batch_time, batch_time])
            
            vector_map = np.vectorize(train_data.hash_table.get)
            sample_node = np.array(sorted(set(batch_train)))
            sample_node = vector_map(sample_node)
            
            train_match_list, train_tranucated_label = train_data.robustness_match_tuple
            node_allow2see_mask = np.isin(sample_node, train_match_list) & (train_data.labels[sample_node] != -1)
            
            labels = train_data.labels[sample_node][node_allow2see_mask]
            assert labels.min() >= 0, f"at snapshot {idx+1}, iteration index {idx}, we get Train labels contain negative values!"

        # print_unique_counts("Train", train_label)
        # print_unique_counts("Validation", val_label)
        # print_unique_counts("NN Validation", nn_val_label)
        # print_unique_counts("Test", test_label)
        # print_unique_counts("NN Test", nn_test_label)
        print("\n\n")
        
        assert (train_data.labels[train_node] == train_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get Train labels cannot match with pre-processed version! Mask or data has problem"
        assert (val_data.labels[val_node] == val_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get Validation labels cannot match with pre-processed version! Mask or data has problem"
        assert (nn_val_data.labels[nn_val_node] == nn_val_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get NN Validation labels cannot match with pre-processed version! Mask or data has problem"
        assert (test_data.labels[test_node] == test_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get Test labels cannot match with pre-processed version! Mask or data has problem"
        assert (nn_test_data.labels[nn_test_node] == nn_test_label).all(), f"at snapshot {idx+1}, iteration index {idx}, we get NN Test labels cannot match with pre-processed version! Mask or data has problem"
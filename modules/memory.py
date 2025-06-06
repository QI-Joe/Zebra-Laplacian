from time import time
from typing import Optional, Any
import torch
from torch import nn
from copy import deepcopy
import numpy as np
from collections import defaultdict

class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, input_dimension, node_fea: torch.Tensor, message_dimension=None, device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device
    self.combination_method = combination_method
    self.node_feature = node_fea
    self.__init_memory__()

  def __init_memory__(self):
    assert self.memory_dimension == self.node_feature.shape[1], f"node_feature {self.node_feature.shape[1]} and feature memory bank {self.memory_dimension} not equal."

    self.memory = self.node_feature.to(torch.float32).to(self.device)   # torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device)
    dim_negative = torch.zeros((1, self.node_feature.shape[1])).to(self.device)
    self.memory = torch.concat((self.memory, dim_negative), dim=0)
    self.last_update = torch.zeros(self.n_nodes).to(self.device)

    self.nodes = np.zeros(self.n_nodes,dtype=bool)
    self.messages = torch.zeros((self.n_nodes, self.message_dimension)).to(self.device)
    self.timestamps = torch.zeros((self.n_nodes)).to(self.device)

  def store_raw_messages(self, nodes, messages, timestamps):
    self.nodes[nodes]=1
    self.messages[nodes]=messages
    self.timestamps[nodes]=timestamps

  def set_device(self,device):
    self.device=device
    self.memory = self.memory.to(self.device)
    self.last_update = self.last_update.to(self.device)
    self.nodes = np.zeros(self.n_nodes,dtype=bool)
    self.messages = self.messages.to(self.device)
    self.timestamps = self.timestamps.to(self.device)

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :]

  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_memory(self):
    return self.memory.clone(), self.last_update.clone(), self.messages.clone(), self.nodes, self.timestamps.clone()

  def restore_memory(self, memory_backup):
    self.memory, self.last_update, self.messages, self.nodes, self.timestamps = memory_backup[0].clone(), memory_backup[1].clone(), memory_backup[2].clone(), memory_backup[3], memory_backup[4].clone()

  def detach_memory(self):
    self.memory.detach_()
    self.messages.detach_()

  def clear_messages(self,positives):
    self.nodes[positives]= 0
    
class EfficentMemory(nn.Module):
  def __init__(self, snapshot: int, device="cpu", combination_method='sum'):
    super(EfficentMemory, self).__init__()
    self.memory: Optional[torch.Tensor | None | Any] = None
    self.device = device
    self.combination_method = combination_method
    self.snapshot = snapshot
    self.snapshot_memory = [np.array([0])*self.snapshot]
    self.memory_position = 0
    
    
  def reset_device(self, device):
    self.device = device
    self.memory = self.memory.to(self.device)
    
  def add_snapshot_memory(self, first_edge_idx_lap: torch.Tensor, first_edge_value_lap: torch.Tensor, node_list = None):
    first_edge_idx_lap, first_edge_value_lap = first_edge_idx_lap.cpu().numpy(), first_edge_value_lap.cpu().numpy()
    self.memory = self.src2dst_tuple_adj_list(first_edge_idx_lap, first_edge_value_lap, node_list)
    self.snapshot_memory[self.memory_position] = deepcopy(self.memory)
    self.memory_position += 1
  
  def src2dst_tuple_adj_list(self, edge_idx, edge_value, node_list: Optional[np.ndarray | None])->dict:
    transpose_edge_idx: np.ndarray = edge_idx.T
    num_edges = transpose_edge_idx.shape[0]
    src2dst_adj_list = defaultdict(lambda:np.float32(1.0), {(nodes, -1): np.float32(0) for nodes in node_list})
    for idx in range(num_edges):
      src2dst_adj_list[tuple(transpose_edge_idx[idx])] = edge_value[idx]
      src2dst_adj_list[tuple(transpose_edge_idx[idx][::-1])] = edge_value[idx]  # Add reverse edge for undirected graph
    return src2dst_adj_list
  
  def adj_list_build(self, edge_idx, edge_value, node_list: int)-> dict:
    """
    :param edge_idx: np.ndarray, shape (2, num_edges)
    :param edge_value: np.ndarray, shape (num_edges,)
    """
    adj_list = {i: [] for i in node_list}
    built_edges = edge_idx.T.shape[0]
    for idx in range(built_edges):
      src, dst = edge_idx[0, idx], edge_idx[1, idx]
      value = edge_value[idx]
      adj_list[src].append((dst, value))
    return adj_list
  
  def get_snapshot_memory(self, src_nodes, node_list: np.ndarray):
    """
    :param src_nodes: np.ndarray, shape (batch_nodes,)
    :param node_list: np.ndarray, shape (batch_nodes, num_neighbors), is selected_nodes from tppr library
    """
    num_iter = len(src_nodes)
    weight_list = list()
    for i in range(num_iter):
      src = src_nodes[i]
      selected_node_list=[self.memory[(src, dst)] for dst in node_list[i]]
      weight_list.append(selected_node_list)
    return np.array(weight_list, dtype=np.float32)
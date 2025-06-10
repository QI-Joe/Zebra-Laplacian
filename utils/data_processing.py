from math import radians
import numpy as np
import random
import pandas as pd
import os
from utils.my_dataloader import data_load, Temporal_Splitting, Temporal_Dataloader
import torch
import copy
from typing import Union, Optional
from utils.robustness_injection import Imbalance, Few_Shot_Learning

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels, hash_table: dict[int, int], node_feat: np.ndarray = None):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)
        self.tbatch = None
        self.n_batch = 0
        self.node_feat = node_feat
        self.hash_table = hash_table

        self.target_node: Optional[set|None] = None
        self.robustness_match_tuple: Optional[tuple[np.ndarray, np.ndarray]] = None
  
  def setup_robustness(self, match_tuple: tuple[np.ndarray, np.ndarray], inductive_match_tuple: Optional[tuple[np.ndarray, np.ndarray]] = None):
    self.robustness_match_tuple = match_tuple
    self.inductive_match_tuple = inductive_match_tuple
  
  def set_up_features(self, node_feat, edge_feat):
    self.node_feat = node_feat
    self.edge_feat = edge_feat

  def call_for_inductive_nodes(self, val_data: 'Data', test_data: 'Data', single_graph: bool):
    validation_node: set = val_data.unique_nodes
    test_node: set = test_data.unique_nodes
    train_node = self.unique_nodes

    common_share = validation_node & test_node & train_node
    train_val_share = validation_node & train_node
    train_test_share = train_node & test_node
    val_test_share = validation_node & test_node

    expected_val = list(validation_node - (common_share | train_val_share))

    if single_graph:
      expected_test = list(test_node - (train_test_share | common_share | val_test_share))
      test_data.propogator_back(expected_test, single_graph = single_graph)
    else:
      t_times_common_data = list(train_test_share | common_share | val_test_share)
      t_times_hash_table = val_data.hash_table
      test_data.propogator_back(t_times_common_data, single_graph, t_times_hash_table)

    assert len(set(expected_val) & train_node) == 0, "train_node data is exposed to validation set"
    if single_graph:
      assert len(set(expected_test) & train_node & set(expected_val)) == 0, "train node and val data has interacted with test data"

    val_data.propogator_back(expected_val, single_graph=True)
    
    self.propogator_back(train_node, single_graph=True)
    return 

  def edge_mask(self, data: 'Data', test_element: set):
    test_element = sorted(test_element)
    src_mask = ~np.isin(data.sources, test_element)
    dst_mask = ~np.isin(data.destinations, test_element)
    return src_mask & dst_mask

  def cover_the_edges(self, val_data: 'Data', test_data: 'Data', single_graph: bool = True):
    """
    delete both edges and nodes appeared in train_data to make pure inductive val_data \n
    also, delete both edges and nodes appeared in train_data and val_data to make pure inductive test_data
    """
    valid_node = val_data.unique_nodes
    test_node = test_data.unique_nodes
    train_node = self.unique_nodes

    # common_share = valid_node & test_node & train_node
    train_val_share = valid_node & train_node
    train_test_share = train_node & test_node
    val_test_share = valid_node & test_node

    node_2be_removed_val = train_val_share
    node_2be_removed_test = val_test_share | train_test_share

    val_data.edge_propagate_back(self.edge_mask(val_data, node_2be_removed_val))
    test_data.edge_propagate_back(self.edge_mask(test_data, node_2be_removed_test))

    return

  def edge_propagate_back(self, edge_mask: np.ndarray):
    """
    keep the edge mask as permanent variable and modify edge \n
    maintain edges to inductive edge mask
    """
    self.inductive_edge_mask = edge_mask
    self.sources = self.sources[self.inductive_edge_mask]
    self.destinations = self.destinations[self.inductive_edge_mask]
    self.timestamps = self.timestamps[self.inductive_edge_mask]

  def propogator_back(self, node_idx: list, single_graph: bool, t_hash_table: Union[dict | None] = None):
    """
    Expected to clear the node index get establish the mask mainly for non-visible data node \n

    :Attention: meaning of node_idx is different(reversed) when single_graph is different!!!

    :param node_idx when single_graph is True -- is the node that uniquness to the given data object,
    :param node_idx when single_graph is False -- it represent node that should be removed in given node set!!!

    :return self.target_node -- whatever how node_idx and single_graph changed, it always present node to be Uniquness
    to the given Data object
    """
    batch_nodes = np.array(sorted(self.unique_nodes))
    if single_graph:
      # self.target_node_mask = np.isin(batch_nodes, sorted(node_idx))
      self.target_node = self.unique_nodes & set(node_idx)
    else:
      t_transfer_map = np.vectorize(t_hash_table.get)
      t1_transfer_map = np.vectorize(self.hash_table.get)

      seen_nodes = t_transfer_map(node_idx)
      test_seen_nodes = t1_transfer_map(batch_nodes)

      test_node = set(test_seen_nodes) - set(seen_nodes)
      reverse_test_hashtable = {v:k for k, v in self.hash_table.items()}
      t1_back_transfer = np.vectorize(reverse_test_hashtable.get)
      t_test_node = t1_back_transfer(test_node)

      # self.target_node_mask = np.isin(batch_nodes, sorted(t_test_node))
      self.target_node = self.unique_nodes - t_test_node

    return

  def sample(self,ratio):
    data_size=self.n_interactions
    sample_size=int(ratio*data_size)
    sample_inds=random.sample(range(data_size),sample_size)
    sample_inds=np.sort(sample_inds)
    sources=self.sources[sample_inds]
    destination=self.destinations[sample_inds]
    timestamps=self.timestamps[sample_inds]
    edge_idxs=self.edge_idxs[sample_inds]
    labels=self.labels[sample_inds]
    return Data(sources,destination,timestamps,edge_idxs,labels)


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []

  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]

    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0

    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
    
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)
  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

def to_TPPR_Data(graph: Temporal_Dataloader) -> Data:
    nodes = graph.x
    edge_idx = np.arange(graph.edge_index.shape[1])
    timestamp = graph.edge_attr.numpy() if isinstance(graph.edge_attr, torch.Tensor) else graph.edge_attr
    if isinstance(graph.edge_index, torch.Tensor):
        graph.edge_index = graph.edge_index.numpy()
    src, dest = graph.edge_index[0, :], graph.edge_index[1, :]
    labels = graph.y.numpy() if isinstance(graph.y, torch.Tensor) else graph.y

    hash_dataframe = copy.deepcopy(graph.my_n_id.node.loc[:, ["index", "node"]].values.T)
    hash_table: dict[int, int] = {node: idx for idx, node in zip(*hash_dataframe)}
    
    if np.any(graph.edge_attr != None):
        edge_attr = graph.edge_attr
    if np.any(graph.pos != None):
        pos = graph.pos
        pos = pos.numpy() if isinstance(pos, torch.Tensor) else pos
    else:
        pos = graph.x

    TPPR_data = Data(sources= src, destinations=dest, timestamps=timestamp, edge_idxs = edge_idx, labels=labels, hash_table=hash_table, node_feat=pos)

    return TPPR_data

def quantile_(threshold: float, timestamps: torch.Tensor) -> tuple[torch.Tensor]:
  full_length = timestamps.shape[0]
  val_idx = int(threshold*full_length)

  if not isinstance(timestamps, torch.Tensor):
     timestamps = torch.from_numpy(timestamps)
  train_mask = torch.zeros_like(timestamps, dtype=bool)
  train_mask[:val_idx] = True

  val_mask = torch.zeros_like(timestamps, dtype=bool)
  val_mask[val_idx:] = True

  return train_mask, val_mask

def quantile_static(val: float, test: float, timestamps: torch.Tensor) -> tuple[torch.Tensor]:
  full_length = timestamps.shape[0]
  val_idx = int(val*full_length)
  test_idx = int(test*full_length)

  if not isinstance(timestamps, torch.Tensor):
     timestamps = torch.from_numpy(timestamps)
  train_mask = torch.zeros_like(timestamps, dtype=bool)
  train_mask[:val_idx] = True

  val_mask = torch.zeros_like(timestamps, dtype=bool)
  val_mask[val_idx:test_idx] = True

  test_mask = torch.zeros_like(timestamps, dtype=bool)
  test_mask[test_idx:] = True

  return train_mask, val_mask, test_mask

def get_Temporal_data_TPPR_Node_Justification(dataset_name, snapshot: int, dynamic: bool, task: str, ratio: float = 0.0):
    r"""
    this function is used to convert the node features to the correct format\n
    e.g. sample node dataset is in the format of [node_id, edge_idx, timestamp, features] with correspoding\n
    shape [(n, ), (m,2), (m,), (m,d)]. be cautious on transformation method\n
    
    2025.4.5 TPPR and data_load method will not support TGB-Series data anymore
    """
    wargs = {"rb_task": task, "ratio": ratio}
    graph, idx_list = data_load(dataset_name, **wargs)
    graph_list = Temporal_Splitting(graph, dynamic=dynamic, idxloader=idx_list).temporal_splitting(snapshot=snapshot)
    graph_num_node, graph_feat, edge_number = max(graph.x), copy.deepcopy(graph.pos), graph.edge_index.shape[1]

    TPPR_list: list[list[Data]] = []
    lenth = len(graph_list) - 1 # no training for the last graph, so -1
        
    for idx in range(lenth):
      # covert Temproal_graph object to Data object
      items = graph_list[idx]
      temporal_node_num = items.x.shape[0]
      items.edge_attr = items.edge_attr 
      
      src_edge = items.edge_index[0, :]
      dst_edge = items.edge_index[1, :]
      all_nodes = items.my_n_id.node["index"].values
      flipped_nodes = items.my_n_id.node["node"].values
      items.y = np.array(items.y)

      t_labels = items.y
      full_data = to_TPPR_Data(items)
      timestamp = full_data.timestamps
      # selected_splitted_timestamps = span_time_quantile(threshold=0.80, tsp=timestamp, dataset=dataset_name)
      
      train_node, train_node_origin, train_label = all_nodes[:int(temporal_node_num*0.8)], flipped_nodes[:int(temporal_node_num*0.8)], t_labels[:int(temporal_node_num*0.8)]
      val_node, val_node_origin, val_label = all_nodes[int(temporal_node_num*0.8):], flipped_nodes[int(temporal_node_num*0.8):], t_labels[int(temporal_node_num*0.8):]
      
      train_selected_src_edge, train_selected_dst_edge = np.isin(src_edge, train_node_origin), np.isin(dst_edge, train_node_origin)
      train_mask = train_selected_src_edge & train_selected_dst_edge
      train_feature_should_be_seen = train_selected_src_edge | train_selected_dst_edge
      
      val_selected_src_edge, val_selected_dst_edge = np.isin(src_edge, val_node_origin), np.isin(dst_edge, val_node_origin)
      val_mask = val_selected_src_edge | val_selected_dst_edge
      nn_val_mask = val_selected_src_edge & val_selected_dst_edge
      
      hash_dataframe = copy.deepcopy(items.my_n_id.node.loc[:, ["index", "node"]].values.T)
      hash_table: dict[int, int] = {node: idx for idx, node in zip(*hash_dataframe)}
      
      train_data = Data(full_data.sources[train_mask], full_data.destinations[train_mask], full_data.timestamps[train_mask],\
                        full_data.edge_idxs[train_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
      train_data.setup_robustness((train_node, train_label))
        
      val_data = Data(full_data.sources[val_mask], full_data.destinations[val_mask], full_data.timestamps[val_mask],\
                        full_data.edge_idxs[val_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
      val_data.setup_robustness((val_node, val_label))
      
      train_data_edge_learn = Data(full_data.sources[train_feature_should_be_seen], full_data.destinations[train_feature_should_be_seen], \
      full_data.timestamps[train_feature_should_be_seen], full_data.edge_idxs[train_feature_should_be_seen], t_labels, hash_table=hash_table, node_feat=full_data.node_feat)
      
      nn_val_data = Data(full_data.sources[nn_val_mask], full_data.destinations[nn_val_mask], full_data.timestamps[nn_val_mask],\
                        full_data.edge_idxs[nn_val_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
      nn_val_node_original = np.array(sorted(set(full_data.sources[nn_val_mask]) | set(full_data.destinations[nn_val_mask])))
      nn_val_node = np.vectorize(nn_val_data.hash_table.get)(nn_val_node_original)
      nn_val_data.setup_robustness((nn_val_node, t_labels[nn_val_node]))
      
      if task in ["imbalance", "fsl"]:
        if task == "imbalance":
          train_ratio, val_ratio = 0.8, 0.2
          transform = Imbalance(train_ratio=train_ratio, ratio=ratio, val_ratio=val_ratio)
          items = transform(items)
          # val_node, train_node are node indices, match with node label in each row
        elif task == "fsl":
          transform = Few_Shot_Learning(fsl_num=ratio)
          items = transform(items)
      
        node_label = items.my_n_id.node["label"].values
          
        train_label, train_node = node_label[items.train_mask], all_nodes[items.train_mask]
        val_label, val_node = node_label[items.val_mask], all_nodes[items.val_mask]
        nn_val_label, nn_val_node = node_label[items.nn_val_mask], all_nodes[items.nn_val_mask]
        
        train_data = fast_Data_object_update(items.my_n_id.node, train_node, full_data)
        val_data = fast_Data_object_update(items.my_n_id.node, val_node, full_data)
        nn_val_data = fast_Data_object_update(items.my_n_id.node, nn_val_node, full_data)
        
        train_data.setup_robustness((train_node, train_label)) 
        val_data.setup_robustness((val_node, val_label))
        nn_val_data.setup_robustness((nn_val_node, nn_val_label))
            
      test: Temporal_Dataloader = graph_list[idx+1]
      test_data = to_TPPR_Data(test)
      test_node, test_label = test.my_n_id.node["index"].values, test.my_n_id.node["label"].values
      test_data.setup_robustness((test_node := test.my_n_id.node["index"].values, test_label))
      
      # test_data.setup_robustness((test_node, test_label))
      nn_test_node_original = np.array(sorted(set(test.my_n_id.node["node"].values) - set(flipped_nodes)))
      nn_test_node = np.vectorize(test_data.hash_table.get)(nn_test_node_original)
      nn_test_src, nn_test_dst = np.isin(test_data.sources, nn_test_node_original), np.isin(test_data.destinations, nn_test_node_original)
      nn_test_mask = nn_test_src | nn_test_dst
      nn_test_data = Data(test_data.sources[nn_test_mask], test_data.destinations[nn_test_mask], test_data.timestamps[nn_test_mask],\
                          test_data.edge_idxs[nn_test_mask], test_label, hash_table = test_data.hash_table, node_feat=test_data.node_feat)
      
      nn_test_label = test_label[nn_test_node]
      nn_test_data.setup_robustness((nn_test_node, nn_test_label))
      
      if task in ["imbalance", "fsl"]:
        test_transform = transform.test_processing(test)
        nn_test_match_list = (test_node[test_transform.nn_test_mask], test_label[test_transform.nn_test_mask])
        nn_test_data = fast_Data_object_update(test.my_n_id.node, nn_test_match_list[0], test_data)
        nn_test_data.setup_robustness(nn_test_match_list)

      node_num = items.num_nodes
      node_edges = items.num_edges

      TPPR_list.append([full_data, train_data, val_data, test_data, train_data_edge_learn, nn_val_data, nn_test_data, node_num, node_edges])
      
    return TPPR_list, graph_num_node, graph_feat, edge_number
      
def span_time_quantile(threshold: float, tsp: np.ndarray, dataset: str):
    val_time = np.quantile(tsp, threshold)
    
    if dataset in ["dblp", "tmall"]:
        spans, span_freq = np.unique(tsp, return_counts=True)
        if val_time == spans[-1]: val_time = spans[int(spans.shape[0]*threshold)]
    return val_time

def fast_Data_object_update(match_table: pd.DataFrame, nodes: np.ndarray, full_data: Data) -> Data:
  """
  Updates a Data object by filtering its edges to include only those between specified nodes.
  Args:
    match_table (pd.DataFrame): A DataFrame containing at least three columns: ["index", "node", "label"]. Used to map new nodes to original nodes.
    nodes (np.ndarray): An array of indices representing the new nodes to be included.
    full_data (Data): The original Data object containing sources, destinations, timestamps, edge indices, labels, and optional attributes.
  Returns:
    Data: A new Data object containing only the edges where both source and destination nodes are among the specified nodes. Other attributes (labels, hash_table, node_feat) are preserved from the original Data object.
  """
  
  nptable = match_table.values # ["index", "node", "label"]
  original_node = nptable[nodes, 1] # nodes consisted by new node, use second column to match original node
  nn_src, nn_dst = np.isin(full_data.sources, original_node), np.isin(full_data.destinations, original_node)
  nn_mask = nn_src & nn_dst
  return Data(full_data.sources[nn_mask], full_data.destinations[nn_mask], full_data.timestamps[nn_mask],\
              full_data.edge_idxs[nn_mask], full_data.labels, hash_table=full_data.hash_table, node_feat=full_data.node_feat)
  

def get_data_TPPR(dataset_name, snapshot: int, dynamic: bool, task: str, ratio: float = 0.0):
    r"""
    this function is used to convert the node features to the correct format\n
    e.g. sample node dataset is in the format of [node_id, edge_idx, timestamp, features] with correspoding\n
    shape [(n, ), (m,2), (m,), (m,d)]. be cautious on transformation method\n
    
    2025.4.5 TPPR and data_load method will not support TGB-Series data anymore
    """
    wargs = {"rb_task": task, "ratio": ratio}
    graph, idx_list = data_load(dataset_name, **wargs)
    if snapshot<=3: 
        graph.edge_attr = np.arange(graph.edge_index.shape[1])
        graph_list = [Temporal_Dataloader(nodes=graph.x, edge_index=graph.edge_index, \
                                          edge_attr=graph.edge_attr, y=graph.y, pos=graph.pos)]
    else:
        graph_list = Temporal_Splitting(graph, dynamic=dynamic, idxloader=idx_list).temporal_splitting(snapshot=snapshot)
    graph_num_node, graph_feat, edge_number = max(graph.x), copy.deepcopy(graph.pos), graph.edge_index.shape[1]

    TPPR_list: list[list[Data]] = []
    lenth = len(graph_list) - 1 # no training for the last graph, so -1
    single_graph = False
    if lenth < 2: 
        single_graph = True

    for idxs in range(0, lenth):
        # covert Temproal_graph object to Data object
        items = graph_list[idxs]
        temporal_node_num = items.x.shape[0]
        items.edge_attr = items.edge_attr 
        all_nodes = items.my_n_id.node["index"].values
        items.y = np.array(items.y)

        t_labels = items.y
        full_data = to_TPPR_Data(items)
        timestamp = full_data.timestamps
        train_mask, val_mask = quantile_(threshold=0.80, timestamps=timestamp)
        train_node_stop_indcies, val_node_stop_indcies = all_nodes[int(temporal_node_num*0.8)], temporal_node_num

        if idxs == lenth-1 or single_graph:
          train_mask, val_mask, test_mask = quantile_static(val=0.1, test=0.2,timestamps=timestamp)
          train_node_stop_indcies = all_nodes[int(temporal_node_num*0.1)]
          val_node_stop_indcies = all_nodes[int(temporal_node_num*0.2)]

        hash_dataframe = copy.deepcopy(items.my_n_id.node.loc[:, ["index", "node"]].values.T)
        hash_table: dict[int, int] = {node: idx for idx, node in zip(*hash_dataframe)}
        
        train_label, train_node = t_labels[:train_node_stop_indcies], all_nodes[: train_node_stop_indcies]
        val_label, val_node = t_labels[train_node_stop_indcies:val_node_stop_indcies], all_nodes[train_node_stop_indcies:val_node_stop_indcies]
        
        train_data = Data(full_data.sources[train_mask], full_data.destinations[train_mask], full_data.timestamps[train_mask],\
                        full_data.edge_idxs[train_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
        train_data.setup_robustness((train_node, train_label))
        
        val_data = Data(full_data.sources[val_mask], full_data.destinations[val_mask], full_data.timestamps[val_mask],\
                        full_data.edge_idxs[val_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
        val_data.setup_robustness((val_node, val_label))
        
        if task in ["imbalance", "fsl"]:
          if task == "imbalance":
            train_ratio, val_ratio = 0.8, 0.2
            if single_graph:  train_ratio, val_ratio = 0.1, 0.1
            transform = Imbalance(train_ratio=train_ratio, ratio=ratio, val_ratio=val_ratio)
            items = transform(items)
            # val_node, train_node are node indices, match with node label in each row
          elif task == "fsl":
            val_ratio = 1
            if single_graph: val_ratio = 0.2
            transform = Few_Shot_Learning(fsl_num=ratio)
            items = transform(items)
          
          node_label = items.my_n_id.node["label"].values
          
          train_label, train_node = node_label[items.train_mask], all_nodes[items.train_mask]
          val_label, val_node = node_label[items.val_mask], all_nodes[items.val_mask]
          nn_val_label, nn_val_node = node_label[items.nn_val_mask], all_nodes[items.nn_val_mask]
          
          train_data.setup_robustness((train_node, train_label))
          val_data.setup_robustness((val_node, val_label), (nn_val_node, nn_val_label))
        
        """
        param t_labels: record ALL the labels in current time-stamp, in shape of (n, )
        param train_mask: record selected optional for edges/labels in current time-stamp (n*0.8, ) or (n_robustness, )
        train_label, train_node: defined by train_mask, showing the node able to be seen
        
        realted of these variables in training:
        during training, node idx is a trival and complex task to handle on. The first problem is we need to setup new-old
        index mapping. Second, mask is necessary to get cooresponding node label. 
        
        What my solution is, during data building, each part will load with labels from full_data, which is the entire timestamp's
        labels. Then, by using [**train_node, val_node, nn_val_node, test_node, nn_test_node**], we can get the cooresponding node allowed
        to be seen. So PLS PAY ATTENTION TO THEM.
        
        Why? becuase if we modify the node idx here based on the mask, you can look at how many layer match we need:
          1. snapshot - entire graph mapping
          2. train_data - snapshot mapping
        we put train_node, val_node in index matched with current snapshot (YEs, node idx within them are reordered), to match
        with label index. Thus, whatever how we setup labels trauncated from full label, we can get coorespond node based on 
        given node from [**train_node, val_node, nn_val_node, test_node, nn_test_node**]. Maintain only first-layer match
        """
        
        if single_graph:
            test_node, test_label = all_nodes[val_node_stop_indcies:], t_labels[val_node_stop_indcies:], 
            test_data = Data(full_data.sources[test_mask], full_data.destinations[test_mask], full_data.timestamps[test_mask],\
                        full_data.edge_idxs[test_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
            test_data.setup_robustness((test_node, test_label))
        else:
            test = graph_list[idxs+1]
            test_data = to_TPPR_Data(test)
            test_node, test_label = test.my_n_id.node["index"].values, test.y
            
            # test_label_missing_mask = test_label != -1
            # test_node, test_label = test_node[test_label_missing_mask], test_label[test_label_missing_mask]
            
            test_data.setup_robustness((test_node, test_label))
        
        if task in ["imbalance", "fsl"]:
          if single_graph:
            test_transform = transform.test_processing(items)
            test_node_label = items.my_n_id.node["label"].values
            test_match_list = (test_node, test_label) # even in robustness task, the test match list is not changed
            nn_test_match_list = (all_nodes[test_transform.nn_test_mask], test_node_label[test_transform.nn_test_mask])
            test_data.setup_robustness(test_match_list, nn_test_match_list)
          else:
            test_transform = transform.test_processing(test)
            test_node_label = test.my_n_id.node["label"].values
            test_match_list = (test_node, test_label) # even in robustness task, the test match list is not changed
            nn_test_match_list = (test_node[test_transform.nn_test_mask], test_node_label[test_transform.nn_test_mask])
            test_data.setup_robustness(test_match_list, nn_test_match_list)
          
        
        # train_data.cover_the_edges(val_data, test_data, single_graph)

        node_num = items.num_nodes
        node_edges = items.num_edges

        TPPR_list.append([full_data, train_data, val_data, test_data, node_num, node_edges])


    return TPPR_list, graph_num_node, graph_feat, edge_number

def batch_processor(data_label: dict[dict[int, np.ndarray]], data: Data)->list[tuple[np.ndarray]]:
  time_stamp = data.timestamps
  unique_ts = np.unique(time_stamp)
  idx_list = np.arange(time_stamp.shape[0])
  time_keys = list(data_label.keys())

  last_idx = 0
  batch_list: list[tuple] = list()
  for ts in unique_ts:
    time_mask = time_stamp==ts
    time_mask[:last_idx] = False
    if ts not in time_keys:
      # print(ts)
      if time_mask.sum() != 0:
        last_idx = idx_list[time_mask][-1]
      continue

    temp_dict = data_label[ts]
    keys = np.array(list(temp_dict.keys()))
    values = np.array(list(temp_dict.values()))

    unique_time_nodes = set(data.sources[time_mask]) | set(data.destinations[time_mask])
    # if len(unique_time_nodes) != len(keys):
    #   print(f"At time {ts}; Under the same timetable the unique node {len(unique_time_nodes)} size and {len(set(keys))} isnt matched")
    #   print(f"The different unique nodes are {unique_time_nodes - set(keys)} \n")

    sort_idx = np.argsort(keys)
    sort_key = keys[sort_idx]
    values = values[sort_idx, :]
    last_idx = idx_list[time_mask][-1]

    backprop_mask = np.isin(np.array(sorted(unique_time_nodes)), sort_key)

    batch_list.append((backprop_mask, values, time_mask))
  return batch_list
  

def TGB_load(train, val, test, node_feat):

  def single_transform(_data):
    _edge_idx = np.arange(_data.src.shape[0])
    transform_data = Data(_data.src.numpy(), _data.dst.numpy(), _data.t.numpy(), \
                      _edge_idx, _data.y.numpy(), hash_table=None)
    transform_data.set_up_features(node_feat, _data.msg.numpy())
    return transform_data
  
  train_data = single_transform(train)
  val_data = single_transform(val)
  test_data = single_transform(test)

  return train_data, val_data, test_data

def batch_processor(data_label: dict[dict[int, np.ndarray]], data: Data)->list[tuple[np.ndarray]]:
  time_stamp = data.timestamps
  unique_ts = np.unique(time_stamp)
  idx_list = np.arange(time_stamp.shape[0])
  time_keys = list(data_label.keys())

  last_idx = 0
  batch_list: list[tuple] = list()
  for ts in unique_ts:
    time_mask = time_stamp==ts
    time_mask[:last_idx] = False
    if ts not in time_keys:
      print(ts)
      last_idx = idx_list[time_mask][-1]
      continue

    temp_dict = data_label[ts]
    keys = np.array(list(temp_dict.keys()))
    values = np.array(list(temp_dict.values()))

    unique_time_nodes = set(data.sources[time_mask]) | set(data.destinations[time_mask])
    if len(unique_time_nodes) != len(keys):
      print(f"At time {ts}; Under the same timetable the unique node {len(unique_time_nodes)} size and {len(set(keys))} isnt matched")
      print(f"The different unique nodes are {unique_time_nodes - set(keys)} \n")

    sort_idx = np.argsort(keys)
    sort_key = keys[sort_idx]
    values = values[sort_idx, :]
    last_idx = idx_list[time_mask][-1]

    backprop_mask = np.isin(np.array(sorted(unique_time_nodes)), sort_key)

    batch_list.append((backprop_mask, values, time_mask))
  return batch_list


# path = "data/mooc/ml_mooc.npy"
# edge = np.load(path)
def load_feat(d):
    node_feats = None
    if os.path.exists('../data/{}/ml_{}_node.npy'.format(d,d)):
        node_feats = np.load('../data/{}/ml_{}_node.npy'.format(d,d)) 

    edge_feats = None
    if os.path.exists('../data/{}/ml_{}.npy'.format(d,d)):
        edge_feats = np.load('../data/{}/ml_{}.npy'.format(d,d))
    return node_feats, edge_feats


############## load a batch of training data ##############
def get_data(dataset_name):
  graph_df = pd.read_csv('data/{}/ml_{}.csv'.format(dataset_name,dataset_name))

  #edge_features = np.load('../data/{}/ml_{}.npy'.format(dataset_name,dataset_name))
  #node_features = np.load('../data/{}/ml_{}_node.npy'.format(dataset_name,dataset_name)) 
  #node_features, edge_features = load_feat(dataset_name)

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values
  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
  
  # ensure we get the same graph
  random.seed(2020)
  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)
  n_edges = len(sources)

  test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
  new_test_node_set = set(random.sample(sorted(test_node_set), int(0.1 * n_total_unique_nodes)))
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0


  # * the val set can indeed contain the new test node
  new_node_set = node_set - train_node_set
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)

  test_mask = timestamps > test_time
  edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
  new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
  new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])
  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])
  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])


  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

  return full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data, n_total_unique_nodes, n_edges


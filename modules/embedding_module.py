import torch
from torch import nn
import numpy as np
from utils.util import tppr_finder
import math
import time
from model.temporal_attention import TemporalAttentionLayer
from numba import njit
import copy


@njit
def numba_unique(nodes):
  return np.unique(nodes)

class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,use_time_proj=True):
    pass


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))
    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, args=None, num_nodes = -1):
    
    super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)
    self.use_memory = use_memory
    self.device = device
    self.args=args
    self.num_nodes=num_nodes
    self.t_tppr=0


class GraphDiffusionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,n_node_features, n_edge_features, n_time_features, embedding_dimension, device,n_heads=2, dropout=0.1, use_memory=True,args=None, num_nodes = -1):
    super(GraphDiffusionEmbedding, self).__init__(node_features, edge_features, memory,
                                            neighbor_finder, time_encoder, n_layers,
                                            n_node_features, n_edge_features,
                                            n_time_features,
                                            embedding_dimension, device,
                                            n_heads, dropout,
                                            use_memory,args,num_nodes)
    self.emb_fea = embedding_dimension
    self.fc1_output = 512
    self.fc2_output = 128

    self.fc1 = torch.nn.Linear(embedding_dimension + n_time_features +n_edge_features, self.fc1_output)

    # -----normalization methods----------------
    self.bc1 = torch.nn.BatchNorm1d(self.fc1_output)
    self.ln1 = torch.nn.LayerNorm(self.fc1_output)
    self.bc2 = torch.nn.BatchNorm1d(self.fc2_output)
    # -----------------------------------------------

    self.fc2 = torch.nn.Linear(self.fc1_output, self.fc2_output)
    self.act = torch.nn.ReLU()
    self.drop = nn.Dropout(0.1)
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

    # for source transformation
    self.fc1_source = torch.nn.Linear(embedding_dimension, self.fc1_output)
    self.bc1_source = torch.nn.BatchNorm1d(self.fc1_output)
    self.fc2_source = torch.nn.Linear(self.fc1_output, self.fc2_output)
    self.drop = nn.Dropout(0.1)
    torch.nn.init.xavier_normal_(self.fc1_source.weight)
    torch.nn.init.xavier_normal_(self.fc2_source.weight)

    # for combination
    self.combiner = torch.nn.Linear(embedding_dimension + embedding_dimension, embedding_dimension)
    self.n_tppr=len(args.alpha_list)
    self.alpha_list=args.alpha_list
    self.beta_list=args.beta_list
    self.k=args.topk
    self.tppr_strategy=self.args.tppr_strategy
    self.width=args.n_degree
    self.depth=args.n_layer
    assert(self.k!=0)

    if self.tppr_strategy=='streaming':
      insert_node = self.num_nodes
      if isinstance(self.num_nodes, torch.Tensor):
        insert_node = self.num_nodes.cpu().numpy()
      self.tppr_finder=tppr_finder(insert_node, self.k, self.n_tppr, self.alpha_list, self.beta_list)

  def reset_tppr(self):
    self.tppr_finder.reset_tppr()
    
  def backup_tppr(self):
    return self.tppr_finder.backup_tppr()

  def restore_tppr(self, backup):
    self.tppr_finder.restore_tppr(backup)


  def streaming_topk(self,source_nodes, timestamps, edge_idxs):
    return self.tppr_finder.streaming_topk(source_nodes,timestamps,edge_idxs)

  def streaming_topk_node(self,source_nodes, timestamps, edge_idxs):
    return self.tppr_finder.streaming_topk_modified(source_nodes, timestamps, edge_idxs)

  def node_last_timestamp(self, source_nodes, timestamps):
    edge_length = source_nodes.shape[0]

    node_time_hashtable: dict[int, float] = {}
    for node, tmp in zip(source_nodes, timestamps):
      if node in node_time_hashtable.keys():
        node_time_hashtable[node] = tmp if tmp > node_time_hashtable[node] else node_time_hashtable[node]
      else:
        node_time_hashtable[node] = tmp
    return node_time_hashtable

  def streaming_topk_extraction(self, source_node: np.ndarray, timestamps):
    """
    to align with processing in node classification, here the negative sampling part will be suspend
    """
    node_len = len(source_node)

    # TODO: A method to keep every nodes last timestamp 
    timetable = self.node_last_timestamp(source_node, copy.deepcopy(timestamps))
    sorted_timetable = dict(sorted(timetable.items()))
    node_timestamp = np.array(list(sorted_timetable.values()))

    sample_node = list(set(source_node))
    assert len(sample_node) == node_timestamp.shape[0], "Sample node and corresponding time stamp is not matched"

    return self.tppr_finder.single_extraction(sample_node, node_timestamp), sample_node

  def streaming_topk_no_fake(self,source_nodes, timestamps, edge_idxs):
    return self.tppr_finder.streaming_topk_no_fake(source_nodes,timestamps,edge_idxs)


  def fill_tppr(self,sources, targets, timestamps, edge_idxs, tppr_filled):
    if tppr_filled:
      self.tppr_finder.restore_val_tppr()
    else:
      self.tppr_finder.compute_val_tppr(sources, targets,timestamps, edge_idxs)

  def compute_embedding_tppr_ensemble(self, memory, source_nodes, timestamps, edge_idxs, memory_updater,train):

    source_nodes=np.array(source_nodes,dtype=np.int32)
    t=time.time()
    if self.tppr_strategy=='streaming': 
      selected_node_list,selected_edge_idxs_list,selected_delta_time_list,selected_weight_list=self.streaming_topk(source_nodes, timestamps, edge_idxs)       
    elif self.tppr_strategy=='pruning':
      selected_node_list,selected_edge_idxs_list,selected_delta_time_list,selected_weight_list=self.pruning_topk(source_nodes, timestamps)
    self.t_tppr+=time.time()-t

    if train:
      memory_nodes=np.hstack(selected_node_list)
      index = numba_unique(memory_nodes)
      memory,_=memory_updater.get_updated_memory(memory,index) 

    n_edge=selected_weight_list[0].shape[0]//3
    self.average_topk=np.mean(np.sum(selected_weight_list[0][:2*n_edge],axis=1))

    ### transfer from CPU to GPU
    for i in range(self.n_tppr):
      selected_node_list[i] = torch.from_numpy(selected_node_list[i]).long().to(self.device,non_blocking=True)
      selected_edge_idxs_list[i] = torch.from_numpy(selected_edge_idxs_list[i]).long().to(self.device,non_blocking=True)
      selected_delta_time_list[i] = torch.from_numpy(selected_delta_time_list[i]).float().to(self.device,non_blocking=True)
      selected_weight_list[i] = torch.from_numpy(selected_weight_list[i]).float().to(self.device,non_blocking=True)

    ### transform source embeddings
    nodes_0 = source_nodes
    nodes_0 = torch.from_numpy(nodes_0).long().to(self.device)
    source_embeddings = memory[nodes_0]
    source_embeddings=self.transform_source(source_embeddings)

    ### get neighbor embeddings
    embeddings=source_embeddings
    for index,selected_nodes in enumerate(selected_node_list):

      # node features
      node_features = memory[selected_nodes]

      # edge features
      selected_edge_idxs=selected_edge_idxs_list[index]
      edge_features = self.edge_features[selected_edge_idxs, :] 
  
      # time encoding
      selected_delta_time=selected_delta_time_list[index]
      time_embeddings = self.time_encoder(selected_delta_time)

      # concat and transform
      neighbor_embeddings = torch.cat([node_features,edge_features,time_embeddings], dim=-1)
      neighbor_embeddings=self.transform(neighbor_embeddings) # [600, X]

      # normalize the weights here, very important step!
      weights=selected_weight_list[index]
      weights_sum=torch.sum(weights,dim=1)
      weights=weights/weights_sum.unsqueeze(1)
      weights[weights_sum==0]=0

      ### concat source embeddings, and neighbor embeddings obtained by different diffusion models
      neighbor_embeddings=neighbor_embeddings*weights[:,:,None]
      neighbor_embeddings=torch.sum(neighbor_embeddings,dim=1)
      embeddings = torch.cat((embeddings,neighbor_embeddings),dim=1)

    return embeddings

  def compute_embedding_tppr_node(self, memory, source_nodes, timestamps, memory_updater, train, lap_memory):
    """
    The difference is, source_nodes here is entire node list instead of edges, be aware that edge_idxs may 
    not be used
    """
    source_nodes=np.array(source_nodes, dtype=np.int32)
    t=time.time()
    (selected_node_list, selected_edge_idxs_list, selected_delta_time_list, selected_weight_list), sample_node = \
      self.streaming_topk_extraction(source_nodes, timestamps)  
         
    self.t_tppr+=time.time()-t

    if train:
      memory_nodes=np.hstack(selected_node_list)
      index = numba_unique(memory_nodes)
      memory,_=memory_updater.get_updated_memory(memory,index) 

    self.average_topk=np.mean(np.sum(selected_weight_list[0],axis=1))
    selected_laplcian_memory = list()

    ### transfer from CPU to GPU
    for i in range(self.n_tppr):
      selected_laplcian_memory.append(lap_memory.get_snapshot_memory(sample_node, selected_node_list[i]))
      selected_node_list[i] = torch.from_numpy(selected_node_list[i]).long().to(self.device,non_blocking=True)
      selected_edge_idxs_list[i] = torch.from_numpy(selected_edge_idxs_list[i]).long().to(self.device,non_blocking=True)
      selected_delta_time_list[i] = torch.from_numpy(selected_delta_time_list[i]).float().to(self.device,non_blocking=True)
      selected_weight_list[i] = torch.from_numpy(selected_weight_list[i]).float().to(self.device,non_blocking=True)

    ### transform source embeddings
    nodes_0 = np.array(sample_node)
    nodes_0 = torch.from_numpy(nodes_0).long().to(self.device)
    source_embeddings = memory[nodes_0]
    source_embeddings=self.transform_source(source_embeddings)

    ### get neighbor embeddings
    embeddings=source_embeddings
    for index,selected_nodes in enumerate(selected_node_list):
      # TODO: Build a function to convert neighbor nodes to adjacency matrix
      
      # node features
      node_features = memory[selected_nodes]
      # edge features
      selected_edge_idxs=selected_edge_idxs_list[index]
      edge_features = self.edge_features[selected_edge_idxs, :] 
  
      # time encoding
      selected_delta_time=selected_delta_time_list[index]
      time_embeddings = self.time_encoder(selected_delta_time)

      # concat and transform
      neighbor_embeddings = torch.cat([node_features,edge_features,time_embeddings], dim=-1)
      neighbor_embeddings=self.transform(neighbor_embeddings) # [600, X]

      lap_weights = torch.from_numpy(selected_laplcian_memory[index]).float().to(self.device,non_blocking=True)
      # lap_weights_sum = torch.from_numpy(np.sum(lap_weights, axis=1)).float().to(self.device)

      # normalize the weights here, very important step!
      weights=selected_weight_list[index]
      # weights = torch.cos(weights) * torch.cos(lap_weights) + torch.sin(weights) * torch.cos(lap_weights)
      weights_sum=torch.sum(weights,dim=1)
      weights=weights/weights_sum.unsqueeze(1)
      weights[weights_sum==0]=0
      
      

      ### concat source embeddings, and neighbor embeddings obtained by different diffusion models
      neighbor_embeddings=neighbor_embeddings*weights[:,:,None]
      neighbor_embeddings=torch.sum(neighbor_embeddings,dim=1)
      embeddings = torch.cat((embeddings,neighbor_embeddings),dim=1)

    return embeddings # nxd 


  def pruning_topk(self,source_nodes, timestamps):
    n_nodes=len(source_nodes)
    node_list = []
    edge_idxs_list = []
    delta_time_list = []
    weight_list = []

    for _ in range(self.n_tppr):
      node_list.append(np.zeros((n_nodes, self.k),dtype=np.int32)) 
      edge_idxs_list.append(np.zeros((n_nodes, self.k),dtype=np.int32)) 
      delta_time_list.append(np.zeros((n_nodes, self.k),dtype=np.float32)) 
      weight_list.append(np.zeros((n_nodes, self.k),dtype=np.float32)) 

    for i,alpha in enumerate(self.alpha_list):
      beta=self.beta_list[i]
      self.neighbor_finder.get_pruned_topk(source_nodes,timestamps,self.width,self.depth,alpha,beta,self.k,node_list[i],edge_idxs_list[i],delta_time_list[i],weight_list[i])

    return node_list,edge_idxs_list,delta_time_list,weight_list

  '''
  def pruning_topk_lower_bound(self,source_nodes, timestamps):
    n_nodes=len(source_nodes)
    node_list = []
    edge_idxs_list = []
    delta_time_list = []
    weight_list = []

    for _ in range(self.n_tppr):
      node_list.append(np.zeros((n_nodes, self.k),dtype=np.int32)) 
      edge_idxs_list.append(np.zeros((n_nodes, self.k),dtype=np.int32)) 
      delta_time_list.append(np.zeros((n_nodes, self.k),dtype=np.float32)) 
      weight_list.append(np.zeros((n_nodes, self.k),dtype=np.float32)) 

    for i,alpha in enumerate(self.alpha_list):
      beta=self.beta_list[i]
      self.neighbor_finder.get_pruned_topk_lower_bound(source_nodes,timestamps,self.width,self.depth,alpha,beta,self.k,node_list[i],edge_idxs_list[i],delta_time_list[i],weight_list[i])

    return node_list,edge_idxs_list,delta_time_list,weight_list
  '''

  def transform_source(self,x):
    h = self.fc1_source(x)
    h = self.act(h)
    h = self.drop(h)
    return self.fc2_source(h)

  def transform(self,x):
    h = self.fc1(x)
    h = self.act(self.transform_bcnorm(h, 1))
    h = self.drop(h)
    return self.transform_bcnorm(self.fc2(h), 2)

  def transform_bcnorm(self, h: torch.Tensor, layer: int):
    midd_shape = h.shape[-2]
    if layer==1:
      h = h.view(-1, self.fc1_output)
      h = self.bc1(h)
      h = h.view(-1, midd_shape, self.fc1_output)
    else: 
      h = h.view(-1, self.fc2_output)
      h = self.bc2(h)
      h = h.view(-1, midd_shape, self.fc2_output)
    return h

  def combine(self,x):
    return self.combiner(x)


  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,neighbor_embeddings,edge_time_embeddings, edge_features, mask):
    source_embedding = None
    return source_embedding



################A collection of less important components ############### 
class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,n_node_features, n_edge_features, n_time_features, embedding_dimension, device,n_heads=2, dropout=0.1, use_memory=True,args=None, num_nodes = -1):

    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory,args,num_nodes)

    self.attention_models = torch.nn.ModuleList(
      [TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])


  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):

    attention_model = self.attention_models[n_layer - 1]

    # shape: [batch_size, n_neighbors]
    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)
    return source_embedding


class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True,args=None, num_nodes = -1):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory,
                                            args=args,
                                            num_nodes=num_nodes)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +n_edge_features, embedding_dimension) for _ in range(n_layers)])
    
    self.linear_2 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,neighbor_embeddings,edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))
    source_features = torch.cat([source_node_features,source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,use_time_proj=True):
    return memory[source_nodes, :]


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True,args=None, num_nodes = -1):

  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                                    args=args,num_nodes=num_nodes)

  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                              args=args,num_nodes=num_nodes)


  elif module_type == "diffusion":
    return GraphDiffusionEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                              args=args,num_nodes=num_nodes)


  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                             args=args,num_nodes=num_nodes)

  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))
















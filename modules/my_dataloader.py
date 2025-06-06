from turtle import pos
from networkx import is_empty
import pandas as pd
import numpy as np
import torch
import random
import math
from torch_geometric.data import Data
import torch_geometric.utils as U
import dgl
import os
from torch_geometric.loader import NeighborLoader
from typing import Union
from multipledispatch import dispatch
import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
from datetime import datetime

MOOC, Mooc_extra = "Temporal_Dataset/act-mooc/act-mooc/", ["mooc_action_features", "mooc_action_labels", "mooc_actions"]
MATHOVERFLOW, MathOverflow_extra = "Temporal_Dataset/mathoverflow/", ["sx-mathoverflow-a2q", "sx-mathoverflow-c2a", "sx-mathoverflow-c2q", "sx-mathoverflow"]
LARGE_SCALE = ["tmall", "tax51"]

class NodeIdxMatching(object):

    """
    Not that appliable on train_mask or node_mask, at least in CLDG train_mask should be aggreated within NeigborLoader
    as seed node, while it is computed manually to present "positive pair"
    """

    def __init__(self, is_df: bool, df_node: pd.DataFrame = None, nodes: np.ndarray = [], label: np.ndarray=[]) -> None:
        """
        self.node: param; pd.Dataframe has 2 columns,\n
        'node': means node index in orginal entire graph\n
        'label': corresponding label
        """
        super(NodeIdxMatching, self).__init__()
        self.is_df = is_df
        if is_df: 
            if not df_node: raise ValueError("df_node is required")
            self.node = df_node
        else:
            if len(nodes) == 0 or len(label) == 0:
                raise ValueError("nodes and label are required")
            elif len(label) != len(nodes):
                raise ValueError("label and nodes should have the same length")
            if not isinstance(nodes, (np.ndarray, list, torch.Tensor)): 
                nodes = list(nodes)
            self.nodes = self.to_numpy(nodes)
            self.node: pd.DataFrame = pd.DataFrame({"node": nodes, "label": label}).reset_index()

    def to_numpy(self, nodes: Union[torch.Tensor, np.array]):
        if isinstance(nodes, torch.Tensor):
            if nodes.device == "cuda:0":
                nodes = nodes.cpu().numpy()
            else: 
                nodes = nodes.numpy()
        return nodes

    def idx2node(self, indices: Union[np.array, torch.Tensor]) -> np.ndarray:
        indices = self.to_numpy(indices)
        node = self.node.node.iloc[indices]
        return node.values
    
    def node2idx(self, node_indices: Union[np.array, torch.Tensor] = None) -> np.ndarray:
        if node_indices is None:
            return np.array(self.node.index)
        node_indices = self.to_numpy(node_indices)
        indices = self.node.node[self.node.node.isin(node_indices)].index
        return indices.values
    
    def edge_restore(self, edges: torch.Tensor, to_tensor: bool = False) -> Union[torch.Tensor, np.array]:
        edges = self.to_numpy(edges)
        df_edges = pd.Series(edges.T).apply(lambda x: x.map(self.node.node))
        if to_tensor: 
            df_edges = torch.tensor(df_edges.values.T)
        return df_edges.values
    
    def edge_replacement(self, df_edge: pd.DataFrame):
        if not isinstance(df_edge, pd.Series):
            df_edge = self.to_numpy(df_edge)
            df_edge = pd.DataFrame(df_edge.T)
        transfor_platform = self.node.node
        transfor_platform = pd.Series(transfor_platform.index, index= transfor_platform.values)
        # given function "map" and data series, iterate through col is the fastest way
        df_edge = df_edge.apply(lambda x: x.map(transfor_platform))
        return df_edge.values.T
    
    def get_label_by_node(self, node_indices: Union[torch.Tensor, list[int], np.ndarray]) -> list:
        node_indices = self.to_numpy(node_indices)
        idx_mask: pd.Series = self.node.node[self.node.node.isin(node_indices)].index
        labels: list = self.node.label[idx_mask].values.tolist()
        return labels
    
    def get_label_by_idx(self, idx: Union[torch.Tensor, list[int], np.ndarray]) -> list:
        idx = self.to_numpy(idx)
        return self.node.label[idx].tolist()
    
    def sample_idx(self, node_indices: Union[torch.Tensor, list[int], np.ndarray]) -> torch.Tensor:
        node_indices = self.to_numpy(node_indices)
        idx_mask: pd.Series = self.node.node[self.node.node.isin(node_indices)].index
        return list(idx_mask.values)

class Temporal_Dataloader(Data):
    """
    an overrided class of Data, to store splitted temporal data and reset their index 
    which support for fast local idx and global idx mapping/fetching
    """
    def __init__(self, nodes: Union[list, np.ndarray], edge_index: np.ndarray,\
                  edge_attr: Union[list|np.ndarray], y: list,\
                    pos: torch.Tensor) -> None:
        
        super(Temporal_Dataloader, self).__init__(x = nodes, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
        self.x = nodes
        self.edge_index = edge_index
        self.ori_edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.kept_train_mask = None
        self.kept_val_mask = None

        self.my_n_id = NodeIdxMatching(False, nodes=self.x, label=self.y)
        self.idx2node = self.my_n_id.node
        self.layer2_n_id: pd.DataFrame = None
    
    def test_fast_sparse_build(self, key: np.ndarray, value: np.ndarray) -> torch.Tensor:
        r"""
        Without considering idx matching, assume that tppr_node idx is consistent with current node idx
        """
        return [[src, dst, value[src, idx]] \
                    for src in range(key.shape[0]) \
                    for idx, dst in enumerate(key[src]) \
                        if dst>0 or value[src, idx] > 0]

    def reverse_idx(self, key: list[list[int]], value: list[list[float]])->list[list[int]]:
        """
        reverse the node index in key back to new sorted node idx
        key:param, a 2-layer nested list stores node grabbed from TPPR list
        """
        indices_sparse_container: list[list[int, int, float]] = []
        for idx, idx_ in enumerate(key):
            weight = value[idx]
            for nid, w in zip(idx_, weight):
                # nid: original node idx need to be replaced
                # w: weight of the node
                if w>0:
                    mask = (self.my_n_id.node.node == nid)
                    # during the tppr updating, becasue this is a progressive incresasing process
                    # current snapshot may not contain the feedback NEIGHBOR node, 
                    # so we need to check if the node is in the current snapshot
                    if mask.sum() == 0:
                        continue
                    new_nid = self.my_n_id.node[mask].values[0][0]
                    indices_sparse_container.append([idx, new_nid, w])
        return indices_sparse_container

    def get_node(self):
        return self.my_n_id.node2idx(None)
    
    def get_edge_index(self):
        return self.my_n_id.edge_replacement(self.edge_index)
    
    def get_Temporalgraph(self):
        self.x = self.get_node()
        self.edge_index = self.get_edge_index()
        return self
    
    def up_report(self):
        reverse: pd.DataFrame = pd.DataFrame(self.idx2node.index, index=self.idx2node.values)
        return reverse.to_dict()
    
    def establish_two_layer_idx_matching(self, large_idx: NodeIdxMatching) -> None:
        if large_idx.node.shape[0] < self.my_n_id.node.shape[0]: # <= or < ?
            raise ValueError("Input object should owns a parent graph of current stored temporal graph")
        self.layer2_n_id = self.my_n_id.node.merge(large_idx.node, on="node", suffixes=["_subTemporal", "_Temporal"])
        return

    def single_layer_mask(self, mask, sub_nodes: torch.Tensor, match_list: pd.DataFrame, large_graph_length = None):
        large_graph_length = self.pos.size(0)
        node_idx = sub_nodes[mask]
        large_graph_idx = match_list.index_Temporal[match_list.index_subTemporal.isin(node_idx)].values

        new_mask = torch.full((large_graph_length, ), False, dtype=bool)
        new_mask[large_graph_idx]=True
        return new_mask

    def mask_adjustment_two_layer_idx(self):
        """
        temporal_data: Temporal_Dataloader object, marked as Data for type declaration
        temporal_data.x: consistent sequence node index of sub temporal graph, from 0 to n
        temporal_data.pos: if dataset is Mathoverflow, then this is default node features compute from pos_encoding
        otherwise it is defualt node features. in RoLAND model pos.size(0) >> x.size(0)
        temporal_data.idx2node: node matching dataframe, based on original node idx to match corresponding index 
        of different layer

        this function aims to solve unmatched node features and sub tmeporal graph size. In RoLAND model input is set
        as total number of nodes; to keep on training the model input must has size n x features; where n is total number of nodes
        """
        mask_train: torch.Tensor = self.train_mask
        mask_val: torch.Tensor = self.val_mask
        nodes = self.x
        match_list: pd.DataFrame = self.layer2_n_id
        self.kept_train_mask, self.kept_val_mask = self.train_mask, self.val_mask
        self.train_mask = self.single_layer_mask(mask_train, nodes, match_list)
        self.val_mask = self.single_layer_mask(mask_val, nodes, match_list)
        return self


class Dynamic_Dataloader(object):
    """
    a class to store a group of temporal dataset, calling with updated event running
    return a temporal data every time
    """
    def __init__(self, data: list[Data], graph: Data) -> None:
        super(Dynamic_Dataloader, self).__init__()
        self.data = data
        self.graph = graph
        self.num_classes = int(self.graph.y.max().item() + 1)
        self.len = len(data)

        self.num_nodes = self.graph.x.shape[0]
        self.num_edges = self.graph.edge_index.shape[-1]
        self.temporal = len(data)

        self.temporal_event = None

    def __getitem__(self, idx)-> Temporal_Dataloader:
        return self.data[idx]
    
    def get_temporal(self) -> Union[Data|Temporal_Dataloader|None]:
        if not self.temporal_event:
            self.update_event()
        return self.temporal_event
    
    def get_T1graph(self, timestamp: int) -> Union[Data|Temporal_Dataloader|None]:
        if timestamp>=self.len-1:
            return self.data[self.len-1]
        if len(self.data) <= 1:
            if self.data.is_empty(): return self.graph
            return self.data[0]
        return self.data[timestamp+1]

    def update_event(self, timestamp: int = -1):
        if timestamp>=self.len-1:
            return
        self.temporal_event = self.data[timestamp+1]


def position_encoding(max_len, emb_size)->torch.Tensor:
    pe = torch.zeros(max_len, emb_size)
    position = torch.arange(0, max_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def load_dblp_interact(path: str = None, dataset: str = "dblp", *wargs) -> pd.DataFrame:
    edges = pd.read_csv(os.path.join("/mnt/d/CodingArea/Python/CLDG/Data/CLDG-datasets/", dataset, '{}.txt'.format(dataset)), sep=' ', names=['src', 'dst', 'time'])
    label = pd.read_csv(os.path.join('/mnt/d/CodingArea/Python/CLDG/Data/CLDG-datasets/', dataset, 'node2label.txt'), sep=' ', names=['node', 'label'])

    return edges, label

def load_mathoverflow_interact(path: str = MATHOVERFLOW, *wargs) -> pd.DataFrame:
    edges = pd.read_csv(os.path.join(path, "sx-mathoverflow"+".txt"), sep=' ', names=['src', 'dst', 'time'])
    label = pd.read_csv(os.path.join(path, "node2label"+".txt"), sep=' ', names=['node', 'label'])

    return edges, label

def get_dataset(path, name: str):
    assert name.lower() in [val.lower() for val in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']]
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS'.lower():
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy'.lower():
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS'.lower():
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers'.lower():
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo'.lower():
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    
    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())

def load_standard(dataset: str, *wargs) -> tuple[Data, NodeIdxMatching]:
    
    path = osp.expanduser('~/datasets')
    path = osp.join(path, dataset)
    dataset = get_dataset(path, dataset)
    return dataset

def load_mathoverflow(path: str = None, dataset: str = "mathoverflow", fea_dim: int = 64, *wargs) -> tuple[Data, NodeIdxMatching]:
    """
    Now this txt file only limited to loading data in from mathoverflow datasets
    path: (path, last three words of dataset) -> (str, str) e.g. ('data/mathoverflow/sx-mathoverflow-a2q.txt', 'a2q')
    node Idx of mathoverflow is not consistent!
    """
    if dataset == "mathoverflow":
        edges, label = load_mathoverflow_interact() if not path else load_mathoverflow_interact(path)
    elif dataset == "dblp":
        edges, label = load_dblp_interact() if not path else load_dblp_interact(path)
    elif dataset in LARGE_SCALE:
        edges, label = load_dblp_interact(dataset=dataset)

    x = label.node.to_numpy()
    nodes = position_encoding(x.max()+1, fea_dim)[x]
    labels = torch.tensor(label.label.to_numpy())

    edge_index = torch.tensor(edges.loc[:, ["src", "dst"]].values.T)
    start_time = edges.time.min()
    edges.time = edges.time.apply(lambda x: x - start_time)
    time = torch.tensor(edges.time)

    graph = Data(x=x, edge_index=edge_index, edge_attr=time, y=labels, pos = nodes)
    # neighborloader = NeighborLoader(graph, num_neighbors=[10, 10], batch_size =2048, shuffle = False)
    idxloader = NodeIdxMatching(False, nodes=x, label=labels)
    return graph, idxloader

def load_padded_dataset(dataset: str = "tmall", fea_dim: int = 64, *wargs) -> tuple[Data, NodeIdxMatching]:
    """
    tmall and tax51 are large scale datasets, but its node label is not completed, so we use
    -1 to pad the node label, and use position encoding to generate node features
    """
    if dataset in LARGE_SCALE:
        edges, label = load_dblp_interact(dataset=dataset)
    else:
        raise ValueError("Dataset not found or not supported")
    
    x = label.node.to_numpy()
    labels = np.array(label.label)

    edge_index = np.array(edges.loc[:, ["src", "dst"]].values.T)
    start_time = edges.time.min()
    edges.time = edges.time.apply(lambda x: x - start_time)
    time = np.array(edges.time)

    max_node, min_node = edge_index.max(), edge_index.min()
    assert min_node == x.min(), "The node index in edge_index is not consistent with label.node"
    full_x = np.arange(min_node, max_node+1)
    full_labels = np.full_like(full_x, -1, dtype=int)
    full_labels[x] = labels
    
    node_feature = position_encoding(max_node+1, fea_dim)

    graph = Data(x=full_x, edge_index=edge_index, edge_attr=time, y=full_labels, pos = node_feature)
    idxloader = NodeIdxMatching(False, nodes=x, label=labels)
    return graph, idxloader

def load_tsv(path: list[tuple[str]], *wargs) -> tuple[pd.DataFrame]:
    """
    Note this function only for loading data in act-mooc dataset
    """
    dfs:dict = {p[1]: pd.read_csv(p[0], sep='\t') for p in path}

    label = dfs["mooc_action_labels"]
    action_features = dfs["mooc_action_features"]
    actions = dfs["mooc_actions"]
    return label, action_features, actions

def load_example():
    return "node_feat", "node_label", "edge_index", "train_indices", "val_indices", "test_indices"

def data_load(dataset: str, for_test:bool=False, **wargs) -> tuple[Data, NodeIdxMatching]:
    dataset = dataset.lower()
    if dataset == "mathoverflow" or dataset == "dblp":
        return load_mathoverflow(dataset=dataset, **wargs)
    elif dataset in ["cora", "citeseer", "wikics"] :
        graph = load_standard(dataset, **wargs)[0]
        if for_test:
            return graph, None
        nodes = [i for i in range(graph.x.shape[0])]
        return graph, NodeIdxMatching(False, nodes=nodes, label=graph.y.numpy())
    elif dataset in LARGE_SCALE:
        return load_padded_dataset(dataset=dataset, **wargs)
    else: 
        raise ValueError("Dataset not found")

class Temporal_Splitting(object):

    def __init__(self, graph: Data) -> None:
        
        super(Temporal_Splitting, self).__init__()
        self.graph = graph 

        if isinstance(self.graph.x, torch.Tensor) and self.graph.x.size(1) > 1:
            self.graph.pos = self.graph.x
            self.graph.x = np.arange(self.graph.x.size(0))

        if self.graph.edge_attr == None:
            self.graph.edge_attr = np.arange(self.graph.edge_index.size(1))

        self.n_id = NodeIdxMatching(False, nodes=self.graph.x, label=self.graph.y)

        self.temporal_list: list[Temporal_Dataloader] = []
    
    @dispatch(int, bool)
    def __getitem__(self, idx: int, is_node:bool = False):
        if is_node:
            return self.tracing_dict[idx]
        return self.temporal_list[idx]
    
    @dispatch(int, int)
    def __getitem__(self, list_idx: int, idx: int):
        return self.temporal_list[list_idx][idx]

    def constrct_tracing_dict(self, temporal_list: list[Temporal_Dataloader]) -> None:
        tracing_dict = {}
        for idx, temporal in enumerate(temporal_list):
            temporal_dict: dict[int: int] = temporal.up_report()
            tracing_dict = {**tracing_dict, **{key: [val, idx] for key, val in temporal_dict.items()}}
        self.tracing_dict = tracing_dict
        return

    def sampling_layer(self, snapshots: int, views: int, span: float, strategy: str):
        T = []
        if strategy == 'random':
            T = [random.uniform(0, span * (snapshots - 1) / snapshots) for _ in range(views)]
        elif strategy == 'low_overlap':
            if (0.75 * views + 0.25) > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
            start = random.uniform(0, span - (0.75 * views + 0.25) * span /  snapshots)
            T = [start + (0.75 * i * span) / snapshots for i in range(views)]
        elif strategy == 'high_overlap':
            if (0.25 * views + 0.75) > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
            start = random.uniform(0, span - (0.25 * views + 0.75) * span /  snapshots)
            T = [start + (0.25 * i * span) / snapshots for i in range(views)]
        elif strategy == 'sequential':
            T = [span * i / (snapshots-1) for i in range(1, snapshots)]
            if views > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
            T = random.sample(T, views)
        T= sorted(T)
        if T[0] == float(0):
            T.pop(0)
        return T

    def sampling_layer_by_time(self, span, duration: int = 30):
        """
        span :param; entire timestamp, expected in Unix timestamp such as 1254192988
        duration: param; how many days
        """
        Times = [datetime.fromtimestamp(stamp).strftime("%Y-%m-%d") for stamp in span]
        start_time = Times[0]

        T_duration: list[int] = []

        for idx, tim in enumerate(span):
            if Times[idx] - start_time >=duration:
                T_duration.append(tim)
                start_time = Times[idx]
        
        return T_duration

    def temporal_splitting(self, time_mode: str, **kwargs) -> list[Data]:
        """
        currently only suitable for CLDG dataset, to present flexibilty of function Data\n
        log 12.3:\n
        Temporal and Any Dynamic data loader will no longer compatable with static graph
        """
        edge_index = self.graph.edge_index
        edge_attr = self.graph.edge_attr
        pos = self.graph.pos

        max_time = max(edge_attr)
        temporal_subgraphs = []

        T: list = []

        if time_mode == "time":
            span = edge_attr.cpu().numpy()
            duration = kwargs["duration"]
            T = self.sampling_layer_by_time(span = span, duration=duration)
        elif time_mode == "view":
            span = (max(edge_attr) - min(edge_attr)).item()
            snapshot, views, strategy = kwargs["snapshot"], kwargs["views"], kwargs["strategy"]
            T = self.sampling_layer(snapshot, views, span, strategy)

        for idx, start in enumerate(T):
            if start<0.01: continue

            sample_time = start

            if kwargs["non_split"]:
                end = min(start + span / snapshot, max_time)
                if idx == 0:
                    end = 10400
                sample_time = (edge_attr <= end) # returns an bool value
            else:
                tem_start = T[idx-1] if idx>0 else 0
                end = T[idx]
                sample_time = ((edge_attr>=tem_start) & (edge_attr<=end))

            sampled_edges = edge_index[:, sample_time]
            sampled_nodes = torch.unique(sampled_edges) # orignal/gobal node index
            y = self.n_id.get_label_by_node(sampled_nodes)
            subpos = pos[self.n_id.sample_idx(sampled_nodes)]

            temporal_subgraph = Temporal_Dataloader(nodes=sampled_nodes, edge_index=sampled_edges, \
                                                    edge_attr=edge_attr[sample_time], y=y, pos=subpos).get_Temporalgraph()
            temporal_subgraphs.append(temporal_subgraph)

        return temporal_subgraphs
    
def to_cuda(graph: Union[Data, Temporal_Dataloader], device:str = "cuda:0"):
    device = torch.device(device)
    if not isinstance(graph.x, torch.Tensor):
        graph.x = torch.tensor(graph.x).to(device)
    if not isinstance(graph.edge_index, torch.Tensor):
        graph.edge_index = torch.tensor(graph.edge_index).to(device)
    if not isinstance(graph.edge_attr, torch.Tensor):
        graph.edge_attr = torch.tensor(graph.edge_attr).to(device)
    if not isinstance(graph.y, torch.Tensor) or graph.y.device != device:
        graph.y = torch.tensor(graph.y).to(device)
    
    pos_x_switch=True
    try:
        graph.pos = graph.pos.to(device)
    except Exception as e:
        pos_x_switch = False
        pass
    if pos_x_switch:
        temp = graph.pos
        graph.x = graph.pos
        graph.pos = temp

    return graph
